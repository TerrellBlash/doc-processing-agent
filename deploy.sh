#!/bin/bash
# ============================================
# DEPLOY TO AZURE CONTAINER APPS
# ============================================
# Creates everything on the first run; later runs reuse what exists and
# roll the app to a new image built from the current commit.
#
# What it creates, in resource group doc-agent-rg:
#   - a Basic container registry for the image
#   - a Postgres Flexible Server (Burstable B1ms) for paused reviews
#   - a Container Apps environment and the app itself, which scales to
#     zero when idle (the first request after a quiet spell takes ~15 s)
#
# The API requires an X-API-Key header. The key and the database password
# are generated on the first run and kept in $SECRETS_FILE (mode 600).
# Nothing secret is printed.
#
# Prerequisites:
#   1. Azure CLI installed (brew install azure-cli) and logged in (az login)
#   2. OPENAI_API_KEY exported
#
# Usage: ./deploy.sh          (LOCATION=westus2 ./deploy.sh for another region)

set -euo pipefail

: "${OPENAI_API_KEY:?Export OPENAI_API_KEY first}"

LOCATION="${LOCATION:-eastus}"
RESOURCE_GROUP="doc-agent-rg"
APP_ENV="doc-agent-env"
APP_NAME="doc-agent"
IMAGE_NAME="doc-processing-agent"
DB_NAME="docagent"
DB_ADMIN="docagent"
SECRETS_FILE="${HOME}/.config/doc-agent/azure.env"

# Registry and database server names must be unique across all of Azure.
# A suffix from the subscription ID keeps them stable between runs.
SUFFIX=$(az account show --query id -o tsv | shasum | cut -c1-6)
REGISTRY="docagent${SUFFIX}"
DB_SERVER="doc-agent-db-${SUFFIX}"

# Ship exactly what's committed: uncommitted or untracked files never
# reach the image.
TAG=$(git rev-parse --short HEAD)
BUILD_DIR=$(mktemp -d)
trap 'rm -rf "$BUILD_DIR"' EXIT
git archive HEAD | tar -x -C "$BUILD_DIR"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "Note: uncommitted changes are not deployed; building commit $TAG."
fi

echo "Deploying commit $TAG to $LOCATION"

# --- Secrets, generated once ---
if [ ! -f "$SECRETS_FILE" ]; then
    mkdir -p "$(dirname "$SECRETS_FILE")"
    chmod 700 "$(dirname "$SECRETS_FILE")"
    (
        umask 077
        echo "DOC_AGENT_API_KEY=$(openssl rand -hex 32)" > "$SECRETS_FILE"
        echo "DB_PASSWORD=$(openssl rand -hex 24)" >> "$SECRETS_FILE"
    )
fi
# shellcheck source=/dev/null
source "$SECRETS_FILE"

echo "Preparing the subscription..."
az extension add --name containerapp --upgrade --only-show-errors
for provider in Microsoft.App Microsoft.OperationalInsights Microsoft.ContainerRegistry Microsoft.DBforPostgreSQL; do
    az provider register --namespace "$provider" --wait --only-show-errors
done

echo "Resource group..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" -o none

echo "Container registry..."
if ! az acr show --name "$REGISTRY" -o none 2>/dev/null; then
    az acr create --resource-group "$RESOURCE_GROUP" --name "$REGISTRY" --sku Basic -o none
fi

echo "Building the image in Azure..."
az acr build --registry "$REGISTRY" --image "$IMAGE_NAME:$TAG" "$BUILD_DIR" -o none

# Public access is limited to Azure services (0.0.0.0) and the server
# requires TLS. A private network would be tighter but costs more.
# Limitation: the admin password passes through this script's argv while
# `az` runs, so `ps` on this machine can see it for that moment.
echo "Postgres (first run takes about 5 minutes)..."
if ! az postgres flexible-server show --resource-group "$RESOURCE_GROUP" --name "$DB_SERVER" -o none 2>/dev/null; then
    az postgres flexible-server create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$DB_SERVER" \
        --location "$LOCATION" \
        --tier Burstable \
        --sku-name Standard_B1ms \
        --storage-size 32 \
        --version 16 \
        --admin-user "$DB_ADMIN" \
        --admin-password "$DB_PASSWORD" \
        --public-access 0.0.0.0 \
        --yes -o none
    az postgres flexible-server db create \
        --resource-group "$RESOURCE_GROUP" \
        --server-name "$DB_SERVER" \
        --database-name "$DB_NAME" -o none
fi
DATABASE_URL="postgresql://${DB_ADMIN}:${DB_PASSWORD}@${DB_SERVER}.postgres.database.azure.com:5432/${DB_NAME}?sslmode=require"

echo "Container Apps environment..."
if ! az containerapp env show --resource-group "$RESOURCE_GROUP" --name "$APP_ENV" -o none 2>/dev/null; then
    az containerapp env create --resource-group "$RESOURCE_GROUP" --name "$APP_ENV" --location "$LOCATION" -o none
fi

# Secrets are Container Apps secrets referenced with secretref:, never
# literal env vars, which anyone who can read the app's configuration
# could see. For production, keep them in Key Vault (keyvaultref:).
IMAGE="$REGISTRY.azurecr.io/$IMAGE_NAME:$TAG"
SECRETS=(
    "openai-api-key=$OPENAI_API_KEY"
    "database-url=$DATABASE_URL"
    "api-key=$DOC_AGENT_API_KEY"
)
echo "App..."
if ! az containerapp show --resource-group "$RESOURCE_GROUP" --name "$APP_NAME" -o none 2>/dev/null; then
    az containerapp create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$APP_NAME" \
        --environment "$APP_ENV" \
        --image "$IMAGE" \
        --registry-server "$REGISTRY.azurecr.io" \
        --registry-identity system \
        --target-port 8000 \
        --ingress external \
        --min-replicas 0 \
        --max-replicas 2 \
        --cpu 1.0 \
        --memory 2.0Gi \
        --secrets "${SECRETS[@]}" \
        --env-vars \
            OPENAI_API_KEY=secretref:openai-api-key \
            DATABASE_URL=secretref:database-url \
            DOC_AGENT_API_KEY=secretref:api-key \
        -o none
else
    az containerapp secret set --resource-group "$RESOURCE_GROUP" --name "$APP_NAME" --secrets "${SECRETS[@]}" -o none
    az containerapp update --resource-group "$RESOURCE_GROUP" --name "$APP_NAME" --image "$IMAGE" -o none
fi

APP_URL=$(az containerapp show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$APP_NAME" \
    --query "properties.configuration.ingress.fqdn" \
    --output tsv)

echo ""
echo "Deployed commit $TAG to https://$APP_URL"
echo "API key: DOC_AGENT_API_KEY in $SECRETS_FILE"
echo ""
echo "Try it:"
echo "  curl https://$APP_URL/health"
echo "  source $SECRETS_FILE"
echo "  curl -X POST https://$APP_URL/process -H \"X-API-Key: \$DOC_AGENT_API_KEY\" \\"
echo "    -H 'Content-Type: application/json' -d '{\"document_text\": \"INVOICE #123 ...\"}'"
