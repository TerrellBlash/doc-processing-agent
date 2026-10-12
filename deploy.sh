#!/bin/bash
# ============================================
# DEPLOY TO AZURE CONTAINER APPS
# ============================================
# Run this once to set up everything on Azure.
# After that, just push new Docker images.
#
# Prerequisites:
#   1. Azure CLI installed (brew install azure-cli)
#   2. Logged in (az login)
#   3. OPENAI_API_KEY set as env variable
#
# Usage: ./deploy.sh

set -e  # Stop on any error

# --- Configuration (edit these) ---
RESOURCE_GROUP="doc-agent-rg"
LOCATION="eastus"                    # Close to Atlanta for low latency
CONTAINER_REGISTRY="docagentacr"     # Must be globally unique
CONTAINER_APP_ENV="doc-agent-env"
CONTAINER_APP_NAME="doc-agent"
IMAGE_NAME="doc-processing-agent"

echo "🚀 Deploying Intelligent Document Processing Agent to Azure"

# 1. Create Resource Group
echo "📦 Creating resource group..."
az group create \
    --name $RESOURCE_GROUP \
    --location $LOCATION

# 2. Create Container Registry (where your Docker image lives)
echo "🏗️  Creating container registry..."
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_REGISTRY \
    --sku Basic

# 3. Build and push Docker image to Azure
echo "🐳 Building and pushing Docker image..."
az acr build \
    --registry $CONTAINER_REGISTRY \
    --image $IMAGE_NAME:latest \
    .

# 4. Create Container Apps Environment
echo "🌍 Creating Container Apps environment..."
az containerapp env create \
    --name $CONTAINER_APP_ENV \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION

# 5. Deploy the Container App
#
# DATABASE_URL must already be exported before running this script, and
# should point at a provisioned Postgres instance (e.g. Azure Database
# for PostgreSQL Flexible Server) — never a hardcoded credential here.
# Omitting it makes the deployed service silently fall back to
# in-memory checkpointing (see app/graph.py:build_checkpointer), which
# defeats the purpose of running --workers 4: a paused human_review
# would only survive in the one worker process that created it.
#
# Both secrets are registered with `az containerapp create --secrets`
# and referenced via `secretref:`, not passed as literal --env-vars
# values. A literal env var is stored in plain text in the container
# app's configuration, visible to anyone who can read the app in the
# portal or via `az containerapp show`. A Container Apps secret is
# stored separately and only resolved inside the container.
# Limitation: the values still pass through this script's argv while
# `az` runs, so `ps` on this machine can see them for that moment.
# For production, store them in Azure Key Vault and pass
# `keyvaultref:` secrets instead.
echo "🚢 Deploying container app..."
az containerapp create \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --environment $CONTAINER_APP_ENV \
    --image "$CONTAINER_REGISTRY.azurecr.io/$IMAGE_NAME:latest" \
    --registry-server "$CONTAINER_REGISTRY.azurecr.io" \
    --target-port 8000 \
    --ingress external \
    --min-replicas 1 \
    --max-replicas 5 \
    --cpu 1.0 \
    --memory 2.0Gi \
    --secrets \
        openai-api-key="$OPENAI_API_KEY" \
        database-url="$DATABASE_URL" \
    --env-vars \
        OPENAI_API_KEY=secretref:openai-api-key \
        DATABASE_URL=secretref:database-url

# 6. Get the URL
APP_URL=$(az containerapp show \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query "properties.configuration.ingress.fqdn" \
    --output tsv)

echo ""
echo "✅ Deployment complete!"
echo "🌐 Your agent is live at: https://$APP_URL"
echo ""
echo "Test it:"
echo "  curl https://$APP_URL/health"
echo ""
echo "Process a document:"
echo '  curl -X POST https://$APP_URL/process \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"document_text": "INVOICE #123...", "document_filename": "test.pdf"}'"'"''
