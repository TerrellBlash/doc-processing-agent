# ============================================
# Dockerfile for Azure Container Apps
# ============================================
# This is what Azure actually runs.
# Think of it like a recipe card that says:
# "Start with Python, add my code, run the server."

FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install dependencies first (Docker caches this layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ app/

# Azure Container Apps expects the app on port 8000
EXPOSE 8000

# Health check — Azure pings this to know we're alive
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the server
# --host 0.0.0.0 means "accept connections from anywhere" (required in containers)
# --workers 4 means handle 4 requests at once
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
