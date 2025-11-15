#!/bin/bash
# Docker Troubleshooting Script for DataAugmentation_ReDSM5
# This script helps resolve common Docker issues

set -e

echo "🔍 DataAugmentation_ReDSM5 Docker Troubleshooting"
echo "=================================================="
echo ""

# Check for host MLflow availability
echo "1. Checking for MLflow server on port 5000..."
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 || ss -tlnp 2>&1 | grep -q :5000; then
    echo "✅ Port 5000 is active. Ensure this is your external MLflow server."
else
    echo "⚠️  No process is listening on port 5000."
    echo "    Start your external MLflow instance (e.g., 'docker compose up' in the MLflow repo)."
fi

echo ""
echo "2. Checking for old redsm5 containers..."
OLD_CONTAINERS=$(docker ps -a --filter "name=redsm5" --format "{{.Names}}" || true)
if [ -n "$OLD_CONTAINERS" ]; then
    echo "Found old containers:"
    docker ps -a --filter "name=redsm5" --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
    echo ""
    read -p "Remove old redsm5 containers? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker rm -f redsm5-dev 2>/dev/null || true
        echo "✅ Removed old containers"
    fi
else
    echo "✅ No old containers found"
fi

echo ""
echo "3. Verifying .env file..."
if [ -f ".devcontainer/.env" ]; then
    echo "✅ .devcontainer/.env exists"
    cat .devcontainer/.env
else
    echo "⚠️  Creating .devcontainer/.env..."
    echo "localWorkspaceFolderBasename=DataAugmentation_ReDSM5" > .devcontainer/.env
    echo "✅ Created .devcontainer/.env"
fi

echo ""
echo "4. Ready to start containers!"
echo "Run: make docker-up"
echo ""
echo "If you still have issues, run: make docker-clean && make docker-up"
