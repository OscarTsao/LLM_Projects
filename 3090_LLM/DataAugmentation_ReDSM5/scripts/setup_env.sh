#!/usr/bin/env bash
# Setup environment variables for MLflow tracking
# Usage: source scripts/setup_env.sh

# Detect if we're in dev container
if [ -f /.dockerenv ] || [ -n "$DEVCONTAINER" ]; then
    echo "✓ Running in dev container - using container MLflow server"
    export MLFLOW_TRACKING_URI="http://mlflow:5000"
else
    echo "✓ Running on host - using localhost MLflow server"
    export MLFLOW_TRACKING_URI="http://localhost:5000"
fi

# Set other environment variables
export DVC_NO_ANALYTICS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# PostgreSQL settings (if connecting directly)
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_USER=mlflow
export POSTGRES_PASSWORD=mlflow
export POSTGRES_DB=mlflow

echo "✓ Environment variables set:"
echo "  MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"
echo "  DVC_NO_ANALYTICS=$DVC_NO_ANALYTICS"
echo ""
echo "MLflow will now auto-log all training runs."
echo "Access MLflow UI at: ${MLFLOW_TRACKING_URI}"
