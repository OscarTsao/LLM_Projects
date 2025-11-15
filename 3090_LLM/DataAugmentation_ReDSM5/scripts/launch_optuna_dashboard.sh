#!/usr/bin/env bash
# Launch Optuna Dashboard for monitoring hyperparameter optimization

set -e

STORAGE="${1:-sqlite:///outputs/train/optuna/optuna_study.db}"

echo "Starting Optuna Dashboard..."
echo "Storage: ${STORAGE}"
echo "Access Optuna Dashboard at: http://localhost:8080"

optuna-dashboard "${STORAGE}" --host=0.0.0.0 --port=8080
