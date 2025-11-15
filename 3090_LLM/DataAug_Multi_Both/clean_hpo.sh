#!/bin/bash
# Script to clean up HPO study history and start fresh

set -e

echo "🧹 Cleaning up HPO history..."

# Stop any running HPO processes
echo "Stopping running HPO processes..."
pkill -f "python.*train.*hpo" 2>/dev/null || true
if [ -f hpo_production.pid ]; then
    kill $(cat hpo_production.pid) 2>/dev/null || true
    rm -f hpo_production.pid
fi

# Remove Optuna database files
echo "Removing Optuna database files..."
rm -f experiments/hpo_*.db
rm -f experiments/*_test.db
rm -f experiments/final_verification.db
rm -f experiments/fixed_pipeline_test.db

# Clean up trial outputs
echo "Removing trial directories..."
rm -rf experiments/trial_*

# Clean up MLflow runs
echo "Cleaning MLflow runs..."
rm -rf mlruns/*

# Remove log files
echo "Removing log files..."
rm -f hpo_production_run.log

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "You can now start a fresh HPO study:"
echo "  make hpo-test       - Quick test (3 trials, ~15 min)"
echo "  make hpo            - Default run (50 trials, ~4-8 hours)"
echo "  make hpo-production - Production run (500 trials, ~40-80 hours)"
