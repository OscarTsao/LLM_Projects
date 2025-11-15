#!/bin/bash
################################################################################
# Launch WITH-AUG HPO for All Architectures
#
# This script launches full HPO runs with augmentation ENABLED for all 4
# architectures (Criteria, Evidence, Share, Joint).
#
# IMPORTANT:
# - All existing NO-AUG results are preserved in _runs/maximal_2025-10-31/
# - New WITH-AUG results will be stored in _runs/with_aug_2025-11-03/
# - Separate Optuna databases prevent conflicts
#
# Expected Runtime: 2-3 weeks for all architectures
# Total Trials: 800 + 1200 + 600 + 600 = 3200 trials
#
# Usage:
#   bash scripts/launch_with_aug_hpo.sh
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "          WITH-AUG HPO Launch Script - All Architectures"
echo "================================================================================"
echo ""
echo "This will launch 4 HPO runs with augmentation ENABLED:"
echo "  - Criteria:  800 trials, ~1 week"
echo "  - Evidence: 1200 trials, ~1.5 weeks"
echo "  - Share:     600 trials, ~5 days"
echo "  - Joint:     600 trials, ~5 days"
echo ""
echo "Total expected runtime: 2-3 weeks (runs in parallel)"
echo ""
echo "IMPORTANT: All NO-AUG results are preserved!"
echo "  - NO-AUG:  _runs/maximal_2025-10-31/"
echo "  - WITH-AUG: _runs/with_aug_2025-11-03/"
echo ""
echo "================================================================================"
echo ""

# Create directories
echo "Creating output directories..."
mkdir -p ./_runs/with_aug_2025-11-03
mkdir -p ./logs/with_aug_2025-11-03
mkdir -p ./_optuna
echo "✓ Directories created"
echo ""

# Set common environment variables
export HPO_EPOCHS=100
export HPO_PATIENCE=20

echo "================================================================================"
echo "                      1/4: Launching Criteria HPO"
echo "================================================================================"
nohup python scripts/tune_max.py \
    --agent criteria \
    --study-name "withaug-criteria-max-2025-11-03" \
    --trials 800 \
    --epochs 100 \
    --patience 20 \
    --outdir "./_runs/with_aug_2025-11-03" \
    --storage "sqlite:///./_optuna/criteria_with_aug.db" \
    > "./logs/with_aug_2025-11-03/criteria_hpo.log" 2>&1 &
echo $! > "./logs/with_aug_2025-11-03/criteria_hpo.pid"
CRITERIA_PID=$(cat ./logs/with_aug_2025-11-03/criteria_hpo.pid)
echo "✓ Criteria HPO started (PID: $CRITERIA_PID)"
echo "  Log: ./logs/with_aug_2025-11-03/criteria_hpo.log"
echo "  Study: withaug-criteria-max-2025-11-03"
echo "  Trials: 800, Epochs: 100"
echo ""

sleep 3

echo "================================================================================"
echo "                      2/4: Launching Evidence HPO"
echo "================================================================================"
nohup python scripts/tune_max.py \
    --agent evidence \
    --study-name "withaug-evidence-max-2025-11-03" \
    --trials 1200 \
    --epochs 100 \
    --patience 20 \
    --outdir "./_runs/with_aug_2025-11-03" \
    --storage "sqlite:///./_optuna/evidence_with_aug.db" \
    > "./logs/with_aug_2025-11-03/evidence_hpo.log" 2>&1 &
echo $! > "./logs/with_aug_2025-11-03/evidence_hpo.pid"
EVIDENCE_PID=$(cat ./logs/with_aug_2025-11-03/evidence_hpo.pid)
echo "✓ Evidence HPO started (PID: $EVIDENCE_PID)"
echo "  Log: ./logs/with_aug_2025-11-03/evidence_hpo.log"
echo "  Study: withaug-evidence-max-2025-11-03"
echo "  Trials: 1200, Epochs: 100"
echo ""

sleep 3

echo "================================================================================"
echo "                      3/4: Launching Share HPO"
echo "================================================================================"
nohup python scripts/tune_max.py \
    --agent share \
    --study-name "withaug-share-max-2025-11-03" \
    --trials 600 \
    --epochs 100 \
    --patience 20 \
    --outdir "./_runs/with_aug_2025-11-03" \
    --storage "sqlite:///./_optuna/share_with_aug.db" \
    > "./logs/with_aug_2025-11-03/share_hpo.log" 2>&1 &
echo $! > "./logs/with_aug_2025-11-03/share_hpo.pid"
SHARE_PID=$(cat ./logs/with_aug_2025-11-03/share_hpo.pid)
echo "✓ Share HPO started (PID: $SHARE_PID)"
echo "  Log: ./logs/with_aug_2025-11-03/share_hpo.log"
echo "  Study: withaug-share-max-2025-11-03"
echo "  Trials: 600, Epochs: 100"
echo ""

sleep 3

echo "================================================================================"
echo "                      4/4: Launching Joint HPO"
echo "================================================================================"
nohup python scripts/tune_max.py \
    --agent joint \
    --study-name "withaug-joint-max-2025-11-03" \
    --trials 600 \
    --epochs 100 \
    --patience 20 \
    --outdir "./_runs/with_aug_2025-11-03" \
    --storage "sqlite:///./_optuna/joint_with_aug.db" \
    > "./logs/with_aug_2025-11-03/joint_hpo.log" 2>&1 &
echo $! > "./logs/with_aug_2025-11-03/joint_hpo.pid"
JOINT_PID=$(cat ./logs/with_aug_2025-11-03/joint_hpo.pid)
echo "✓ Joint HPO started (PID: $JOINT_PID)"
echo "  Log: ./logs/with_aug_2025-11-03/joint_hpo.log"
echo "  Study: withaug-joint-max-2025-11-03"
echo "  Trials: 600, Epochs: 100"
echo ""

# Wait for initialization
echo "Waiting 15s for all processes to initialize..."
sleep 15
echo ""

# Verify all processes are running
echo "================================================================================"
echo "                      Process Status Verification"
echo "================================================================================"

all_running=true

if ps -p $CRITERIA_PID > /dev/null 2>&1; then
    echo "✓ Criteria HPO running (PID: $CRITERIA_PID)"
else
    echo "✗ Criteria HPO failed to start!"
    all_running=false
fi

if ps -p $EVIDENCE_PID > /dev/null 2>&1; then
    echo "✓ Evidence HPO running (PID: $EVIDENCE_PID)"
else
    echo "✗ Evidence HPO failed to start!"
    all_running=false
fi

if ps -p $SHARE_PID > /dev/null 2>&1; then
    echo "✓ Share HPO running (PID: $SHARE_PID)"
else
    echo "✗ Share HPO failed to start!"
    all_running=false
fi

if ps -p $JOINT_PID > /dev/null 2>&1; then
    echo "✓ Joint HPO running (PID: $JOINT_PID)"
else
    echo "✗ Joint HPO failed to start!"
    all_running=false
fi

echo ""
echo "================================================================================"
echo "                          Launch Summary"
echo "================================================================================"

if [ "$all_running" = true ]; then
    echo "✅ SUCCESS: All 4 HPO runs launched successfully!"
else
    echo "⚠️  WARNING: Some HPO runs failed to start. Check logs above."
fi

echo ""
echo "Study Information:"
echo "  - Criteria:  withaug-criteria-max-2025-11-03  (800 trials)"
echo "  - Evidence:  withaug-evidence-max-2025-11-03  (1200 trials)"
echo "  - Share:     withaug-share-max-2025-11-03     (600 trials)"
echo "  - Joint:     withaug-joint-max-2025-11-03     (600 trials)"
echo ""
echo "Output Directories:"
echo "  - Results:   ./_runs/with_aug_2025-11-03/"
echo "  - Logs:      ./logs/with_aug_2025-11-03/"
echo "  - Optuna DB: ./_optuna/*_with_aug.db"
echo ""
echo "Monitoring Commands:"
echo "  # Check process status"
echo "  ps -p $CRITERIA_PID $EVIDENCE_PID $SHARE_PID $JOINT_PID"
echo ""
echo "  # Monitor Criteria progress"
echo "  tail -f ./logs/with_aug_2025-11-03/criteria_hpo.log | grep -E 'Trial|Best|finished'"
echo ""
echo "  # Monitor Evidence progress"
echo "  tail -f ./logs/with_aug_2025-11-03/evidence_hpo.log | grep -E 'Trial|Best|finished'"
echo ""
echo "  # Monitor Share progress"
echo "  tail -f ./logs/with_aug_2025-11-03/share_hpo.log | grep -E 'Trial|Best|finished'"
echo ""
echo "  # Monitor Joint progress"
echo "  tail -f ./logs/with_aug_2025-11-03/joint_hpo.log | grep -E 'Trial|Best|finished'"
echo ""
echo "  # View all active HPO processes"
echo "  ps aux | grep 'tune_max.py' | grep -v grep"
echo ""
echo "  # Stop all HPO runs (if needed)"
echo "  kill $CRITERIA_PID $EVIDENCE_PID $SHARE_PID $JOINT_PID"
echo ""
echo "Expected Completion: 2-3 weeks from now"
echo ""
echo "================================================================================"
echo "                   WITH-AUG HPO Launch Complete!"
echo "================================================================================"
