#!/bin/bash
################################################################################
# Evaluate best WITH-AUG configurations on test set
#
# This script uses tune_max.py to run single trials with the best
# hyperparameters found during HPO, then evaluates on test set.
################################################################################

set -e

echo "================================================================================"
echo "          WITH-AUG Test Evaluation - All Architectures"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p ./outputs/test_eval_withaug
mkdir -p ./logs/test_eval_withaug

EPOCHS=100
PATIENCE=20

# Get best trial numbers from summary
CRITERIA_TRIAL=719
EVIDENCE_TRIAL=1044
SHARE_TRIAL=718
JOINT_TRIAL=474

echo "Best Trial Numbers (from HPO):"
echo "  Criteria:  #$CRITERIA_TRIAL"
echo "  Evidence:  #$EVIDENCE_TRIAL"
echo "  Share:     #$SHARE_TRIAL"
echo "  Joint:     #$JOINT_TRIAL"
echo ""

echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Patience: $PATIENCE"
echo "  Mode: Test Evaluation (on held-out test set)"
echo ""

read -p "Continue with test evaluation? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo "================================================================================"
echo "                      Evaluating Criteria"
echo "================================================================================"

python scripts/tune_max.py \
    --agent criteria \
    --study-name "test-eval-criteria-withaug" \
    --trials 1 \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --outdir "./outputs/test_eval_withaug" \
    --storage "sqlite:///./outputs/test_eval_withaug/criteria_test.db" \
    > "./logs/test_eval_withaug/criteria_test.log" 2>&1

echo "✓ Criteria evaluation complete"
echo "  Log: ./logs/test_eval_withaug/criteria_test.log"
echo ""

echo "================================================================================"
echo "                      Summary"
echo "================================================================================"
echo "All evaluations complete. Check logs in ./logs/test_eval_withaug/"
echo "Results in ./outputs/test_eval_withaug/"
echo ""

