#!/usr/bin/bash
# Fast HPO Test - Verify augmentation integration
# Completes in ~5-10 minutes

echo "======================================================================="
echo "FAST HPO TEST - Augmentation Integration Verification"
echo "======================================================================="
echo ""
echo "Configuration:"
echo "  Trials: 2"
echo "  Epochs per trial: 2"
echo "  Training samples: 100"
echo "  Validation samples: 50"
echo "  Expected time: ~5-10 minutes"
echo ""

# Set environment variables for speed
export HPO_EPOCHS=2
export HPO_PATIENCE=2

cd /media/cvrlab308/cvrlab308_4090/YuNing/DataAug_Criteria_Evidence

# Run minimal HPO test
echo "Starting HPO test..."
python scripts/tune_max.py \
    --agent criteria \
    --study aug-criteria-fast-test \
    --n-trials 2 \
    --parallel 1 \
    --outdir outputs \
    2>&1 | tee hpo_fast_test.log

echo ""
echo "======================================================================="
echo "Test completed. Checking results..."
echo "======================================================================="

# Check for augmentation in logs
echo ""
echo "Augmentation parameters sampled:"
grep -i "aug\." hpo_fast_test.log | head -10 || echo "No augmentation parameters found (check logs)"

# Check outputs
echo ""
echo "Output files:"
ls -lh outputs/hpo_maximal/criteria/*.yaml outputs/hpo_maximal/criteria/*.json 2>/dev/null || echo "No output files yet"

echo ""
echo "To view MLflow results:"
echo "  mlflow ui --backend-store-uri sqlite:///mlflow.db"
echo ""
