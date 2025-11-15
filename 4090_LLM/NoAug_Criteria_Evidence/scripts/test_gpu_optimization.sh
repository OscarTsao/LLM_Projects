#!/bin/bash
# Test script for GPU utilization optimization
# Tests the changes to tune_max.py with a small trial run

set -e

echo "=========================================="
echo "GPU Utilization Optimization Test"
echo "=========================================="
echo ""

# Configuration
TEST_TRIALS=10
TEST_PARALLEL=4
TEST_STUDY="noaug-criteria-gpu-opt-test"
TEST_OUTDIR="./_runs_gpu_test"

echo "Test Configuration:"
echo "  Trials: $TEST_TRIALS"
echo "  Parallel: $TEST_PARALLEL"
echo "  Study: $TEST_STUDY"
echo "  Output: $TEST_OUTDIR"
echo ""

# Create test output directory
mkdir -p "$TEST_OUTDIR"

echo "Starting test run..."
echo "Monitor GPU utilization with: watch -n 2 nvidia-smi"
echo ""

# Run test with optimized settings
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
HPO_EPOCHS=100 \
HPO_PATIENCE=20 \
poetry run python scripts/tune_max.py \
    --agent criteria \
    --study "$TEST_STUDY" \
    --n-trials $TEST_TRIALS \
    --parallel $TEST_PARALLEL \
    --outdir "$TEST_OUTDIR" 2>&1 | tee "$TEST_OUTDIR/test_run.log"

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""

# Analyze results
if [ -f "$TEST_OUTDIR/test_run.log" ]; then
    echo "Analyzing results..."

    # Check for OOM errors
    OOM_COUNT=$(grep -c "CUDA out of memory" "$TEST_OUTDIR/test_run.log" || true)
    echo "  OOM errors: $OOM_COUNT"

    # Check for completed trials
    COMPLETE_COUNT=$(grep -c "COMPLETE" "$TEST_OUTDIR/test_run.log" || true)
    echo "  Completed trials: $COMPLETE_COUNT"

    # Check for pruned trials
    PRUNED_COUNT=$(grep -c "PRUNED" "$TEST_OUTDIR/test_run.log" || true)
    echo "  Pruned trials: $PRUNED_COUNT"

    if [ "$OOM_COUNT" -gt 0 ]; then
        echo ""
        echo "WARNING: OOM errors detected!"
        echo "Consider reducing num_workers or parallel trials."
    else
        echo ""
        echo "SUCCESS: No OOM errors detected."
    fi
fi

echo ""
echo "Check detailed results in: $TEST_OUTDIR/test_run.log"
echo ""
echo "Next steps:"
echo "  1. Review GPU utilization during test (should be 95-100%)"
echo "  2. If successful and no OOM, optimization is safe"
echo "  3. Can apply to main HPO run (changes already active)"
echo ""
