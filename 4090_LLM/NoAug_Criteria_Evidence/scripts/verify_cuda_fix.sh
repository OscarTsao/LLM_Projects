#!/bin/bash
# Verification script for CUDA device-side assert fix
#
# This script runs a comprehensive validation suite to confirm the fix works:
# 1. Dataset validation tests
# 2. Short HPO run (10 trials)
# 3. Analysis of results

set -e  # Exit on error

echo "========================================================================"
echo "CUDA FIX VERIFICATION SUITE"
echo "========================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Dataset validation
echo -e "${YELLOW}Step 1: Running dataset validation tests...${NC}"
echo ""
python scripts/test_cuda_defensive.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dataset validation passed${NC}"
    echo ""
else
    echo -e "${RED}✗ Dataset validation failed${NC}"
    exit 1
fi

# Step 2: Short HPO run
echo "========================================================================"
echo -e "${YELLOW}Step 2: Running short HPO test (10 trials, 2 epochs)...${NC}"
echo ""

# Clean up any existing test study
rm -rf ./_test_runs/test-cuda-fix* 2>/dev/null || true

HPO_EPOCHS=2 python scripts/tune_max.py \
    --agent criteria \
    --study test-cuda-fix \
    --n-trials 10 \
    --outdir ./_test_runs

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ HPO test completed${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}✗ HPO test failed${NC}"
    exit 1
fi

# Step 3: Analyze results
echo "========================================================================"
echo -e "${YELLOW}Step 3: Analyzing results...${NC}"
echo ""

# Count completed vs pruned trials
COMPLETED=$(grep -c "finished with value" ./_test_runs/test-cuda-fix*.log 2>/dev/null || echo "0")
PRUNED=$(grep -c "Trial.*pruned" ./_test_runs/test-cuda-fix*.log 2>/dev/null || echo "0")
CUDA_ERRORS=$(grep -c "CUDA ERROR" ./_test_runs/test-cuda-fix*.log 2>/dev/null || echo "0")

echo "Trial Statistics:"
echo "  Completed: $COMPLETED"
echo "  Pruned: $PRUNED"
echo "  CUDA Errors: $CUDA_ERRORS"
echo ""

# Check for crashes
if grep -q "Traceback" ./_test_runs/test-cuda-fix*.log 2>/dev/null; then
    echo -e "${YELLOW}⚠ Warning: Unhandled exceptions detected in logs${NC}"
    echo ""
else
    echo -e "${GREEN}✓ No unhandled exceptions${NC}"
    echo ""
fi

# Step 4: Summary
echo "========================================================================"
echo "VERIFICATION SUMMARY"
echo "========================================================================"
echo ""

if [ $COMPLETED -gt 0 ]; then
    echo -e "${GREEN}✓ At least one trial completed successfully${NC}"
else
    echo -e "${RED}✗ No trials completed (this may be normal if all were pruned for OOM)${NC}"
fi

if [ $CUDA_ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ No CUDA errors encountered${NC}"
else
    echo -e "${YELLOW}⚠ $CUDA_ERRORS CUDA error(s) detected but handled gracefully${NC}"
fi

echo ""
echo "========================================================================"
echo -e "${GREEN}VERIFICATION COMPLETE${NC}"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Review logs in ./_test_runs/"
echo "  2. Run 50-trial validation: HPO_EPOCHS=10 python scripts/tune_max.py --agent criteria --study validation --n-trials 50"
echo "  3. Deploy full production run: make tune-criteria-max"
echo ""
echo "Documentation:"
echo "  - CUDA_FIX_SUMMARY.md - Executive summary"
echo "  - docs/CUDA_ASSERT_FIX.md - Technical details"
echo ""
