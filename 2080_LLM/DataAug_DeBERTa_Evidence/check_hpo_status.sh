#!/bin/bash
# Quick HPO status checker

LOG_FILE="hpo_run_fixed.log"

echo "=== HPO Status Report $(date) ==="
echo ""

# Check if process is running
if pgrep -f "python src/dataaug_multi_both/console.py tune" > /dev/null; then
    echo "✓ HPO Process: RUNNING"
else
    echo "✗ HPO Process: NOT RUNNING"
fi

# Count trials
COMPLETED=$(grep -c "Trial.*finished" "$LOG_FILE" 2>/dev/null || echo "0")
PRUNED=$(grep -c "Trial.*pruned" "$LOG_FILE" 2>/dev/null || echo "0")
TOTAL=$((COMPLETED + PRUNED))

echo "✓ Trials Completed: $COMPLETED"
echo "✓ Trials Pruned: $PRUNED"  
echo "✓ Total Trials: $TOTAL / 380 (Stage A)"

# Get best trial
BEST=$(grep "Best is trial" "$LOG_FILE" | tail -1)
if [ -n "$BEST" ]; then
    echo "✓ $BEST"
fi

# Check for recent errors (excluding warnings)
ERRORS=$(tail -100 "$LOG_FILE" | grep -E "Error|Exception|Traceback" | grep -v "UserWarning" | grep -v "ExperimentalWarning" | grep -v "Error initializing" | grep -v "Manual initialization")
if [ -n "$ERRORS" ]; then
    echo ""
    echo "⚠ Recent Errors Detected:"
    echo "$ERRORS"
else
    echo "✓ No Errors Detected"
fi

# Latest activity
echo ""
echo "Latest Trial Activity:"
grep -E "Trial.*finished|Trial.*pruned" "$LOG_FILE" | tail -3

echo ""
echo "=== End of Status Report ==="
