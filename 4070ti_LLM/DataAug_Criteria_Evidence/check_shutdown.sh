#!/bin/bash

echo "=== WITH-AUG HPO SHUTDOWN SUMMARY ==="
echo ""

echo "Verification - checking all tune_max.py processes:"
ps aux | grep 'tune_max.py.*withaug' | grep -v grep || echo "✓ No WITH-AUG HPO processes running"
echo ""

echo "Final trial counts from logs:"
for arch in criteria evidence share joint; do
    log_file="./logs/with_aug_2025-11-03/${arch}_hpo.log"
    if [ -f "$log_file" ]; then
        finished=$(grep -c "Trial [0-9]* finished" "$log_file" 2>/dev/null || echo 0)
        pruned=$(grep -c "Trial [0-9]* pruned" "$log_file" 2>/dev/null || echo 0)
        failed=$(grep -c "Trial [0-9]* failed" "$log_file" 2>/dev/null || echo 0)
        total=$((finished + pruned + failed))
        echo "  $arch: $total trials attempted ($finished finished, $pruned pruned, $failed failed)"
    fi
done
echo ""

echo "Data preserved in:"
echo "  - Logs: ./logs/with_aug_2025-11-03/"
echo "  - Results: ./_runs/with_aug_2025-11-03/"
echo "  - Databases: ./_optuna/*_with_aug.db"
