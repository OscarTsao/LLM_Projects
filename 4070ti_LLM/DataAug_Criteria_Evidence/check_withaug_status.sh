#!/bin/bash
################################################################################
# WITH-AUG HPO Status Check
################################################################################

echo "=== WITH-AUG HPO STATUS SUMMARY ==="
echo ""

echo "Process Status:"
ps -p 3410686 3410952 3411296 3411566 -o pid,stat,time,cmd --no-headers | while read pid stat time cmd; do
    echo "  PID $pid: $stat ($time)"
done
echo ""

echo "Recent Trial Counts:"
for arch in criteria evidence share joint; do
    echo "  $arch:"
    tail -50 ./logs/with_aug_2025-11-03/${arch}_hpo.log 2>/dev/null | grep -E "Trial [0-9]+ (finished|pruned)" | tail -3 | sed 's/^/    /'
done
echo ""

echo "Augmentation Usage:"
for arch in criteria evidence share joint; do
    aug_count=$(grep -c "'aug.enabled': True" ./logs/with_aug_2025-11-03/${arch}_hpo.log 2>/dev/null || echo 0)
    total_count=$(grep -c "Trial [0-9]+ (finished|pruned|failed)" ./logs/with_aug_2025-11-03/${arch}_hpo.log 2>/dev/null || echo 0)
    echo "  $arch: $aug_count/$total_count trials with augmentation"
done
