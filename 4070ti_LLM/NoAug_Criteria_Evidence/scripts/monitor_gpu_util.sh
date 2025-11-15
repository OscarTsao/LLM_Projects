#!/bin/bash
# Monitor GPU utilization during HPO
# Shows GPU util, memory, and running trials

echo "GPU Utilization Monitor"
echo "Press Ctrl+C to exit"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

while true; do
    clear
    echo "=========================================="
    echo "GPU Utilization Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""

    # GPU stats
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits | while IFS=, read -r idx name gpu_util mem_util mem_used mem_total temp; do
        # Color-code GPU utilization
        if [ "$gpu_util" -ge 95 ]; then
            COLOR=$GREEN
        elif [ "$gpu_util" -ge 80 ]; then
            COLOR=$YELLOW
        else
            COLOR=$RED
        fi

        echo -e "  GPU $idx: ${COLOR}${gpu_util}%${NC} util | ${mem_used}MB / ${mem_total}MB | ${temp}°C"
    done

    echo ""
    echo "Running Trials:"
    # Count active trials
    TRIAL_COUNT=$(ps aux | grep -c "[p]ython.*tune_max.py" || echo "0")
    echo "  Active processes: $TRIAL_COUNT"

    # Show recent trial activity from Optuna DB
    if [ -f "./_optuna/noaug.db" ]; then
        echo ""
        echo "Recent Trial Activity:"
        sqlite3 ./_optuna/noaug.db "
            SELECT
                trial_id,
                state,
                CASE
                    WHEN datetime_complete IS NULL THEN 'RUNNING (' || CAST((julianday('now') - julianday(datetime_start)) * 24 * 60 AS INTEGER) || ' min)'
                    ELSE state || ' (' || CAST((julianday(datetime_complete) - julianday(datetime_start)) * 24 * 60 AS INTEGER) || ' min)'
                END as status
            FROM trials
            WHERE study_id = (SELECT study_id FROM studies WHERE study_name = 'noaug-criteria-supermax')
            ORDER BY trial_id DESC
            LIMIT 5
        " 2>/dev/null | while IFS='|' read -r trial_id state status; do
            echo "  Trial $trial_id: $status"
        done

        # Trial statistics
        echo ""
        echo "Trial Statistics:"
        sqlite3 ./_optuna/noaug.db "
            SELECT
                state,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM trials WHERE study_id = (SELECT study_id FROM studies WHERE study_name = 'noaug-criteria-supermax')), 1) as percentage
            FROM trials
            WHERE study_id = (SELECT study_id FROM studies WHERE study_name = 'noaug-criteria-supermax')
            GROUP BY state
            ORDER BY count DESC
        " 2>/dev/null | while IFS='|' read -r state count percentage; do
            echo "  $state: $count ($percentage%)"
        done
    fi

    echo ""
    echo "=========================================="
    echo "Target: 95-100% GPU utilization"
    echo "Safe GPU Memory: < 22GB / 24GB"
    echo "=========================================="

    sleep 5
done
