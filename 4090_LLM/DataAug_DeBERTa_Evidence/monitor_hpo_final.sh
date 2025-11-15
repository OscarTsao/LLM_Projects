#!/bin/bash
# Final HPO Monitoring Script
# Monitors the HPO run with all distribution fixes applied

LOG_FILE="hpo_monitor_final.log"
HPO_LOG="hpo_run_distribution_fix.log"
CHECK_INTERVAL=600  # 10 minutes

echo "=== HPO Monitor Started at $(date) ===" | tee -a "$LOG_FILE"
echo "Monitoring: $HPO_LOG" | tee -a "$LOG_FILE"
echo "All distribution consistency fixes applied:" | tee -a "$LOG_FILE"
echo "  - _suggest_categorical: always uses same choices" | tee -a "$LOG_FILE"
echo "  - _suggest_float: always uses suggest_float" | tee -a "$LOG_FILE"
echo "  - _suggest_int: always uses suggest_int" | tee -a "$LOG_FILE"
echo "  - OneCycleLR: fallback to cosine on errors" | tee -a "$LOG_FILE"

while true; do
    if pgrep -f "python.*console.*tune" > /dev/null; then
        echo "[$(date)] HPO running" | tee -a "$LOG_FILE"

        # Check for errors
        if tail -50 "$HPO_LOG" | grep -E "Error|Exception|Traceback" | grep -v "UserWarning" | grep -v "ExperimentalWarning" > /dev/null; then
            echo "[$(date)] ERROR DETECTED!" | tee -a "$LOG_FILE"
            tail -30 "$HPO_LOG" | grep -E "Error|Exception|Traceback" | tee -a "$LOG_FILE"
        else
            TRIAL_COUNT=$(grep -E "Trial.*finished|Trial.*pruned" "$HPO_LOG" | wc -l)
            BEST_VALUE=$(grep -E "Best is trial" "$HPO_LOG" | tail -1 | grep -oP "value: \K[0-9.]+")
            STAGE_A=$(grep -c "stage_a" "$HPO_LOG")
            STAGE_B=$(grep -c "stage_b" "$HPO_LOG")

            echo "[$(date)] Progress: $TRIAL_COUNT trials" | tee -a "$LOG_FILE"
            [ -n "$BEST_VALUE" ] && echo "[$(date)] Best value: $BEST_VALUE" | tee -a "$LOG_FILE"
            [ $STAGE_B -gt 0 ] && echo "[$(date)] Stage B active!" | tee -a "$LOG_FILE"
        fi
    else
        echo "[$(date)] HPO stopped" | tee -a "$LOG_FILE"

        if tail -30 "$HPO_LOG" | grep -qE "stage_b_best|Study completed"; then
            echo "[$(date)] ✅ HPO COMPLETED!" | tee -a "$LOG_FILE"
            tail -50 "$HPO_LOG" | tee -a "$LOG_FILE"
            exit 0
        else
            echo "[$(date)] Process failed or interrupted" | tee -a "$LOG_FILE"
            tail -50 "$HPO_LOG" | tee -a "$LOG_FILE"
            exit 1
        fi
    fi

    sleep "$CHECK_INTERVAL"
done
