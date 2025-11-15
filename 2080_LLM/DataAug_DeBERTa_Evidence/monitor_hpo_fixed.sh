#!/bin/bash
# HPO Monitoring Script with OneCycleLR fix
# Monitors the HPO run and reports progress

LOG_FILE="hpo_monitor_fixed.log"
HPO_LOG="hpo_run_scheduler_fix.log"
CHECK_INTERVAL=600  # 10 minutes

echo "=== HPO Monitor Started at $(date) ===" | tee -a "$LOG_FILE"
echo "Monitoring log file: $HPO_LOG" | tee -a "$LOG_FILE"

while true; do
    # Check if make hpo process is still running
    if pgrep -f "python.*console.*tune" > /dev/null; then
        echo "[$(date)] HPO process is running" | tee -a "$LOG_FILE"

        # Check for errors in the log
        if tail -100 "$HPO_LOG" | grep -E "Error|Exception|Traceback" | grep -v "UserWarning" | grep -v "ExperimentalWarning" > /dev/null; then
            echo "[$(date)] ERROR DETECTED! Checking details..." | tee -a "$LOG_FILE"
            tail -50 "$HPO_LOG" | grep -E "Error|Exception|Traceback" | tee -a "$LOG_FILE"
        else
            # Count completed/pruned trials
            TRIAL_COUNT=$(grep -E "Trial.*finished|Trial.*pruned" "$HPO_LOG" | wc -l)
            LAST_TRIAL=$(grep -E "Trial.*finished|Trial.*pruned" "$HPO_LOG" | tail -1)
            BEST_VALUE=$(grep -E "Best is trial" "$HPO_LOG" | tail -1 | grep -oP "value: \K[0-9.]+")

            echo "[$(date)] Progress: $TRIAL_COUNT trials completed" | tee -a "$LOG_FILE"
            if [ -n "$BEST_VALUE" ]; then
                echo "[$(date)] Best value so far: $BEST_VALUE" | tee -a "$LOG_FILE"
            fi
            if [ -n "$LAST_TRIAL" ]; then
                echo "[$(date)] Last trial: $LAST_TRIAL" | tee -a "$LOG_FILE"
            fi
        fi
    else
        echo "[$(date)] HPO process NOT running! Checking exit status..." | tee -a "$LOG_FILE"

        # Check if it completed successfully or failed
        if tail -30 "$HPO_LOG" | grep -qE "stage_b_best|Best trial.*value"; then
            echo "[$(date)] HPO COMPLETED SUCCESSFULLY!" | tee -a "$LOG_FILE"
            echo "=== Final Summary ===" | tee -a "$LOG_FILE"
            tail -50 "$HPO_LOG" | tee -a "$LOG_FILE"
            exit 0
        else
            echo "[$(date)] HPO process stopped. Checking for errors..." | tee -a "$LOG_FILE"
            echo "[$(date)] Last 50 lines of log:" | tee -a "$LOG_FILE"
            tail -50 "$HPO_LOG" | tee -a "$LOG_FILE"

            # Check if it's a recoverable error
            if tail -50 "$HPO_LOG" | grep -q "ZeroDivisionError"; then
                echo "[$(date)] ZeroDivisionError detected - this should have been fixed!" | tee -a "$LOG_FILE"
            fi

            echo "[$(date)] Process stopped. Please investigate." | tee -a "$LOG_FILE"
            exit 1
        fi
    fi

    # Wait before next check
    sleep "$CHECK_INTERVAL"
done
