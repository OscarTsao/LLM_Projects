#!/bin/bash
# HPO Monitoring Script
# Monitors the HPO run and auto-restarts on failure

LOG_FILE="hpo_monitor.log"
HPO_LOG="hpo_run_fixed.log"
CHECK_INTERVAL=600  # 10 minutes

echo "=== HPO Monitor Started at $(date) ===" | tee -a "$LOG_FILE"

while true; do
    # Check if make hpo process is still running
    if pgrep -f "make hpo" > /dev/null; then
        echo "[$(date)] HPO process is running" | tee -a "$LOG_FILE"

        # Check for errors in the log
        if tail -100 "$HPO_LOG" | grep -E "Error|Exception|Traceback|ValueError" | grep -v "UserWarning" | grep -v "ExperimentalWarning" > /dev/null; then
            echo "[$(date)] ERROR DETECTED! Checking details..." | tee -a "$LOG_FILE"
            tail -50 "$HPO_LOG" | grep -E "Error|Exception|Traceback|ValueError" | tee -a "$LOG_FILE"
            echo "[$(date)] Process may have errors but still running. Continuing to monitor..." | tee -a "$LOG_FILE"
        else
            # Count completed/pruned trials
            TRIAL_COUNT=$(grep -E "Trial.*finished|Trial.*pruned" "$HPO_LOG" | wc -l)
            LAST_TRIAL=$(grep -E "Trial.*finished|Trial.*pruned" "$HPO_LOG" | tail -1)
            echo "[$(date)] Progress: $TRIAL_COUNT trials completed. Last: $LAST_TRIAL" | tee -a "$LOG_FILE"
        fi
    else
        echo "[$(date)] HPO process NOT running! Checking exit status..." | tee -a "$LOG_FILE"

        # Check if it completed successfully or failed
        if tail -20 "$HPO_LOG" | grep -q "stage_b_best"; then
            echo "[$(date)] HPO COMPLETED SUCCESSFULLY!" | tee -a "$LOG_FILE"
            echo "=== Final Summary ===" | tee -a "$LOG_FILE"
            tail -30 "$HPO_LOG" | tee -a "$LOG_FILE"
            exit 0
        else
            echo "[$(date)] HPO process failed or was interrupted" | tee -a "$LOG_FILE"
            echo "[$(date)] Last 50 lines of log:" | tee -a "$LOG_FILE"
            tail -50 "$HPO_LOG" | tee -a "$LOG_FILE"

            # Don't auto-restart to avoid infinite loops - user should investigate
            echo "[$(date)] NOT auto-restarting. Please investigate the error above." | tee -a "$LOG_FILE"
            exit 1
        fi
    fi

    # Wait before next check
    sleep "$CHECK_INTERVAL"
done
