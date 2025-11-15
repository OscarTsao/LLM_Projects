#!/bin/bash
# HPO Progress monitoring script - tracks trial completion and detects silent failures

LOG_FILE="${1:-hpo_progress.log}"
MLFLOW_DB="mlflow.db"
CHECK_INTERVAL=60  # Check every 60 seconds

echo "=== HPO Progress Monitoring Started at $(date) ===" >> "$LOG_FILE"
echo "Monitoring MLflow DB: $MLFLOW_DB" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

LAST_TRIAL_COUNT=0
STALL_COUNT=0
MAX_STALLS=10  # Alert if no progress for 10 minutes

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Count total runs in MLflow (if DB exists)
    if [ -f "$MLFLOW_DB" ]; then
        TRIAL_COUNT=$(sqlite3 "$MLFLOW_DB" "SELECT COUNT(*) FROM runs;" 2>/dev/null || echo "0")

        # Check for progress
        if [ "$TRIAL_COUNT" -gt "$LAST_TRIAL_COUNT" ]; then
            NEW_TRIALS=$((TRIAL_COUNT - LAST_TRIAL_COUNT))
            echo "[$TIMESTAMP] ✓ Progress: +$NEW_TRIALS trials | Total: $TRIAL_COUNT" >> "$LOG_FILE"
            LAST_TRIAL_COUNT=$TRIAL_COUNT
            STALL_COUNT=0
        else
            STALL_COUNT=$((STALL_COUNT + 1))
            echo "[$TIMESTAMP] - No new trials (stall count: $STALL_COUNT/$MAX_STALLS) | Total: $TRIAL_COUNT" >> "$LOG_FILE"

            if [ "$STALL_COUNT" -ge "$MAX_STALLS" ]; then
                echo "[$TIMESTAMP] ⚠️  WARNING: No progress for $((STALL_COUNT * CHECK_INTERVAL / 60)) minutes! Possible silent failure." >> "$LOG_FILE"
            fi
        fi

        # Get recent run statuses
        RECENT_STATUS=$(sqlite3 "$MLFLOW_DB" "SELECT status, COUNT(*) FROM runs GROUP BY status;" 2>/dev/null || echo "N/A")
        if [ "$RECENT_STATUS" != "N/A" ]; then
            echo "[$TIMESTAMP] Status breakdown: $RECENT_STATUS" >> "$LOG_FILE"
        fi
    else
        echo "[$TIMESTAMP] Waiting for MLflow DB to be created..." >> "$LOG_FILE"
    fi

    # Check for active Python processes
    ACTIVE_PROCS=$(ps aux | grep -E "(tune_max\.py|train_criteria\.py)" | grep -v grep | wc -l)
    echo "[$TIMESTAMP] Active HPO processes: $ACTIVE_PROCS" >> "$LOG_FILE"

    if [ "$ACTIVE_PROCS" -eq 0 ] && [ "$TRIAL_COUNT" -gt 0 ]; then
        echo "[$TIMESTAMP] ⚠️  WARNING: No active HPO processes but trials exist! Possible crash." >> "$LOG_FILE"
    fi

    echo "" >> "$LOG_FILE"
    sleep $CHECK_INTERVAL
done
