#!/bin/bash
# Continuous HPO Monitoring - runs indefinitely until HPO completes
# Checks every 5 minutes and alerts on issues

set -euo pipefail

LOG_FILE="hpo_continuous_monitor.log"
CHECK_INTERVAL=300  # 5 minutes
ALERT_THRESHOLD_FAILURES=15  # Alert if >15% failure rate

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Continuous HPO Monitor Started ==="

while true; do
    # Check if HPO process is still running
    if [ -f hpo_supermax.pid ]; then
        PID=$(cat hpo_supermax.pid)
        if ! ps -p $PID > /dev/null 2>&1; then
            log "⚠️  WARNING: HPO process (PID: $PID) not running!"

            # Check if completed successfully
            if grep -q "ALL SUPER-MAX HPO RUNS COMPLETE" hpo_supermax_run.log 2>/dev/null; then
                log "✅ HPO COMPLETED SUCCESSFULLY!"
                break
            else
                log "❌ HPO TERMINATED UNEXPECTEDLY!"
                log "Check hpo_supermax_run.log for errors"
                break
            fi
        fi
    else
        log "⚠️  No PID file found"
        break
    fi

    # Get trial statistics
    if [ -f _optuna/noaug.db ]; then
        TOTAL=$(sqlite3 _optuna/noaug.db "SELECT COUNT(*) FROM trials;" 2>/dev/null || echo "0")
        COMPLETE=$(sqlite3 _optuna/noaug.db "SELECT COUNT(*) FROM trials WHERE state='COMPLETE';" 2>/dev/null || echo "0")
        RUNNING=$(sqlite3 _optuna/noaug.db "SELECT COUNT(*) FROM trials WHERE state='RUNNING';" 2>/dev/null || echo "0")
        FAILED=$(sqlite3 _optuna/noaug.db "SELECT COUNT(*) FROM trials WHERE state='FAIL';" 2>/dev/null || echo "0")
        PRUNED=$(sqlite3 _optuna/noaug.db "SELECT COUNT(*) FROM trials WHERE state='PRUNED';" 2>/dev/null || echo "0")

        # Calculate failure rate
        NON_PRUNED=$((COMPLETE + RUNNING + FAILED))
        if [ "$NON_PRUNED" -gt 0 ]; then
            FAILURE_RATE=$(awk "BEGIN {printf \"%.1f\", ($FAILED/$NON_PRUNED)*100}")
        else
            FAILURE_RATE="0.0"
        fi

        log "📊 Status: Total=$TOTAL, Complete=$COMPLETE, Running=$RUNNING, Failed=$FAILED ($FAILURE_RATE%), Pruned=$PRUNED"

        # Alert on high failure rate
        if (( $(echo "$FAILURE_RATE > $ALERT_THRESHOLD_FAILURES" | bc -l) )); then
            log "⚠️  HIGH FAILURE RATE: ${FAILURE_RATE}% (threshold: ${ALERT_THRESHOLD_FAILURES}%)"
        fi

        # Check progress
        if [ "$COMPLETE" -gt 0 ]; then
            PROGRESS=$(awk "BEGIN {printf \"%.2f\", ($COMPLETE/5000)*100}")
            log "📈 Criteria Phase Progress: ${PROGRESS}% ($COMPLETE/5000)"
        fi
    fi

    # Check GPU utilization
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{printf "%.0f", ($1/$2)*100}')
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0")

    log "🎮 GPU: ${GPU_UTIL}% util, ${GPU_MEM}% mem, ${GPU_TEMP}°C"

    # Alert on low GPU utilization (should be >70% most of the time)
    if [ "$GPU_UTIL" -lt 70 ]; then
        log "⚠️  WARNING: Low GPU utilization (${GPU_UTIL}%)"
    fi

    # Alert on high temperature
    if [ "$GPU_TEMP" -gt 85 ]; then
        log "⚠️  WARNING: High GPU temperature (${GPU_TEMP}°C)"
    fi

    sleep $CHECK_INTERVAL
done

log "=== Continuous HPO Monitor Stopped ==="
