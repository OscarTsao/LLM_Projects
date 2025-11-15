#!/bin/bash
# Smart HPO Monitor - Continuous monitoring with intelligent alerts
# Checks every 3 minutes, reports detailed status

LOG_FILE="smart_monitor.log"
CHECK_INTERVAL=180  # 3 minutes

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

while true; do
    echo "======================================" | tee -a "$LOG_FILE"
    log "STATUS CHECK $(date '+%Y-%m-%d %H:%M:%S')"

    # Check process
    if [ -f hpo_supermax.pid ]; then
        PID=$(cat hpo_supermax.pid)
        if ps -p $PID > /dev/null 2>&1; then
            RUNTIME=$(ps -p $PID -o etime= | xargs)
            log "✅ HPO Running (PID: $PID, Runtime: $RUNTIME)"
        else
            log "❌ CRITICAL: HPO process died!"
            if grep -q "ALL SUPER-MAX HPO RUNS COMPLETE" hpo_supermax_run.log 2>/dev/null; then
                log "🎉 COMPLETED SUCCESSFULLY!"
                exit 0
            fi
        fi
    fi

    # Check trials
    if [ -f _optuna/noaug.db ]; then
        COMPLETE=$(sqlite3 _optuna/noaug.db "SELECT COUNT(*) FROM trials WHERE state='COMPLETE';" 2>/dev/null)
        RUNNING=$(sqlite3 _optuna/noaug.db "SELECT COUNT(*) FROM trials WHERE state='RUNNING';" 2>/dev/null)
        TOTAL=$(sqlite3 _optuna/noaug.db "SELECT COUNT(*) FROM trials;" 2>/dev/null)

        log "📊 Trials: $COMPLETE complete, $RUNNING running, $TOTAL total"

        # Calculate rate
        if [ -f .last_complete_time ]; then
            LAST_COMPLETE=$(cat .last_complete_count 2>/dev/null || echo "0")
            LAST_TIME=$(cat .last_complete_time)
            CURRENT_TIME=$(date +%s)
            TIME_DIFF=$((CURRENT_TIME - LAST_TIME))

            if [ "$COMPLETE" -gt "$LAST_COMPLETE" ]; then
                TRIALS_DIFF=$((COMPLETE - LAST_COMPLETE))
                RATE=$(awk "BEGIN {printf \"%.2f\", ($TRIALS_DIFF / ($TIME_DIFF / 3600))}")
                log "📈 Rate: $RATE trials/hour (gained $TRIALS_DIFF in last check)"
                echo "$COMPLETE" > .last_complete_count
                echo "$CURRENT_TIME" > .last_complete_time
            fi
        else
            echo "$COMPLETE" > .last_complete_count
            echo $(date +%s) > .last_complete_time
        fi
    fi

    # Check GPU
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{printf "%.0f", ($1/$2)*100}')
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)

    if [ "$GPU_UTIL" -ge 95 ]; then
        log "🎮 GPU: ${GPU_UTIL}% ✅ OPTIMAL | Mem: ${GPU_MEM}% | Temp: ${GPU_TEMP}°C"
    elif [ "$GPU_UTIL" -ge 85 ]; then
        log "🎮 GPU: ${GPU_UTIL}% ⚠️  GOOD | Mem: ${GPU_MEM}% | Temp: ${GPU_TEMP}°C"
    else
        log "🎮 GPU: ${GPU_UTIL}% ❌ LOW | Mem: ${GPU_MEM}% | Temp: ${GPU_TEMP}°C"
    fi

    # Check resources
    RAM_PCT=$(free | grep Mem | awk '{printf "%.0f", ($3/$2)*100}')
    CPU_PCT=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{printf "%.0f", 100-$1}')

    log "💻 Resources: RAM ${RAM_PCT}% | CPU ${CPU_PCT}%"

    # Check for recent errors
    ERROR_COUNT=$(grep -c "Error\|CUDA.*error" hpo_supermax_run.log | tail -1 || echo "0")
    if [ "$ERROR_COUNT" -gt 50 ]; then
        log "⚠️  High error count: $ERROR_COUNT"
    fi

    log ""
    sleep $CHECK_INTERVAL
done
