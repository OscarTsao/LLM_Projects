#!/bin/bash
# HPO Watchdog - Comprehensive monitoring and automatic recovery
# Monitors resources, progress, and can restart on failures

set -euo pipefail

HPO_LOG="hpo_supermax_run.log"
RESOURCE_LOG="hpo_resources.log"
PROGRESS_LOG="hpo_progress.log"
PID_FILE="hpo_supermax.pid"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$HPO_LOG"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$HPO_LOG"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$HPO_LOG"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}" | tee -a "$HPO_LOG"
}

# Check if HPO process is running
check_process() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0  # Running
        fi
    fi
    return 1  # Not running
}

# Monitor and report system resources
check_resources() {
    # RAM check
    RAM_PERCENT=$(free | grep Mem | awk '{printf "%.0f", ($3/$2) * 100}')
    if [ "$RAM_PERCENT" -gt 90 ]; then
        log_error "RAM usage critical: ${RAM_PERCENT}%"
        return 1
    elif [ "$RAM_PERCENT" -gt 85 ]; then
        log_warning "RAM usage high: ${RAM_PERCENT}%"
    fi

    # GPU check
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{printf "%.0f", ($1/$2)*100}')
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

    if [ "$GPU_MEM" -gt 95 ]; then
        log_error "GPU memory critical: ${GPU_MEM}%"
        return 1
    fi

    log "Resources: RAM=${RAM_PERCENT}%, GPU_MEM=${GPU_MEM}%, GPU_UTIL=${GPU_UTIL}%"
    return 0
}

# Check if trials are progressing
check_progress() {
    if [ ! -f "mlflow.db" ]; then
        log_warning "MLflow DB not found yet"
        return 0
    fi

    TRIAL_COUNT=$(sqlite3 mlflow.db "SELECT COUNT(*) FROM runs;" 2>/dev/null || echo "0")

    if [ -f ".last_trial_count" ]; then
        LAST_COUNT=$(cat .last_trial_count)
        if [ "$TRIAL_COUNT" -eq "$LAST_COUNT" ]; then
            STALL_TIME=$(($(date +%s) - $(stat -c %Y .last_trial_count)))
            if [ "$STALL_TIME" -gt 600 ]; then  # 10 minutes
                log_error "No progress for $((STALL_TIME / 60)) minutes (trials: $TRIAL_COUNT)"
                return 1
            fi
        else
            log_success "Progress: $TRIAL_COUNT trials (+$((TRIAL_COUNT - LAST_COUNT)))"
            echo "$TRIAL_COUNT" > .last_trial_count
        fi
    else
        echo "$TRIAL_COUNT" > .last_trial_count
    fi

    return 0
}

# Main monitoring loop
log "=== HPO Watchdog Started ==="
log "Monitoring: $HPO_LOG"
log "Resources: $RESOURCE_LOG"
log "Progress: $PROGRESS_LOG"

while true; do
    if check_process; then
        log "HPO process alive (PID: $(cat $PID_FILE))"

        if ! check_resources; then
            log_error "Resource check failed! Consider manual intervention."
        fi

        if ! check_progress; then
            log_error "Progress check failed! Possible silent failure."
        fi
    else
        log_warning "HPO process not running or PID file missing"

        # Check if completed successfully
        if grep -q "ALL SUPER-MAX HPO RUNS COMPLETE" "$HPO_LOG" 2>/dev/null; then
            log_success "HPO completed successfully!"
            break
        else
            log_error "HPO process terminated unexpectedly!"
            break
        fi
    fi

    sleep 60  # Check every minute
done

log "=== HPO Watchdog Stopped ==="
