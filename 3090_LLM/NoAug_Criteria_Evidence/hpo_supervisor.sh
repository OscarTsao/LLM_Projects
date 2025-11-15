#!/bin/bash
# HPO Supervisor - Comprehensive monitoring with auto-recovery
# Monitors: GPU utilization, resource usage, progress, silent failures
# Actions: Alerts, auto-restart on crashes, resource management

set -euo pipefail

LOG_FILE="hpo_supervisor.log"
CHECK_INTERVAL=120  # Check every 2 minutes

# Thresholds
RAM_CRITICAL=90
CPU_CRITICAL=95
GPU_LOW_THRESHOLD=70
GPU_HIGH_THRESHOLD=100
STALL_THRESHOLD=10  # Minutes without progress

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}" | tee -a "$LOG_FILE"
}

# Check if process is alive
check_process() {
    if [ -f hpo_supermax.pid ]; then
        PID=$(cat hpo_supermax.pid)
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Check system resources
check_resources() {
    local issues=0

    # RAM check
    RAM_PERCENT=$(free | grep Mem | awk '{printf "%.0f", ($3/$2) * 100}')
    if [ "$RAM_PERCENT" -gt "$RAM_CRITICAL" ]; then
        log_error "RAM CRITICAL: ${RAM_PERCENT}% (limit: ${RAM_CRITICAL}%)"
        issues=$((issues + 1))
    elif [ "$RAM_PERCENT" -gt 80 ]; then
        log_warning "RAM HIGH: ${RAM_PERCENT}%"
    fi

    # CPU check
    CPU_PERCENT=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{printf "%.0f", 100 - $1}')
    if [ "$CPU_PERCENT" -gt "$CPU_CRITICAL" ]; then
        log_error "CPU CRITICAL: ${CPU_PERCENT}% (limit: ${CPU_CRITICAL}%)"
        issues=$((issues + 1))
    fi

    # GPU check
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0")
    GPU_MEM_PERCENT=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{printf "%.0f", ($1/$2)*100}')
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0")

    if [ "$GPU_UTIL" -lt "$GPU_LOW_THRESHOLD" ]; then
        log_warning "GPU underutilized: ${GPU_UTIL}% (target: >${GPU_LOW_THRESHOLD}%)"
    fi

    if [ "$GPU_TEMP" -gt 85 ]; then
        log_error "GPU temperature HIGH: ${GPU_TEMP}°C"
        issues=$((issues + 1))
    fi

    log "Resources: RAM=${RAM_PERCENT}% CPU=${CPU_PERCENT}% GPU=${GPU_UTIL}% GPU_MEM=${GPU_MEM_PERCENT}% GPU_TEMP=${GPU_TEMP}°C"

    return $issues
}

# Check trial progress
check_progress() {
    if [ ! -f _optuna/noaug.db ]; then
        log_warning "Optuna DB not found"
        return 1
    fi

    TOTAL=$(sqlite3 _optuna/noaug.db "SELECT COUNT(*) FROM trials;" 2>/dev/null || echo "0")
    COMPLETE=$(sqlite3 _optuna/noaug.db "SELECT COUNT(*) FROM trials WHERE state='COMPLETE';" 2>/dev/null || echo "0")
    RUNNING=$(sqlite3 _optuna/noaug.db "SELECT COUNT(*) FROM trials WHERE state='RUNNING';" 2>/dev/null || echo "0")
    FAILED=$(sqlite3 _optuna/noaug.db "SELECT COUNT(*) FROM trials WHERE state='FAIL';" 2>/dev/null || echo "0")
    PRUNED=$(sqlite3 _optuna/noaug.db "SELECT COUNT(*) FROM trials WHERE state='PRUNED';" 2>/dev/null || echo "0")

    # Calculate failure rate (exclude pruned)
    NON_PRUNED=$((COMPLETE + RUNNING + FAILED))
    if [ "$NON_PRUNED" -gt 0 ]; then
        FAILURE_RATE=$(awk "BEGIN {printf \"%.1f\", ($FAILED/$NON_PRUNED)*100}")
    else
        FAILURE_RATE="0.0"
    fi

    log "Progress: Total=$TOTAL Complete=$COMPLETE Running=$RUNNING Failed=$FAILED ($FAILURE_RATE%) Pruned=$PRUNED"

    # Check for high failure rate
    if (( $(echo "$FAILURE_RATE > 15" | bc -l) )); then
        log_error "FAILURE RATE HIGH: ${FAILURE_RATE}%"
        return 1
    fi

    # Check for stalled progress
    if [ -f .last_complete_count ]; then
        LAST_COMPLETE=$(cat .last_complete_count)
        if [ "$COMPLETE" -eq "$LAST_COMPLETE" ]; then
            STALL_TIME=$(($(date +%s) - $(stat -c %Y .last_complete_count)))
            STALL_MINUTES=$((STALL_TIME / 60))

            if [ "$STALL_MINUTES" -gt "$STALL_THRESHOLD" ] && [ "$RUNNING" -eq 0 ]; then
                log_error "STALLED: No progress for ${STALL_MINUTES} minutes with no running trials"
                return 1
            elif [ "$STALL_MINUTES" -gt 30 ]; then
                log_warning "Slow progress: ${STALL_MINUTES} minutes since last completion"
            fi
        else
            echo "$COMPLETE" > .last_complete_count
            log_success "Progress detected: $((COMPLETE - LAST_COMPLETE)) new completions"
        fi
    else
        echo "$COMPLETE" > .last_complete_count
    fi

    return 0
}

# Main monitoring loop
log "=========================================="
log "HPO Supervisor Started"
log "=========================================="

RESTART_COUNT=0
MAX_RESTARTS=5

while true; do
    log "--- Check Cycle ---"

    # Check if process is running
    if ! check_process; then
        log_error "HPO process not running!"

        # Check if completed successfully
        if grep -q "ALL SUPER-MAX HPO RUNS COMPLETE" hpo_supermax_run.log 2>/dev/null; then
            log_success "🎉 HPO COMPLETED SUCCESSFULLY!"
            exit 0
        fi

        # Check restart count
        if [ "$RESTART_COUNT" -ge "$MAX_RESTARTS" ]; then
            log_error "Maximum restart attempts ($MAX_RESTARTS) reached. Manual intervention required."
            exit 1
        fi

        # Auto-restart
        RESTART_COUNT=$((RESTART_COUNT + 1))
        log_warning "Attempting auto-restart ($RESTART_COUNT/$MAX_RESTARTS)..."

        mv hpo_supermax_run.log "hpo_supermax_run_crash${RESTART_COUNT}.log.bak" 2>/dev/null || true

        nohup make tune-all-supermax > hpo_supermax_run.log 2>&1 &
        HPO_PID=$!
        echo $HPO_PID > hpo_supermax.pid

        log_success "HPO restarted (PID: $HPO_PID)"
        sleep 30  # Give it time to start
        continue
    fi

    # Check resources
    if ! check_resources; then
        log_warning "Resource constraints detected"
    fi

    # Check progress
    if ! check_progress; then
        log_warning "Progress issues detected"
    fi

    # Check for recent errors in log
    ERROR_COUNT=$(grep -c "Error\|Exception\|FATAL" hpo_supermax_run.log 2>/dev/null | tail -1 || echo "0")
    if [ "$ERROR_COUNT" -gt 100 ]; then
        log_warning "High error count in log: $ERROR_COUNT errors"
    fi

    # Report phase progress
    if [ "$COMPLETE" -gt 0 ]; then
        PHASE1_PROGRESS=$(awk "BEGIN {printf \"%.2f\", ($COMPLETE/5000)*100}")
        log "Phase 1 (Criteria): ${PHASE1_PROGRESS}% complete ($COMPLETE/5000)"
    fi

    log "Status: ✅ HEALTHY"
    log ""

    sleep $CHECK_INTERVAL
done
