#!/bin/bash
# Real-time HPO status dashboard

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

clear
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         HPO Super-Max Real-Time Monitoring Dashboard             ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 1. Process Status
echo -e "${CYAN}[1] Process Status${NC}"
if [ -f hpo_supermax.pid ]; then
    PID=$(cat hpo_supermax.pid)
    if ps -p $PID > /dev/null 2>&1; then
        RUNTIME=$(ps -p $PID -o etime= | xargs)
        echo -e "   ${GREEN}✓${NC} HPO Process: RUNNING (PID: $PID, Runtime: $RUNTIME)"
    else
        echo -e "   ${RED}✗${NC} HPO Process: STOPPED"
    fi
else
    echo -e "   ${YELLOW}⚠${NC} No PID file found"
fi

# Count active Python workers
WORKER_COUNT=$(ps aux | grep -E "tune_max\.py|train_criteria" | grep -v grep | wc -l)
echo -e "   ${CYAN}→${NC} Active Workers: $WORKER_COUNT"
echo ""

# 2. System Resources
echo -e "${CYAN}[2] System Resources${NC}"

# CPU
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{printf "%.1f", 100 - $1}')
if (( $(echo "$CPU_USAGE > 90" | bc -l) )); then
    CPU_COLOR="${RED}"
elif (( $(echo "$CPU_USAGE > 75" | bc -l) )); then
    CPU_COLOR="${YELLOW}"
else
    CPU_COLOR="${GREEN}"
fi
echo -e "   ${CPU_COLOR}CPU${NC}: ${CPU_USAGE}%"

# RAM
RAM_INFO=$(free | grep Mem)
RAM_TOTAL=$(echo $RAM_INFO | awk '{print $2}')
RAM_USED=$(echo $RAM_INFO | awk '{print $3}')
RAM_PERCENT=$(awk "BEGIN {printf \"%.1f\", ($RAM_USED/$RAM_TOTAL)*100}")
RAM_GB=$(awk "BEGIN {printf \"%.1f\", $RAM_USED/1024/1024}")
RAM_TOTAL_GB=$(awk "BEGIN {printf \"%.1f\", $RAM_TOTAL/1024/1024}")

if (( $(echo "$RAM_PERCENT > 90" | bc -l) )); then
    RAM_COLOR="${RED}"
elif (( $(echo "$RAM_PERCENT > 80" | bc -l) )); then
    RAM_COLOR="${YELLOW}"
else
    RAM_COLOR="${GREEN}"
fi
echo -e "   ${RAM_COLOR}RAM${NC}: ${RAM_PERCENT}% (${RAM_GB}GB / ${RAM_TOTAL_GB}GB)"

# GPU
GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null)
if [ -n "$GPU_INFO" ]; then
    GPU_UTIL=$(echo $GPU_INFO | cut -d',' -f1 | xargs)
    GPU_MEM_USED=$(echo $GPU_INFO | cut -d',' -f2 | xargs)
    GPU_MEM_TOTAL=$(echo $GPU_INFO | cut -d',' -f3 | xargs)
    GPU_TEMP=$(echo $GPU_INFO | cut -d',' -f4 | xargs)
    GPU_POWER=$(echo $GPU_INFO | cut -d',' -f5 | xargs)
    GPU_MEM_PERCENT=$(awk "BEGIN {printf \"%.1f\", ($GPU_MEM_USED/$GPU_MEM_TOTAL)*100}")

    # Color based on utilization (higher is better for GPU)
    if (( $(echo "$GPU_UTIL > 90" | bc -l) )); then
        GPU_COLOR="${GREEN}"
    elif (( $(echo "$GPU_UTIL > 70" | bc -l) )); then
        GPU_COLOR="${YELLOW}"
    else
        GPU_COLOR="${RED}"
    fi

    echo -e "   ${GPU_COLOR}GPU Util${NC}: ${GPU_UTIL}% | Mem: ${GPU_MEM_PERCENT}% (${GPU_MEM_USED}MB/${GPU_MEM_TOTAL}MB)"
    echo -e "   ${CYAN}GPU Temp${NC}: ${GPU_TEMP}°C | Power: ${GPU_POWER}W"
fi
echo ""

# 3. HPO Progress
echo -e "${CYAN}[3] HPO Progress${NC}"

if [ -f "mlflow.db" ]; then
    TOTAL_RUNS=$(sqlite3 mlflow.db "SELECT COUNT(*) FROM runs;" 2>/dev/null || echo "0")
    echo -e "   ${CYAN}Total Trials${NC}: $TOTAL_RUNS"

    # Status breakdown
    FINISHED=$(sqlite3 mlflow.db "SELECT COUNT(*) FROM runs WHERE status='FINISHED';" 2>/dev/null || echo "0")
    RUNNING=$(sqlite3 mlflow.db "SELECT COUNT(*) FROM runs WHERE status='RUNNING';" 2>/dev/null || echo "0")
    FAILED=$(sqlite3 mlflow.db "SELECT COUNT(*) FROM runs WHERE status='FAILED';" 2>/dev/null || echo "0")

    echo -e "   ${GREEN}✓ Finished${NC}: $FINISHED | ${YELLOW}⟳ Running${NC}: $RUNNING | ${RED}✗ Failed${NC}: $FAILED"

    if [ "$TOTAL_RUNS" -gt 0 ]; then
        FAILURE_RATE=$(awk "BEGIN {printf \"%.1f\", ($FAILED/$TOTAL_RUNS)*100}")
        if (( $(echo "$FAILURE_RATE > 10" | bc -l) )); then
            echo -e "   ${RED}⚠ Failure Rate${NC}: ${FAILURE_RATE}% (HIGH!)"
        elif (( $(echo "$FAILURE_RATE > 5" | bc -l) )); then
            echo -e "   ${YELLOW}⚠ Failure Rate${NC}: ${FAILURE_RATE}%"
        else
            echo -e "   ${GREEN}✓ Failure Rate${NC}: ${FAILURE_RATE}%"
        fi
    fi
else
    echo -e "   ${YELLOW}⚠${NC} MLflow DB not found"
fi
echo ""

# 4. Current Phase (from log)
echo -e "${CYAN}[4] Current Phase${NC}"
if [ -f hpo_supermax_run.log ]; then
    CURRENT_PHASE=$(grep -E "\[1/4\]|\[2/4\]|\[3/4\]|\[4/4\]" hpo_supermax_run.log | tail -1 | grep -oE "\[[0-9]/4\] Running [A-Za-z]+" || echo "Unknown")
    if [ "$CURRENT_PHASE" != "Unknown" ]; then
        echo -e "   ${GREEN}→${NC} $CURRENT_PHASE"
    else
        echo -e "   ${YELLOW}⚠${NC} Unable to determine current phase"
    fi

    # Recent errors (last 5 minutes worth)
    ERROR_COUNT=$(grep -c -E "(ERROR|Error)" hpo_supermax_run.log 2>/dev/null || echo "0")
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo -e "   ${RED}⚠ Errors in log${NC}: $ERROR_COUNT"
    fi
fi
echo ""

# 5. Estimated Completion
echo -e "${CYAN}[5] Estimates${NC}"
if [ -f hpo_supermax_run.log ] && [ -f mlflow.db ]; then
    TOTAL_TRIALS=19000
    COMPLETED=$(sqlite3 mlflow.db "SELECT COUNT(*) FROM runs WHERE status='FINISHED';" 2>/dev/null || echo "0")

    if [ "$COMPLETED" -gt 10 ]; then
        # Get time of first and last trial
        START_TIME=$(sqlite3 mlflow.db "SELECT MIN(start_time) FROM runs;" 2>/dev/null || echo "0")
        CURRENT_TIME=$(date +%s)000  # milliseconds

        if [ "$START_TIME" != "0" ] && [ "$START_TIME" != "" ]; then
            ELAPSED_MS=$((CURRENT_TIME - START_TIME))
            ELAPSED_HOURS=$(awk "BEGIN {printf \"%.1f\", $ELAPSED_MS/1000/3600}")
            AVG_TIME_PER_TRIAL=$(awk "BEGIN {printf \"%.0f\", $ELAPSED_MS/$COMPLETED}")
            REMAINING_TRIALS=$((TOTAL_TRIALS - COMPLETED))
            REMAINING_MS=$((AVG_TIME_PER_TRIAL * REMAINING_TRIALS))
            REMAINING_HOURS=$(awk "BEGIN {printf \"%.1f\", $REMAINING_MS/1000/3600}")

            echo -e "   ${CYAN}Elapsed${NC}: ${ELAPSED_HOURS}h"
            echo -e "   ${CYAN}Remaining${NC}: ~${REMAINING_HOURS}h (${REMAINING_TRIALS} trials left)"
            echo -e "   ${CYAN}Progress${NC}: $(awk "BEGIN {printf \"%.1f\", ($COMPLETED/$TOTAL_TRIALS)*100}")%"
        fi
    else
        echo -e "   ${YELLOW}⚠${NC} Not enough data for estimates (need >10 completed trials)"
    fi
fi
echo ""

# 6. Quick Actions
echo -e "${CYAN}[6] Quick Actions${NC}"
echo -e "   ${BLUE}View Logs${NC}:        tail -f hpo_supermax_run.log"
echo -e "   ${BLUE}Resource Monitor${NC}: tail -f hpo_resources.log"
echo -e "   ${BLUE}Progress Monitor${NC}: tail -f hpo_progress.log"
echo -e "   ${BLUE}MLflow UI${NC}:        mlflow ui --backend-store-uri sqlite:///mlflow.db"
echo -e "   ${BLUE}Stop HPO${NC}:         kill -TERM \$(cat hpo_supermax.pid)"
echo ""

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "Last updated: $(date '+%Y-%m-%d %H:%M:%S')"
