#!/usr/bin/env bash
# Run maximal HPO for all architectures with monitoring and error recovery
# Date: 2025-10-31

set -e  # Exit on error

# Configuration
export HPO_EPOCHS=100
export HPO_PATIENCE=20
DATE_SUFFIX=$(date +%Y-%m-%d)
OUTDIR="./_runs/maximal_${DATE_SUFFIX}"
LOGDIR="./logs/maximal_${DATE_SUFFIX}"

mkdir -p "$OUTDIR" "$LOGDIR"

echo "==================================================================="
echo "Maximal HPO Run - All Architectures"
echo "==================================================================="
echo "Date: $(date)"
echo "Output Directory: $OUTDIR"
echo "Log Directory: $LOGDIR"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

run_hpo() {
    local agent=$1
    local trials=$2
    local study_name="noaug-${agent}-max-${DATE_SUFFIX}"
    local logfile="${LOGDIR}/${agent}_hpo.log"
    local pidfile="${LOGDIR}/${agent}_hpo.pid"

    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${agent} HPO...${NC}"
    echo "  Study: $study_name"
    echo "  Trials: $trials"
    echo "  Log: $logfile"
    echo ""

    # Run HPO in background with separate Optuna database per agent
    local storage="sqlite:///./_optuna/${agent}_maximal.db"
    nohup python scripts/tune_max.py \
        --agent "$agent" \
        --study-name "$study_name" \
        --trials "$trials" \
        --epochs "$HPO_EPOCHS" \
        --patience "$HPO_PATIENCE" \
        --outdir "$OUTDIR" \
        --storage "$storage" \
        > "$logfile" 2>&1 &

    local pid=$!
    echo $pid > "$pidfile"

    echo -e "${GREEN}  Started with PID: $pid${NC}"
    echo ""

    # Wait a bit to check if it starts successfully
    sleep 5

    if ps -p $pid > /dev/null; then
        echo -e "${GREEN}  ✓ $agent HPO running successfully${NC}"
        return 0
    else
        echo -e "${RED}  ✗ $agent HPO failed to start!${NC}"
        echo "  Check log: $logfile"
        tail -30 "$logfile"
        return 1
    fi
}

monitor_hpo() {
    local agent=$1
    local pidfile="${LOGDIR}/${agent}_hpo.pid"
    local logfile="${LOGDIR}/${agent}_hpo.log"

    if [ ! -f "$pidfile" ]; then
        echo -e "${RED}✗ No PID file for $agent${NC}"
        return 1
    fi

    local pid=$(cat "$pidfile")

    if ps -p $pid > /dev/null; then
        # Get last few lines from log
        local last_trial=$(grep -E "Trial [0-9]+ finished" "$logfile" | tail -1 | grep -oP "Trial \K[0-9]+")
        echo -e "${GREEN}✓ $agent (PID: $pid) - Last trial: ${last_trial:-0}${NC}"
        return 0
    else
        echo -e "${RED}✗ $agent (PID: $pid) - Process not running${NC}"
        return 1
    fi
}

# Main execution
echo "-------------------------------------------------------------------"
echo "1. Launching Criteria HPO (800 trials)"
echo "-------------------------------------------------------------------"
run_hpo "criteria" 800
CRITERIA_STATUS=$?

sleep 10

echo ""
echo "-------------------------------------------------------------------"
echo "2. Launching Evidence HPO (1200 trials)"
echo "-------------------------------------------------------------------"
run_hpo "evidence" 1200
EVIDENCE_STATUS=$?

sleep 10

echo ""
echo "-------------------------------------------------------------------"
echo "3. Launching Share HPO (600 trials)"
echo "-------------------------------------------------------------------"
run_hpo "share" 600
SHARE_STATUS=$?

sleep 10

echo ""
echo "-------------------------------------------------------------------"
echo "4. Launching Joint HPO (600 trials)"
echo "-------------------------------------------------------------------"
run_hpo "joint" 600
JOINT_STATUS=$?

echo ""
echo "==================================================================="
echo "All HPO Runs Launched"
echo "==================================================================="
echo ""

# Summary
echo "Launch Summary:"
[ $CRITERIA_STATUS -eq 0 ] && echo -e "${GREEN}✓ Criteria${NC}" || echo -e "${RED}✗ Criteria${NC}"
[ $EVIDENCE_STATUS -eq 0 ] && echo -e "${GREEN}✓ Evidence${NC}" || echo -e "${RED}✗ Evidence${NC}"
[ $SHARE_STATUS -eq 0 ] && echo -e "${GREEN}✓ Share${NC}" || echo -e "${RED}✗ Share${NC}"
[ $JOINT_STATUS -eq 0 ] && echo -e "${GREEN}✓ Joint${NC}" || echo -e "${RED}✗ Joint${NC}"

echo ""
echo "-------------------------------------------------------------------"
echo "Monitoring Loop (Ctrl+C to exit)"
echo "-------------------------------------------------------------------"

# Continuous monitoring
while true; do
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] HPO Status:"
    echo "-------------------------------------------------------------------"

    monitor_hpo "criteria"
    monitor_hpo "evidence"
    monitor_hpo "share"
    monitor_hpo "joint"

    # Check if all are done
    all_done=true
    for agent in criteria evidence share joint; do
        pidfile="${LOGDIR}/${agent}_hpo.pid"
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            if ps -p $pid > /dev/null; then
                all_done=false
                break
            fi
        fi
    done

    if [ "$all_done" = true ]; then
        echo ""
        echo -e "${GREEN}All HPO runs completed!${NC}"
        break
    fi

    # Wait before next check
    sleep 60
done

echo ""
echo "==================================================================="
echo "Maximal HPO Complete"
echo "==================================================================="
echo "Results are in: $OUTDIR"
echo "Logs are in: $LOGDIR"
