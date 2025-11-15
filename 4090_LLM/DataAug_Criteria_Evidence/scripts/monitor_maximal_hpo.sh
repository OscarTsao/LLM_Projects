#!/usr/bin/env bash
# Continuous monitoring for maximal HPO runs

LOGDIR="./logs/maximal_2025-10-31"
INTERVAL=600  # 10 minutes

echo "==================================================================="
echo "Maximal HPO Monitoring - $(date)"
echo "==================================================================="

while true; do
    echo ""
    echo "-------------------------------------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] HPO Progress Check"
    echo "-------------------------------------------------------------------"
    
    # Check if processes are running
    for agent in criteria evidence share joint; do
        pidfile="${LOGDIR}/${agent}_hpo.pid"
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            if ps -p "$pid" > /dev/null 2>&1; then
                # Count completed trials
                logfile="${LOGDIR}/${agent}_hpo.log"
                completed=$(grep -c "Trial [0-9]* finished" "$logfile" 2>/dev/null || echo "0")
                pruned=$(grep -c "Trial [0-9]* pruned" "$logfile" 2>/dev/null || echo "0")
                total=$((completed + pruned))
                
                # Get last trial info
                last_trial=$(grep -E "Trial [0-9]+ (finished|pruned)" "$logfile" 2>/dev/null | tail -1 | grep -oP "Trial \K[0-9]+")
                
                echo "✓ $agent (PID: $pid) - Trials: $total (completed: $completed, pruned: $pruned, last: ${last_trial:-N/A})"
            else
                echo "✗ $agent (PID: $pid) - Process not running!"
            fi
        else
            echo "✗ $agent - No PID file found"
        fi
    done
    
    # Check for recent errors
    echo ""
    echo "Recent Errors (last 5):"
    grep -i "error\|exception" ${LOGDIR}/*.log 2>/dev/null | tail -5 || echo "  No errors found"
    
    # Check log sizes
    echo ""
    echo "Log Sizes:"
    ls -lh ${LOGDIR}/*.log 2>/dev/null | awk '{print "  " $9 ": " $5}'
    
    echo ""
    echo "Next check in ${INTERVAL}s ($(date -d "+${INTERVAL} seconds" '+%H:%M:%S'))"
    echo "-------------------------------------------------------------------"
    
    sleep $INTERVAL
done
