#!/bin/bash
# Resource monitoring script - logs every 30 seconds
# Prevents resource exhaustion by tracking CPU/RAM/GPU usage

LOG_FILE="${1:-resource_monitor.log}"
ALERT_THRESHOLD_RAM=90
ALERT_THRESHOLD_CPU=95
ALERT_THRESHOLD_GPU_MEM=95

echo "=== Resource Monitoring Started at $(date) ===" >> "$LOG_FILE"
echo "Thresholds: RAM=${ALERT_THRESHOLD_RAM}%, CPU=${ALERT_THRESHOLD_CPU}%, GPU_MEM=${ALERT_THRESHOLD_GPU_MEM}%" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # CPU Usage (average across all cores)
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')

    # RAM Usage
    RAM_INFO=$(free | grep Mem)
    RAM_TOTAL=$(echo $RAM_INFO | awk '{print $2}')
    RAM_USED=$(echo $RAM_INFO | awk '{print $3}')
    RAM_PERCENT=$(awk "BEGIN {printf \"%.1f\", ($RAM_USED/$RAM_TOTAL)*100}")

    # GPU Usage
    GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
    if [ -n "$GPU_INFO" ]; then
        GPU_UTIL=$(echo $GPU_INFO | cut -d',' -f1 | xargs)
        GPU_MEM_USED=$(echo $GPU_INFO | cut -d',' -f2 | xargs)
        GPU_MEM_TOTAL=$(echo $GPU_INFO | cut -d',' -f3 | xargs)
        GPU_TEMP=$(echo $GPU_INFO | cut -d',' -f4 | xargs)
        GPU_MEM_PERCENT=$(awk "BEGIN {printf \"%.1f\", ($GPU_MEM_USED/$GPU_MEM_TOTAL)*100}")
    else
        GPU_UTIL="N/A"
        GPU_MEM_PERCENT="N/A"
        GPU_TEMP="N/A"
    fi

    # Log current stats
    echo "[$TIMESTAMP] CPU: ${CPU_USAGE}% | RAM: ${RAM_PERCENT}% ($(awk "BEGIN {printf \"%.1f\", $RAM_USED/1024/1024}")GB/$(awk "BEGIN {printf \"%.1f\", $RAM_TOTAL/1024/1024}")GB) | GPU: ${GPU_UTIL}% | GPU_MEM: ${GPU_MEM_PERCENT}% (${GPU_MEM_USED}MB/${GPU_MEM_TOTAL}MB) | TEMP: ${GPU_TEMP}°C" >> "$LOG_FILE"

    # Alert on high usage
    if (( $(echo "$RAM_PERCENT > $ALERT_THRESHOLD_RAM" | bc -l) )); then
        echo "[$TIMESTAMP] ⚠️  WARNING: RAM usage at ${RAM_PERCENT}% (threshold: ${ALERT_THRESHOLD_RAM}%)" >> "$LOG_FILE"
    fi

    if (( $(echo "$CPU_USAGE > $ALERT_THRESHOLD_CPU" | bc -l) )); then
        echo "[$TIMESTAMP] ⚠️  WARNING: CPU usage at ${CPU_USAGE}% (threshold: ${ALERT_THRESHOLD_CPU}%)" >> "$LOG_FILE"
    fi

    if [ "$GPU_MEM_PERCENT" != "N/A" ] && (( $(echo "$GPU_MEM_PERCENT > $ALERT_THRESHOLD_GPU_MEM" | bc -l) )); then
        echo "[$TIMESTAMP] ⚠️  WARNING: GPU memory at ${GPU_MEM_PERCENT}% (threshold: ${ALERT_THRESHOLD_GPU_MEM}%)" >> "$LOG_FILE"
    fi

    sleep 30
done
