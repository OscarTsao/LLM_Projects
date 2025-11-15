#!/usr/bin/env bash
# Monitor GPU and RAM usage during HPO runs
# Alerts if GPU utilization <90% or RAM >90%

set -euo pipefail

LOG_FILE="hpo_monitor.log"
ALERT_FILE="hpo_alerts.log"
INTERVAL=10  # Check every 10 seconds

echo "===== HPO Resource Monitor Started: $(date) =====" | tee -a "$LOG_FILE"
echo "Target: GPU >90%, RAM <90%" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

while true; do
    # Get GPU utilization
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{printf "%.1f", ($1/$2)*100}')

    # Get RAM utilization
    ram_info=$(free | grep Mem)
    ram_total=$(echo "$ram_info" | awk '{print $2}')
    ram_used=$(echo "$ram_info" | awk '{print $3}')
    ram_percent=$(echo "scale=1; ($ram_used/$ram_total)*100" | bc)

    # Get CPU utilization
    cpu_percent=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)

    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    status="[${timestamp}] GPU: ${gpu_util}% | GPU_MEM: ${gpu_mem}% | RAM: ${ram_percent}% | CPU: ${cpu_percent}%"

    # Log status
    echo "$status" | tee -a "$LOG_FILE"

    # Check alerts
    if (( $(echo "$gpu_util < 90" | bc -l) )); then
        alert="[ALERT] GPU utilization below 90%: ${gpu_util}%"
        echo "$alert" | tee -a "$ALERT_FILE"
    fi

    if (( $(echo "$ram_percent > 90" | bc -l) )); then
        alert="[CRITICAL] RAM usage above 90%: ${ram_percent}%"
        echo "$alert" | tee -a "$ALERT_FILE"
        echo "Consider reducing batch_size or num_workers!" | tee -a "$ALERT_FILE"
    fi

    sleep "$INTERVAL"
done
