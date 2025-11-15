#!/bin/bash
# Fully automated HPO start (no user interaction required)
# Run in background: ./scripts/auto_start_hpo_unattended.sh > /tmp/auto_hpo.log 2>&1 &

set -e

LOG_FILE="/tmp/auto_hpo_startup.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "Auto-Start HPO Script (Unattended Mode)"
log "=========================================="
log ""

# Wait for pre-computation to complete
log "Waiting for pre-computation to complete..."
WAIT_COUNT=0
while true; do
    if docker exec 75bb13cca2c5 test -f experiments/augmentation_cache.pkl && \
       docker exec 75bb13cca2c5 test -f experiments/augmentation_cache.json; then
        log "✓ Cache files found!"
        break
    fi

    # Check if process is still running
    if ! docker exec 75bb13cca2c5 pgrep -f "precompute" > /dev/null; then
        log "WARNING: Pre-computation process not running and cache not found!"
        log "Waiting 2 more minutes in case it just finished..."
        sleep 120

        # Final check
        if docker exec 75bb13cca2c5 test -f experiments/augmentation_cache.pkl; then
            log "✓ Cache found after waiting"
            break
        else
            log "ERROR: Cache not found. Exiting."
            exit 1
        fi
    fi

    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ $WAIT_COUNT -ge 120 ]; then  # 2 hours max wait
        log "ERROR: Timeout waiting for pre-computation (2 hours)"
        exit 1
    fi

    log "  Still waiting... ($WAIT_COUNT minutes elapsed)"
    sleep 60
done

log ""
log "=========================================="
log "Pre-computation Complete!"
log "=========================================="
log ""

# Show cache statistics
CACHE_SIZE=$(docker exec 75bb13cca2c5 ls -lh experiments/augmentation_cache.pkl | awk '{print $5}')
log "Cache size: $CACHE_SIZE"

if docker exec 75bb13cca2c5 test -f experiments/augmentation_cache.json; then
    STATS=$(docker exec 75bb13cca2c5 python -c "
import json
with open('experiments/augmentation_cache.json') as f:
    meta = json.load(f)
print(f\"{meta['num_texts']} texts, {meta['num_methods']} methods, {meta['total_cached']:,} cached\")
" 2>/dev/null || echo "Stats unavailable")
    log "Cache stats: $STATS"
fi

log ""
log "=========================================="
log "Cleaning Old HPO Studies"
log "=========================================="
log ""

# Clean old databases
for db in experiments/criteria_hpo.db experiments/criteria_hpo_v2.db; do
    if docker exec 75bb13cca2c5 test -f "$db"; then
        log "Removing: $db"
        docker exec 75bb13cca2c5 rm "$db"
    fi
done

log "✓ Old studies cleaned"
log ""

log "=========================================="
log "Installing Latest Package"
log "=========================================="
log ""

docker exec 75bb13cca2c5 pip install -e . --no-deps > /dev/null 2>&1
log "✓ Package installed"
log ""

log "=========================================="
log "Starting Fresh HPO"
log "=========================================="
log ""
log "Configuration:"
log "  - Experiment: criteria_hpo_final"
log "  - Study: criteria_hpo_final"
log "  - Database: experiments/criteria_hpo_final.db"
log "  - Stage A: 380 trials (100 epochs each)"
log "  - Stage B: 120 trials (100 epochs each)"
log "  - Total: 500 trials"
log ""

docker exec -d 75bb13cca2c5 bash -c "
cd /workspaces/DataAug_DeBERTa_Criteria &&
/home/vscode/.local/bin/dataaug-train \
  --hpo \
  --experiment-name criteria_hpo_final \
  --study-name criteria_hpo_final \
  --study-db experiments/criteria_hpo_final.db \
  --trials-a 380 \
  --trials-b 120 \
  --epochs-a 100 \
  --epochs-b 100 \
  > /tmp/hpo_final.log 2>&1
"

log "✓ HPO started in background"
log ""

# Wait for startup
log "Waiting for HPO to initialize (60 seconds)..."
sleep 60

log ""
log "=========================================="
log "HPO Startup Verification"
log "=========================================="
log ""

# Check if process is running
if docker exec 75bb13cca2c5 pgrep -f "dataaug-train.*hpo" > /dev/null; then
    log "✓ HPO process is running"

    # Show process info
    PROC_INFO=$(docker exec 75bb13cca2c5 ps aux | grep "dataaug-train.*hpo" | grep -v grep | awk '{print "PID: " $2 ", CPU: " $3 "%, Memory: " $4 "%"}')
    log "  $PROC_INFO"

    # Check GPU
    GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "N/A")
    log "  GPU: $GPU_INFO"

    log ""
    log "=========================================="
    log "✓✓✓ HPO Successfully Started! ✓✓✓"
    log "=========================================="
    log ""
    log "Monitor commands:"
    log "  tail -f /tmp/hpo_final.log"
    log "  nvidia-smi -l 5"
    log "  python scripts/check_precompute_progress.py"
    log ""
    log "This script will now exit. HPO continues in background."
    log ""

else
    log "✗ HPO process not running!"
    log ""
    log "Last 30 lines of log:"
    docker exec 75bb13cca2c5 tail -30 /tmp/hpo_final.log | tee -a "$LOG_FILE"
    exit 1
fi
