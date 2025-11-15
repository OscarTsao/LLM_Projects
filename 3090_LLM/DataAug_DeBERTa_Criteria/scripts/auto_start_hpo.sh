#!/bin/bash
# Automatically start HPO when pre-computation completes

set -e

echo "=========================================="
echo "Auto-Start HPO Script"
echo "=========================================="
echo ""
echo "This script will:"
echo "1. Wait for pre-computation to complete"
echo "2. Verify cache integrity"
echo "3. Clean old HPO studies"
echo "4. Start fresh HPO with cached augmentations"
echo "5. Monitor initial startup"
echo ""
echo "Press Ctrl+C to cancel"
echo ""

# Wait for pre-computation to complete
echo "⏳ Waiting for pre-computation to complete..."
while true; do
    if docker exec 75bb13cca2c5 test -f experiments/augmentation_cache.pkl && \
       docker exec 75bb13cca2c5 test -f experiments/augmentation_cache.json; then
        echo "✓ Cache files found!"
        break
    fi

    # Check if process is still running
    if ! docker exec 75bb13cca2c5 pgrep -f "precompute" > /dev/null; then
        echo "✗ Pre-computation process not running and cache not found!"
        echo "Check logs: docker exec 75bb13cca2c5 tail -50 /tmp/precompute_parallel.log"
        exit 1
    fi

    echo "  Still waiting... ($(date +%H:%M:%S))"
    sleep 60
done

echo ""
echo "=========================================="
echo "Pre-computation Complete!"
echo "=========================================="
echo ""

# Show cache statistics
echo "--- Cache Statistics ---"
CACHE_SIZE=$(docker exec 75bb13cca2c5 ls -lh experiments/augmentation_cache.pkl | awk '{print $5}')
echo "Cache size: $CACHE_SIZE"

if docker exec 75bb13cca2c5 test -f experiments/augmentation_cache.json; then
    docker exec 75bb13cca2c5 python -c "
import json
with open('experiments/augmentation_cache.json') as f:
    meta = json.load(f)
print(f\"Unique texts: {meta['num_texts']}\")
print(f\"Methods: {meta['num_methods']}\")
print(f\"Total cached: {meta['total_cached']:,}\")
print(f\"Success rate: {meta['stats']['success'] / meta['total_cached'] * 100:.1f}%\")
" 2>/dev/null || echo "Metadata available"
fi

echo ""
read -p "Proceed with starting HPO? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled by user"
    exit 0
fi

echo ""
echo "=========================================="
echo "Cleaning Old HPO Studies"
echo "=========================================="
echo ""

# Clean old databases
if docker exec 75bb13cca2c5 test -f experiments/criteria_hpo.db; then
    echo "Removing: experiments/criteria_hpo.db"
    docker exec 75bb13cca2c5 rm experiments/criteria_hpo.db
fi

if docker exec 75bb13cca2c5 test -f experiments/criteria_hpo_v2.db; then
    echo "Removing: experiments/criteria_hpo_v2.db"
    docker exec 75bb13cca2c5 rm experiments/criteria_hpo_v2.db
fi

echo "✓ Old studies cleaned"
echo ""

echo "=========================================="
echo "Starting Fresh HPO with Cached Augmentations"
echo "=========================================="
echo ""

# Reinstall package to ensure latest code
echo "Installing latest package version..."
docker exec 75bb13cca2c5 pip install -e . --no-deps > /dev/null 2>&1
echo "✓ Package installed"
echo ""

# Start HPO
echo "Starting HPO with parameters:"
echo "  - Experiment: criteria_hpo_final"
echo "  - Study: criteria_hpo_final"
echo "  - Database: experiments/criteria_hpo_final.db"
echo "  - Stage A trials: 380"
echo "  - Stage B trials: 120"
echo "  - Total: 500 trials"
echo ""

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

echo "✓ HPO started in background"
echo ""

# Wait for startup
echo "Waiting for HPO to initialize (30 seconds)..."
sleep 30

echo ""
echo "=========================================="
echo "HPO Startup Status"
echo "=========================================="
echo ""

# Check if process is running
if docker exec 75bb13cca2c5 pgrep -f "dataaug-train.*hpo" > /dev/null; then
    echo "✓ HPO process is running"

    # Show process info
    docker exec 75bb13cca2c5 ps aux | grep "dataaug-train.*hpo" | grep -v grep | awk '{print "  PID: " $2 ", CPU: " $3 "%, Memory: " $4 "%"}'

    echo ""
    echo "--- Recent Log Output ---"
    docker exec 75bb13cca2c5 tail -30 /tmp/hpo_final.log

    echo ""
    echo "=========================================="
    echo "✓✓✓ HPO Successfully Started! ✓✓✓"
    echo "=========================================="
    echo ""
    echo "Monitor with:"
    echo "  docker exec 75bb13cca2c5 tail -f /tmp/hpo_final.log"
    echo ""
    echo "Check GPU usage:"
    echo "  nvidia-smi"
    echo ""
    echo "Check trial progress:"
    echo "  docker exec 75bb13cca2c5 python -c \\"
    echo "    import optuna; \\"
    echo "    study = optuna.load_study(study_name='criteria_hpo_final_stage_A', storage='sqlite:///experiments/criteria_hpo_final.db'); \\"
    echo "    print(f'Trials: {len(study.trials)}, Complete: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}') \\"
    echo "  "
    echo ""

else
    echo "✗ HPO process not running!"
    echo ""
    echo "Check logs:"
    docker exec 75bb13cca2c5 tail -50 /tmp/hpo_final.log
    exit 1
fi
