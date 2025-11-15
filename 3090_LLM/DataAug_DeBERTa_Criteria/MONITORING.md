# Pre-computation & HPO Monitoring Guide

## Quick Status Check

### Option 1: Automated Monitoring Script (Recommended)
```bash
# Continuous monitoring (updates every 30 seconds)
./scripts/monitor_precompute.sh
```

### Option 2: Python Progress Checker
```bash
# One-time detailed check
python scripts/check_precompute_progress.py
```

### Option 3: Manual Commands

**Check if pre-computation is running:**
```bash
docker exec 75bb13cca2c5 ps aux | grep precompute | grep python
```

**Check recent logs:**
```bash
docker exec 75bb13cca2c5 tail -50 /tmp/precompute_parallel.log | grep -E "(INFO|Progress|Success)"
```

**Check cache file status:**
```bash
docker exec 75bb13cca2c5 ls -lh experiments/augmentation_cache*
```

**Check completion:**
```bash
# If this returns data, pre-computation is complete!
docker exec 75bb13cca2c5 cat experiments/augmentation_cache.json 2>/dev/null
```

## Understanding the Output

### Pre-computation Stages

1. **Model Loading (10-15 min)**
   - You'll see: "Loading dataset...", "Found X unique texts", model sharding messages
   - 9 workers load 25 augmentation models in parallel
   - High CPU, low disk activity

2. **Text Processing (30-75 min)**
   - You'll see: "Progress: X/20000 (Y%)" every 100 items
   - High CPU across all 9 workers
   - Checkpoint saves every 2,500 items

3. **Final Save (1-2 min)**
   - You'll see: "Saving final cache...", "✓ Cache saved"
   - Creates `augmentation_cache.pkl` (~50-200 MB)
   - Creates `augmentation_cache.json` (metadata)

### Success Indicators

✓ Cache file exists: `experiments/augmentation_cache.pkl`
✓ Metadata exists: `experiments/augmentation_cache.json`
✓ Total cached entries: ~20,000 (800 texts × 25 methods)
✓ Success rate: >90%

## HPO Trial Structure

### Two-Stage Approach (500 total trials)

**Stage A: 450 trials** - Broad exploration
- Augmentation methods: 0-25 (Optuna selects)
- ~150 trials with aug=0 (no augmentation, instant)
- ~300 trials with aug>0 (uses cache, instant lookup)

**Stage B: 50 trials** - Fine-tuning
- Narrows search around top-8 results from Stage A
- Inherits augmentation settings from winners

### Cache Benefits

**Without cache:**
- Trial with 12 aug methods: ~2 hours (pre-compute each time)
- 300 trials with aug: ~600 hours = 25 days

**With cache:**
- Trial with 12 aug methods: ~10-15 min (instant cache lookup)
- 300 trials with aug: ~50-75 hours = 2-3 days
- **Speedup: 10x faster**

## Troubleshooting

### Pre-computation stuck?
```bash
# Check if workers are active
docker exec 75bb13cca2c5 top -b -n 1 | grep python

# Check full logs
docker exec 75bb13cca2c5 cat /tmp/precompute_parallel.log

# Restart if needed (preserves checkpoint)
docker exec 75bb13cca2c5 pkill -9 -f precompute
docker exec -d 75bb13cca2c5 bash -c "cd /workspaces/DataAug_DeBERTa_Criteria && CUDA_VISIBLE_DEVICES='' python -u scripts/precompute_augmentations_parallel.py > /tmp/precompute_parallel.log 2>&1"
```

### After completion
```bash
# Verify cache
docker exec 75bb13cca2c5 python -c "
import pickle
with open('experiments/augmentation_cache.pkl', 'rb') as f:
    cache = pickle.load(f)
print(f'Cache entries: {len(cache)}')
print(f'Sample keys: {list(cache.keys())[:5]}')
"

# Clean old HPO studies
rm experiments/criteria_hpo.db
rm experiments/criteria_hpo_v2.db

# Start fresh HPO
docker exec -d 75bb13cca2c5 bash -c "cd /workspaces/DataAug_DeBERTa_Criteria && /home/vscode/.local/bin/dataaug-train --hpo --experiment-name criteria_hpo_final --study-name criteria_hpo_final --study-db experiments/criteria_hpo_final.db --trials-a 450 --trials-b 50 > /tmp/hpo_final.log 2>&1"
```

## Timeline

| Phase | Duration | What's Happening |
|-------|----------|------------------|
| Model loading | 10-15 min | 9 workers load 25 augmentation models |
| Text processing | 30-75 min | Process 800 texts × 25 methods |
| Final save | 1-2 min | Write cache to disk |
| **Total** | **45-90 min** | One-time cost |
| HPO (after) | 2-4 days | All 500 trials use cache instantly |

## Current Status

Run `python scripts/check_precompute_progress.py` to see current progress!
