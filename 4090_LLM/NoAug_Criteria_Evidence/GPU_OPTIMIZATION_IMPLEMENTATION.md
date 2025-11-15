# GPU Utilization Optimization - Implementation Summary

**Date:** 2025-10-24
**Status:** ✅ IMPLEMENTED (Ready for Testing)
**Approach:** Option D - Hybrid (DataLoader + Pruning optimization)

---

## Changes Made

### 1. DataLoader Optimization (tune_max.py lines 592-617)

**Before:**
```python
num_workers = 2  # Conservative

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True if torch.cuda.is_available() else False,
)
```

**After:**
```python
num_workers = 4  # Increased for better GPU feeding

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True if torch.cuda.is_available() else False,
    persistent_workers=True if num_workers > 0 else False,  # NEW
    prefetch_factor=3 if num_workers > 0 else None,  # NEW (was default 2)
)
```

**Impact:**
- Faster data loading (4 workers vs 2)
- Workers stay alive between epochs (persistent_workers)
- More aggressive prefetching (3 batches vs 2)
- Reduces GPU idle time waiting for data

### 2. Pruner Optimization (tune_max.py lines 826-837)

**Before:**
```python
hb = HyperbandPruner(
    min_resource=1,  # Prune after just 1 epoch
    max_resource=100,
    reduction_factor=3,
)
return PatientPruner(hb, patience=2)  # Very impatient
```

**After:**
```python
hb = HyperbandPruner(
    min_resource=3,  # Wait at least 3 epochs before pruning
    max_resource=100,
    reduction_factor=3,
)
return PatientPruner(hb, patience=4)  # More patient
```

**Impact:**
- Trials run longer before being pruned (3+ epochs vs 1+ epochs)
- Expected pruning rate: 60-70% (down from 86.5%)
- More GPU time spent training, less on trial startup overhead

---

## Expected Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| GPU Utilization | 82-84% | 95-100% | +13-18% |
| GPU Memory | 12GB | 15-18GB | +3-6GB |
| Trial Duration (avg) | 2-4 min | 4-8 min | 2x |
| Pruning Rate | 86.5% | 60-70% | -16-26% |
| DataLoader Workers | 8 total | 16 total | 2x |
| CPU Utilization | 66.7% | 100% | +33% |

---

## Safety Analysis

### Resource Headroom
- **GPU Memory:** 6-9GB headroom (24GB total - 15-18GB used)
- **System RAM:** 38GB headroom (48GB total - 10GB used)
- **CPU Cores:** 12 available, fully utilized (optimal)

### OOM Risk Assessment
**Risk Level:** 🟢 **LOW**

**Rationale:**
1. Current GPU memory: 12GB/24GB (50%)
2. Expected increase: +3-6GB (to 15-18GB)
3. Safety margin: 6-9GB (25-37%)
4. No batch size increase (OOM trigger)
5. Workers use CPU RAM, not GPU VRAM

**Rollback Plan if OOM:**
1. Reduce num_workers: 4 → 2
2. Keep persistent_workers and prefetch_factor
3. Keep pruning changes (min_resource=3, patience=4)
4. This gives +8-12% util without OOM risk

---

## Testing & Validation

### Test Script
```bash
# Run 10-trial test
./scripts/test_gpu_optimization.sh
```

**What it tests:**
- No OOM errors
- Improved GPU utilization
- Trial completion rate
- Pruning behavior

**Expected output:**
- 0 OOM errors
- 2-4 completed trials
- 4-6 pruned trials (vs 8-9 before)
- GPU util 95-100% during training

### Monitoring Script
```bash
# Real-time GPU monitoring
./scripts/monitor_gpu_util.sh
```

**Shows:**
- Live GPU utilization %
- GPU memory usage (MB)
- Running trials count
- Recent trial activity
- Trial state statistics

---

## Application to Running HPO

### Current HPO Status
```
Study: noaug-criteria-supermax
Total trials: 200
  - COMPLETE: 8 (4%)
  - PRUNED: 173 (86.5%)
  - FAILED: 15 (7.5%)
  - RUNNING: 4

Currently on trial: 215
```

### How Changes Apply

**Important:** Changes are already active in `scripts/tune_max.py`

**For Running HPO:**
- New trials (216+) will use optimized settings
- Existing running trials (208-215) use old settings
- You'll see improvement within 10-20 minutes

**Options:**

**Option 1: Let it Continue (RECOMMENDED)**
- Pros: No progress loss
- Cons: Gradual improvement
- Time to full effect: 20-40 minutes

**Option 2: Restart HPO**
- Pros: Immediate full effect
- Cons: Lose current 4 running trials
- Command:
  ```bash
  pkill -f "tune_max.py"
  make tune-criteria-supermax
  ```

---

## Validation Checklist

After changes are active (20-40 min):

- [ ] GPU utilization sustained at 95-100%
- [ ] GPU memory stays below 22GB
- [ ] No OOM errors in logs
- [ ] Trial duration increased (4-8 min avg)
- [ ] Pruning rate decreased (60-70% vs 86.5%)
- [ ] At least 2-3 trials training simultaneously

**Success Criteria:**
- ✅ GPU util > 95%
- ✅ GPU mem < 22GB
- ✅ No crashes or OOM
- ✅ Trial throughput stable or improved

**If any criterion fails:**
- See rollback plan in Safety Analysis section

---

## Monitoring Commands

```bash
# Watch GPU in real-time
watch -n 2 nvidia-smi

# Or use custom monitor
./scripts/monitor_gpu_util.sh

# Check trial statistics
sqlite3 ./_optuna/noaug.db "
  SELECT state, COUNT(*) as count,
         ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM trials
         WHERE study_id = (SELECT study_id FROM studies
         WHERE study_name = 'noaug-criteria-supermax')), 1) as pct
  FROM trials
  WHERE study_id = (SELECT study_id FROM studies
  WHERE study_name = 'noaug-criteria-supermax')
  GROUP BY state
"

# Average trial duration by state
sqlite3 ./_optuna/noaug.db "
  SELECT state,
         AVG(CAST((julianday(datetime_complete) - julianday(datetime_start)) * 24 * 60 AS INTEGER)) as avg_min,
         COUNT(*) as count
  FROM trials
  WHERE study_id = (SELECT study_id FROM studies
  WHERE study_name = 'noaug-criteria-supermax')
    AND datetime_complete IS NOT NULL
  GROUP BY state
"

# Count active training processes
ps aux | grep -c "[p]ython.*tune_max.py"
```

---

## Files Modified

1. **scripts/tune_max.py** (2 sections)
   - Lines 592-617: DataLoader optimization
   - Lines 826-837: Pruner optimization

## Files Created

1. **GPU_UTILIZATION_ANALYSIS.md** - Comprehensive root cause analysis
2. **GPU_OPTIMIZATION_IMPLEMENTATION.md** - This file (implementation summary)
3. **scripts/test_gpu_optimization.sh** - Test script for validation
4. **scripts/monitor_gpu_util.sh** - Real-time GPU monitoring

---

## Next Steps

### Immediate (Now)
1. ✅ Changes implemented in tune_max.py
2. ⏳ Wait 20-40 minutes for new trials to start
3. 🔍 Monitor GPU utilization with scripts

### Validation (20-40 min)
1. Run monitoring script: `./scripts/monitor_gpu_util.sh`
2. Verify GPU util > 95%
3. Verify GPU mem < 22GB
4. Check for OOM errors in logs

### Optional Testing (If Paranoid)
1. Stop current HPO: `pkill -f tune_max.py`
2. Run test: `./scripts/test_gpu_optimization.sh`
3. Restart HPO: `make tune-criteria-supermax`

### Long-term (Next 1-2 hours)
1. Monitor trial statistics
2. Verify pruning rate decreased
3. Confirm stable performance

---

## Rollback Instructions

**If OOM or crashes occur:**

```bash
# 1. Stop HPO
pkill -f "tune_max.py"

# 2. Revert changes
cd /media/user/SSD1/YuNing/NoAug_Criteria_Evidence

# 2a. Reduce num_workers (keep other changes)
# Edit scripts/tune_max.py line 597:
#   num_workers = 2  # Back to conservative

# OR

# 2b. Full revert (if needed)
git checkout scripts/tune_max.py

# 3. Restart HPO
make tune-criteria-supermax
```

---

## Technical Notes

### Why This Works

**DataLoader Optimization:**
- More workers = faster data loading
- persistent_workers = no startup overhead per epoch
- prefetch_factor = batch queue ready for GPU
- Combined: GPU spends less time waiting for data

**Pruner Optimization:**
- min_resource=3: Trials must train 3 epochs before pruning consideration
- patience=4: Trials get 4 chances to improve before pruning
- Less aggressive pruning = longer trial life
- Longer trials = less startup overhead, more GPU training time

**Why It's Safe:**
- No batch size increase (main OOM trigger)
- Workers use CPU RAM, not GPU VRAM
- 6-9GB GPU memory headroom
- Can easily rollback if issues

### Performance Expectations

**Before:**
- Trial startup: 30-60s
- Trial training: 2-4 min (1-2 epochs before prune)
- GPU utilization: 82-84%
- Time distribution: 20% startup, 80% training

**After:**
- Trial startup: 20-40s (faster data loading)
- Trial training: 4-8 min (3-6 epochs before prune)
- GPU utilization: 95-100%
- Time distribution: 10% startup, 90% training

**Overall Impact:**
- More GPU time spent training
- Less GPU time idle during startup/shutdown
- Better trial quality (longer training before prune)
- Potentially better hyperparameters (less aggressive pruning)

---

## Support

**If you need to:**
- **Revert changes:** See Rollback Instructions
- **Debug OOM:** Check GPU memory with `nvidia-smi`
- **Verify changes:** `git diff scripts/tune_max.py`
- **Check logs:** `tail -f _runs/*.log`

**Questions or issues:**
- Check GPU_UTILIZATION_ANALYSIS.md for detailed explanation
- Use monitoring scripts for real-time feedback
- Trial statistics show pruning rate trends
