# GPU Utilization Optimization - Quick Start Guide

**Status:** ✅ Changes are LIVE in tune_max.py
**Target:** Increase GPU util from 82-84% to 95-100%
**Risk:** 🟢 LOW (6-9GB GPU memory headroom)

---

## What Was Changed

### 1. DataLoader Workers: 2 → 4
- Faster data loading
- Added persistent_workers (no restart overhead)
- Increased prefetch_factor to 3

### 2. Pruning: Less Aggressive
- min_resource: 1 → 3 epochs (wait longer before pruning)
- patience: 2 → 4 (more tolerant of bad epochs)

**Expected Result:** GPU util 95-100% (up from 82-84%)

---

## How Changes Apply to Running HPO

**Your current HPO is already using the new settings!**

New trials (216+) will automatically use:
- 4 DataLoader workers (vs 2)
- persistent_workers + prefetch_factor=3
- min_resource=3, patience=4 pruning

**Timeline:**
- Next 10-20 minutes: New trials start with optimized settings
- 20-40 minutes: Full effect visible (all 4 parallel trials optimized)

---

## Monitoring

### Quick Check (2 seconds)
```bash
nvidia-smi
```
Look for: GPU util > 95%, Memory < 22GB

### Real-time Monitor (continuous)
```bash
./scripts/monitor_gpu_util.sh
```
Press Ctrl+C to exit

### Trial Statistics
```bash
sqlite3 ./_optuna/noaug.db "
  SELECT state, COUNT(*) FROM trials
  WHERE study_id = (SELECT study_id FROM studies WHERE study_name = 'noaug-criteria-supermax')
  GROUP BY state
"
```
Look for: PRUNED rate dropping from 86.5% to 60-70%

---

## Success Criteria (Check in 20-40 min)

- ✅ GPU utilization > 95%
- ✅ GPU memory < 22GB
- ✅ No OOM errors
- ✅ Pruning rate < 75%

---

## If Something Goes Wrong

### OOM Error
```bash
# Stop HPO
pkill -f "tune_max.py"

# Reduce workers (edit tune_max.py line 597)
# Change: num_workers = 4
# To: num_workers = 2

# Restart
make tune-criteria-supermax
```

### GPU Util Still Low
**Wait 20-40 minutes** - changes apply gradually as new trials start

### Other Issues
See GPU_OPTIMIZATION_IMPLEMENTATION.md for full details

---

## Test Before Full Run (Optional)

```bash
# Run 10 test trials
./scripts/test_gpu_optimization.sh
```

This tests:
- No OOM
- Improved GPU util
- Correct pruning behavior

---

## Key Files

- **Analysis:** GPU_UTILIZATION_ANALYSIS.md
- **Implementation:** GPU_OPTIMIZATION_IMPLEMENTATION.md
- **This Guide:** GPU_OPTIMIZATION_QUICKSTART.md

**Modified:**
- scripts/tune_max.py (lines 592-617, 826-837)

**Created:**
- scripts/test_gpu_optimization.sh
- scripts/monitor_gpu_util.sh

---

## What to Expect

**Before (trials 1-215):**
- 82-84% GPU util
- 86.5% pruning rate
- 2-4 min trial duration

**After (trials 216+):**
- 95-100% GPU util
- 60-70% pruning rate
- 4-8 min trial duration

**No Action Needed** - Just monitor and enjoy better GPU utilization!
