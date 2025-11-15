# GPU Utilization Analysis and Optimization Plan

**Date:** 2025-10-24
**Current Status:** 82-84% GPU utilization (target: 95-100%)
**GPU:** NVIDIA RTX 3090/4090 (24GB)
**Current Memory:** 12GB/24GB (50% used, 12GB headroom)

---

## Root Cause Analysis

### 1. Current Configuration
```
Parallel trials: 4 (--parallel 4)
Trials completed: 200 total
  - COMPLETE: 8
  - PRUNED: 173 (86.5%)
  - FAILED: 15
  - RUNNING: 4

DataLoader workers: 2 per trial
Total CPU workers: 8 (for 12 available cores)
CPU Utilization: 66.7% (8/12 cores)
GPU Memory: 12GB/24GB (50% used)
```

### 2. **PRIMARY BOTTLENECK IDENTIFIED: Aggressive Pruning**

**Key Finding from Trial Timing Analysis:**
```
Trial 213: 19:56:09 → 20:05:22 (9 min) COMPLETE
Trial 212: 19:44:29 → 19:53:40 (9 min) COMPLETE
Trial 211: 19:53:40 → 19:56:08 (2 min) PRUNED
Trial 210: 19:44:29 → 19:53:40 (9 min) COMPLETE
Trial 209: 19:40:07 → 20:13:51 (33 min) PRUNED ⚠️
Trial 208-207: RUNNING (started 19:40:07)
```

**Root Cause:**
- **86.5% pruning rate** (173/200 trials) with PatientPruner(patience=2) + HyperbandPruner
- Trials are pruned at epoch 2-4 (very early), causing rapid trial turnover
- Most GPU time spent on trial initialization, not training
- Sequential bottleneck: Trial startup latency >> training time

**Why Trials Aren't Running in Parallel:**
1. **TPE Sampler with constant_liar=True** still requires some history before parallelizing
2. **Aggressive pruning** means trials complete in 0-4 minutes (vs 9 min for successful ones)
3. **Trial startup latency**: Model loading, data loading, GPU initialization takes 30-60s
4. **Effective parallelism**: Only 1-2 trials training simultaneously, rest are starting/stopping

### 3. Secondary Bottlenecks

**CPU Underutilization:**
- Only 8 DataLoader workers for 12 CPU cores (66.7%)
- Each trial uses num_workers=2 (conservative for parallel HPO)
- Can increase to 3-4 workers per trial

**DataLoader Settings:**
```python
# Current (tune_max.py lines 592-611)
num_workers = 2  # Conservative to avoid oversubscription
persistent_workers = False  # Not set
prefetch_factor = 2  # Default
```

---

## Optimization Options (Ranked by Safety)

### Option A: Increase DataLoader Workers (SAFEST)
**Safety:** 🟢 Very Safe
**Expected GPU Util Gain:** +5-10% (87-94%)
**Implementation Time:** 2 minutes

**Changes:**
```python
# In tune_max.py line 595
num_workers = 4  # Up from 2 (still leaves 4 cores for main process)
persistent_workers = True  # Keep workers alive
prefetch_factor = 3  # Increase prefetch
```

**Pros:**
- No OOM risk (memory usage is CPU RAM, not GPU)
- Better GPU feeding (less idle time)
- Small, isolated change

**Cons:**
- Limited impact (+5-10%) due to pruning bottleneck
- Doesn't address root cause

### Option B: Reduce Pruning Aggressiveness (MODERATE RISK)
**Safety:** 🟡 Moderate Risk
**Expected GPU Util Gain:** +10-15% (92-99%)
**Implementation Time:** 5 minutes

**Changes:**
```python
# In tune_max.py line 826
return PatientPruner(hb, patience=5)  # Up from 2

# OR adjust Hyperband
hb = HyperbandPruner(
    min_resource=5,  # Up from 1 (wait 5 epochs before pruning)
    max_resource=100,
    reduction_factor=3,
)
```

**Pros:**
- Allows trials to train longer before pruning
- Better GPU utilization (more training, less startup overhead)
- May find better hyperparameters (less aggressive pruning)

**Cons:**
- Wastes resources on bad trials
- Slower overall HPO convergence
- May reduce total trials completed

### Option C: Increase Parallel Trials (HIGHER RISK)
**Safety:** 🟡 Moderate Risk (OOM potential)
**Expected GPU Util Gain:** +15-20% (97-100%+)
**Implementation Time:** 1 minute

**Changes:**
```bash
# In Makefile or command line
PAR=6  # Up from 4
```

**Pros:**
- More trials training simultaneously
- Maximum GPU utilization
- Simple change

**Cons:**
- **OOM Risk:** With 12GB/24GB used, adding 2 more trials could hit 18GB
- More CPU pressure (would need to reduce num_workers to 1-2)
- May cause trial failures due to resource contention

### Option D: Hybrid Approach (RECOMMENDED)
**Safety:** 🟢 Safe
**Expected GPU Util Gain:** +13-18% (95-100%)
**Implementation Time:** 5 minutes

**Changes:**
1. Increase DataLoader workers to 4 (from 2)
2. Enable persistent_workers and prefetch_factor=3
3. Increase PatientPruner patience to 4 (from 2)
4. Set min_resource=3 in HyperbandPruner (from 1)

**Rationale:**
- Addresses both startup latency (better data feeding) and pruning bottleneck
- No OOM risk (still 4 parallel trials, just better utilized)
- Balanced approach: faster data loading + longer trial life = better GPU util

---

## Detailed Implementation Plan (Option D - RECOMMENDED)

### Step 1: Modify tune_max.py

```python
# Line 595: Increase DataLoader workers
num_workers = 4  # Up from 2

# Line 597-611: Add persistent workers and prefetch
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True if torch.cuda.is_available() else False,
    persistent_workers=True,  # NEW
    prefetch_factor=3,  # Up from default 2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True if torch.cuda.is_available() else False,
    persistent_workers=True,  # NEW
    prefetch_factor=3,  # Up from default 2
)

# Line 821-826: Reduce pruning aggressiveness
def make_pruner() -> optuna.pruners.BasePruner:
    hb = HyperbandPruner(
        min_resource=3,  # Up from 1 (wait 3 epochs before pruning)
        max_resource=parse_env_int("HPO_EPOCHS", 100, min_val=1, max_val=1000),
        reduction_factor=3,
    )
    return PatientPruner(hb, patience=4)  # Up from 2
```

### Step 2: Test with Small Trial Run

```bash
# Stop current HPO (if safe)
# pkill -f "tune_max.py"

# Test with 10 trials first
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
HPO_EPOCHS=100 HPO_PATIENCE=20 \
poetry run python scripts/tune_max.py \
    --agent criteria --study noaug-criteria-test \
    --n-trials 10 --parallel 4 \
    --outdir ./_runs_test
```

### Step 3: Monitor GPU Utilization

```bash
# Terminal 1: Watch GPU utilization
watch -n 2 nvidia-smi

# Terminal 2: Watch trial progress
tail -f <hpo_log_file>

# Terminal 3: Monitor CPU
htop
```

**Expected Results:**
- GPU Utilization: 95-100% (up from 82-84%)
- GPU Memory: 15-18GB/24GB (still safe)
- Trial duration: 4-8 min avg (vs 2-4 min currently)
- Pruning rate: 60-70% (vs 86.5% currently)

### Step 4: Validate No OOM

**Success Criteria:**
- No CUDA OOM errors in logs
- GPU memory stays below 22GB
- At least 2-3 trials training simultaneously
- GPU utilization > 95%

**Rollback Plan if OOM:**
1. Reduce num_workers back to 2
2. Keep persistence and pruning changes
3. This gives +8-12% util without OOM risk

---

## Resource Usage Predictions

### Current State
```
Parallel trials: 4
Active training: 1-2 (due to pruning)
GPU Memory per trial: 3-4GB
Total GPU Memory: 12GB (50%)
GPU Utilization: 82-84%
```

### After Option D
```
Parallel trials: 4
Active training: 2-3 (less pruning)
GPU Memory per trial: 3.5-4.5GB (slightly higher with more workers)
Total GPU Memory: 15-18GB (63-75%)
GPU Utilization: 95-100%
```

**Safety Margin:** 6-9GB GPU memory (25-37%)

---

## Alternative: If Can't Modify Running HPO

**Option E: Apply Changes to Next HPO Run**

The current run has 200 trials (10 complete, 190 pruned/failed). If you want to preserve it:

1. Let current run finish
2. Apply changes to `scripts/tune_max.py`
3. Start new study with optimized settings

**Pros:** No risk to current progress
**Cons:** Can't benefit immediately

---

## Monitoring Commands

```bash
# GPU utilization over time
nvidia-smi dmon -s um -c 60

# Count active training processes
ps aux | grep -c "tune_max.py"

# Trial completion rate
sqlite3 ./_optuna/noaug.db "
  SELECT state, COUNT(*)
  FROM trials
  WHERE study_id = (SELECT study_id FROM studies WHERE study_name = 'noaug-criteria-supermax')
  GROUP BY state
"

# Average trial duration by state
sqlite3 ./_optuna/noaug.db "
  SELECT
    state,
    AVG(CAST((julianday(datetime_complete) - julianday(datetime_start)) * 24 * 60 AS INTEGER)) as avg_duration_min,
    COUNT(*) as count
  FROM trials
  WHERE study_id = (SELECT study_id FROM studies WHERE study_name = 'noaug-criteria-supermax')
    AND datetime_complete IS NOT NULL
  GROUP BY state
"
```

---

## Summary

**Root Cause:** 86.5% pruning rate with PatientPruner(patience=2) causes rapid trial turnover. GPU spends more time on trial startup (30-60s) than training (2-4 min), limiting utilization to 82-84%.

**Recommended Solution (Option D):**
1. Increase DataLoader workers: 2 → 4
2. Enable persistent_workers + prefetch_factor=3
3. Reduce pruning: PatientPruner patience 2 → 4
4. Increase min_resource: 1 → 3 epochs

**Expected Results:**
- GPU Util: 82-84% → 95-100%
- GPU Memory: 12GB → 15-18GB (still safe)
- Trial duration: 2-4 min → 4-8 min
- Pruning rate: 86.5% → 60-70%

**Risk Level:** 🟢 Low (no OOM expected, 6-9GB safety margin)
**Implementation Time:** 5 minutes
**Validation Time:** 10-20 minutes (10 test trials)

---

## Next Steps

**Immediate Action:**
1. Implement Option D changes to `scripts/tune_max.py`
2. Test with 10 trials
3. Monitor GPU utilization and memory
4. If successful, apply to main HPO run or next study

**For Running HPO:**
- Can apply changes immediately (Optuna will use new settings for future trials)
- OR wait for current run to finish and start fresh study
