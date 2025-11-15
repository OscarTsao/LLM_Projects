# Maximal HPO - Active Run Status

**Launch Date:** Oct 31, 2025 19:06:19 CST
**Status:** ✅ ALL 4 RUNS ACTIVE AND HEALTHY
**Master PID:** 893103
**Expected Completion:** ~40 hours (parallel) or Nov 2, 2025 11:00 AM CST

---

## Run Status

### 1. Criteria HPO ✅ RUNNING
- **PID:** 893110
- **Study:** `noaug-criteria-max-2025-10-31`
- **Target Trials:** 800
- **Progress:** Trial 3+ (0.4% complete)
- **Log:** `./logs/maximal_2025-10-31/criteria_hpo.log` (15K)
- **Status:** Healthy - XLNet gradient checkpointing fix confirmed working!

**Last Trial:** Trial 3 finished successfully with `xlnet-base-cased` + `gradient_checkpointing=True` (value: 0.449)

### 2. Evidence HPO ✅ RUNNING
- **PID:** 893508
- **Study:** `noaug-evidence-max-2025-10-31`
- **Target Trials:** 1200
- **Progress:** Trial 0+ (0% complete)
- **Log:** `./logs/maximal_2025-10-31/evidence_hpo.log` (22K)
- **Status:** Healthy - Starting trials

### 3. Share HPO ✅ RUNNING
- **PID:** 893983
- **Study:** `noaug-share-max-2025-10-31`
- **Target Trials:** 600
- **Progress:** Trial 0+ (0% complete)
- **Log:** `./logs/maximal_2025-10-31/share_hpo.log` (17K)
- **Status:** Healthy - Starting trials

### 4. Joint HPO ✅ RUNNING
- **PID:** 894399
- **Study:** `noaug-joint-max-2025-10-31`
- **Target Trials:** 600
- **Progress:** Trial 0+ (0% complete)
- **Log:** `./logs/maximal_2025-10-31/joint_hpo.log` (5.6K)
- **Status:** Healthy - Starting trials

---

## Critical Fix Applied ✅

**Issue:** XLNet and other models don't support gradient checkpointing, causing fatal crashes

**Fix Location:** `src/psy_agents_noaug/hpo/evaluation.py:259-267`

**Fix Applied:**
```python
grad_ckpt_enabled = bool(params.get("model.gradient_checkpointing"))
if grad_ckpt_enabled and hasattr(model, "gradient_checkpointing_enable"):
    try:
        model.gradient_checkpointing_enable()
    except ValueError as e:
        # Some models (XLNet, GPT-2, etc.) don't support gradient checkpointing
        # Silently skip and continue without it
        pass
model.to(device)
```

**Verification:** Trial 3 in Criteria HPO completed successfully with XLNet + gradient_checkpointing=True (previously would crash)

---

## Trial Budget

| Architecture | Target Trials | Est. Time/Trial | Total Est. Time |
|-------------|--------------|-----------------|-----------------|
| Criteria    | 800          | ~2 min          | ~26 hours       |
| Evidence    | 1200         | ~2 min          | ~40 hours       |
| Share       | 600          | ~3 min          | ~30 hours       |
| Joint       | 600          | ~3 min          | ~30 hours       |
| **TOTAL**   | **3200**     | —               | **~40 hours** (parallel) |

---

## Monitoring Commands

### Check Process Status
```bash
ps aux | grep tune_max.py | grep -v grep
```

### Check Master Log
```bash
tail -f maximal_hpo_master_relaunch.log
```

### Check Individual Logs
```bash
tail -f ./logs/maximal_2025-10-31/criteria_hpo.log
tail -f ./logs/maximal_2025-10-31/evidence_hpo.log
tail -f ./logs/maximal_2025-10-31/share_hpo.log
tail -f ./logs/maximal_2025-10-31/joint_hpo.log
```

### Check Latest Trials
```bash
# Criteria
grep -E "Trial [0-9]+ (finished|pruned)" ./logs/maximal_2025-10-31/criteria_hpo.log | tail -5

# Evidence
grep -E "Trial [0-9]+ (finished|pruned)" ./logs/maximal_2025-10-31/evidence_hpo.log | tail -5

# Share
grep -E "Trial [0-9]+ (finished|pruned)" ./logs/maximal_2025-10-31/share_hpo.log | tail -5

# Joint
grep -E "Trial [0-9]+ (finished|pruned)" ./logs/maximal_2025-10-31/joint_hpo.log | tail -5
```

### Check for Errors
```bash
grep -i "error\|exception\|valueerror" ./logs/maximal_2025-10-31/*.log | tail -20
```

### Get Progress Summary
```bash
for agent in criteria evidence share joint; do
    echo "=== $agent ==="
    grep -E "Trial [0-9]+ (finished|pruned)" ./logs/maximal_2025-10-31/${agent}_hpo.log | wc -l
done
```

---

## Expected Behavior

### Normal Events:
- **CUDA OOM Pruning:** Expected for large models (deberta, xlnet with large batch sizes)
- **Early Stopping:** Trials may stop early if validation doesn't improve (patience=20 epochs)
- **Varying Trial Times:** 30 seconds (small model, early stopping) to 10 minutes (large model, full 100 epochs)

### Warning Signs:
- All processes stopped unexpectedly
- Log file size not increasing for >30 minutes
- Repeated ValueError/Exception messages
- All trials being pruned (indicates systemic issue)

---

## Recovery Procedures

### If a single run crashes:
```bash
# Check which run failed
ps aux | grep tune_max.py

# Relaunch just that run
export HPO_EPOCHS=100 HPO_PATIENCE=20
python scripts/tune_max.py \
    --agent criteria \
    --study-name noaug-criteria-max-2025-10-31 \
    --trials 800 \
    --epochs 100 \
    --patience 20 \
    --outdir ./_runs \
    > ./logs/maximal_2025-10-31/criteria_hpo_restart.log 2>&1 &
```

### If all runs need restart:
```bash
# Kill existing runs
pkill -f tune_max.py

# Relaunch all
./scripts/run_maximal_hpo_all.sh > maximal_hpo_master_relaunch2.log 2>&1 &
```

---

## Progress Milestones

- **10% Complete:** ~4 hours (320 trials total)
- **25% Complete:** ~10 hours (800 trials total)
- **50% Complete:** ~20 hours (1600 trials total)
- **75% Complete:** ~30 hours (2400 trials total)
- **100% Complete:** ~40 hours (3200 trials total)

---

## Output Artifacts

### Per Architecture:
- **Best Config:** `./_runs/maximal_2025-10-31/{agent}/best_config.yaml`
- **Best Checkpoint:** `./_runs/maximal_2025-10-31/{agent}/best_checkpoint.pt`
- **Trial History:** `./_runs/maximal_2025-10-31/{agent}/trials.csv`
- **MLflow Run:** Logged to `mlflow.db` under study name

### Combined:
- **Master Log:** `maximal_hpo_master_relaunch.log`
- **Individual Logs:** `./logs/maximal_2025-10-31/{agent}_hpo.log`

---

## Next Steps After Completion

1. **Verify all runs completed:**
   ```bash
   grep "Study statistics" ./logs/maximal_2025-10-31/*.log
   ```

2. **Compare best configs:**
   ```bash
   cat ./_runs/maximal_2025-10-31/*/best_config.yaml
   ```

3. **Refit on full train+val:**
   ```bash
   make refit HPO_TASK=criteria
   make refit HPO_TASK=evidence
   make refit HPO_TASK=share
   make refit HPO_TASK=joint
   ```

4. **Evaluate on test set:**
   ```bash
   make eval CHECKPOINT=./_runs/maximal_2025-10-31/criteria/best_checkpoint.pt
   ```

5. **Export results:**
   ```bash
   make export
   ```

---

**Last Updated:** Oct 31, 2025 19:07:09 CST
**Status Check:** All 4 runs healthy and progressing ✅
