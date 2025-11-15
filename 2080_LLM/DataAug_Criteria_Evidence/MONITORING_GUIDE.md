# SUPERMAX HPO Monitoring Guide

**Status:** ✅ RUNNING
**Started:** 2025-10-25 18:14 CST
**PID:** See `supermax.pid`

---

## Quick Status Check

```bash
# 1. Check if running
ps aux | grep tune_max | grep -v grep | wc -l
# Should show: 1-5 processes (1 main + workers)

# 2. Check latest progress
tail -20 supermax_final.log | grep -E "Trial [0-9]+"

# 3. Check GPU usage
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
# Target: >90% during training, ~10-30% between trials

# 4. Check monitor alerts
tail -10 hpo_monitor.log | grep -E "ALERT|WARNING"
```

---

## Real-Time Monitoring Commands

### Watch GPU Usage (updates every 2 seconds)
```bash
watch -n 2 nvidia-smi
```

### Follow HPO Logs
```bash
tail -f supermax_final.log
```

### Follow Resource Monitor
```bash
tail -f hpo_monitor.log
```

### Check Trial Count
```bash
# Count completed trials
tail -100 supermax_final.log | grep "Trial [0-9]" | wc -l
```

---

## Expected Behavior

### Normal Trial Flow
1. **Trial starts**: GPU mem increases to 15-20GB
2. **Training**: GPU util >90% for 2-5 minutes
3. **Validation**: GPU util >95% for 10-20 seconds
4. **Trial completes**: GPU mem drops to ~3GB
5. **Repeat**: Next trial starts

### Pruned Trials (10-15% expected)
```
[I 2025-10-25 18:14:38] Trial 8 pruned. Invalid configuration: RuntimeError...
```

**This is NORMAL!** Optuna explores invalid configs and prunes them.

Common pruning reasons:
- `max_length > model.max_position_embeddings` (544 > 512)
- `pooler_output` not available (rare with our fix)
- Early stopping triggered (performance too low)

---

## Performance Metrics

### Current Stats (as of Trial 9)
- **Trials completed**: 9
- **Pruned**: 2 (22% - will decrease as Optuna learns)
- **GPU utilization**: 3% (between trials, normal)
- **GPU memory**: 2.9GB / 24.5GB (idle)

### Target Metrics
- **Trial success rate**: >85% (after first 50 trials)
- **GPU utilization (training)**: >90%
- **GPU memory (training)**: 15-20GB
- **Trials per hour**: ~20-30 (3-4 min/trial with ES)

---

## Progress Tracking

### Trials Remaining
```bash
# Criteria: 5000 trials total
COMPLETED=$(tail -500 supermax_final.log | grep -c "Trial [0-9]* pruned\|Trial [0-9]* finished")
REMAINING=$((5000 - COMPLETED))
echo "Completed: $COMPLETED / 5000"
echo "Remaining: $REMAINING"
echo "ETA: $((REMAINING * 3 / 60)) hours"
```

### Overall Progress (all 4 architectures)
```bash
# Architecture status
grep -E "Running.*for (Criteria|Evidence|Share|Joint)" supermax_final.log | tail -1
```

---

## Troubleshooting

### If GPU util stays < 50%
```bash
# Check if trial is actually running or just loading model
tail -50 supermax_final.log | grep -E "Training|Epoch"

# If no training happening, might be stuck on pruned trials
# Wait for 10-20 trials - Optuna will learn to avoid invalid configs
```

### If RAM > 90%
```bash
# Current RAM usage
free -h | awk '/^Mem:/ {print "RAM: "$3" / "$2" ("int($3*100/$2)"%)"}'

# If > 90%, reduce workers and restart
kill $(cat supermax.pid)
NUM_WORKERS=14 nohup make tune-all-supermax > supermax_final.log 2>&1 & echo $! > supermax.pid
```

### If process died
```bash
# Check if running
ps -p $(cat supermax.pid) > /dev/null 2>&1 && echo "Running" || echo "Died"

# Check last error in log
tail -100 supermax_final.log | grep -E "ERROR|Exception|Traceback" -A 5

# Restart from where it left off (Optuna resumes automatically)
NUM_WORKERS=18 nohup make tune-all-supermax > supermax_final.log 2>&1 & echo $! > supermax.pid
```

---

## Stopping/Pausing

### Graceful Stop (finish current trial)
```bash
# Send SIGTERM (allows current trial to finish)
kill $(cat supermax.pid)

# Wait for trial to complete (can take 1-5 minutes)
tail -f supermax_final.log
```

### Force Stop (immediate)
```bash
# Kill all related processes
pkill -9 -f tune_max.py

# Verify stopped
ps aux | grep tune_max | grep -v grep
```

### Resume After Stop
```bash
# Optuna automatically resumes from last trial
NUM_WORKERS=18 nohup make tune-all-supermax > supermax_final.log 2>&1 & echo $! > supermax.pid
```

---

## Key Log Patterns

### Successful Trial
```
[I 2025-10-25 18:20:15] Trial 15 finished with value: 0.8234 and parameters: {...}
```

### Pruned Trial (invalid config)
```
[I 2025-10-25 18:21:30] Trial 16 pruned. Invalid configuration: RuntimeError...
```

### Early Stopped Trial (performance)
```
EarlyStopping triggered at epoch 12 (patience=20)
[I 2025-10-25 18:23:45] Trial 17 pruned at step 12 with metric 0.6543
```

### OOM Error (shouldn't happen with our fixes)
```
[ERROR] Trial 18 failed with config error: OutOfMemoryError: CUDA out of memory
```

**If you see OOM**: Reduce batch size cap in `tune_max.py` from 32 to 24.

---

## Database Monitoring

### Check Study Status
```bash
# Database size (should grow over time)
ls -lh _optuna/noaug.db

# Number of trials in database
sqlite3 _optuna/noaug.db "SELECT COUNT(*) FROM trials WHERE study_id=(SELECT study_id FROM studies WHERE study_name='noaug-criteria-supermax');"
```

### Best Trial So Far
```bash
# View best parameters (after 50+ trials)
cat ./_runs/noaug-criteria-supermax_topk.json | jq '.[0]'
```

---

## Timeline Expectations

### Criteria (Current Phase)
- **Trials**: 5,000
- **Duration**: ~10-15 days with EarlyStopping
- **Progress milestones**:
  - Day 1: ~500 trials (10%)
  - Day 3: ~1,500 trials (30%)
  - Day 7: ~3,500 trials (70%)
  - Day 10-15: Complete

### Full Run (All 4 Architectures)
- **Criteria**: Days 1-15
- **Evidence**: Days 16-38
- **Share**: Days 39-48
- **Joint**: Days 49-58
- **Total**: ~58 days (with EarlyStopping)

---

## Daily Checklist

**Every day, check:**
- [ ] Process still running: `ps -p $(cat supermax.pid)`
- [ ] No OOM errors: `grep -i "out of memory" supermax_final.log`
- [ ] Trial success rate: Should improve to >85% after 50 trials
- [ ] Disk space: `df -h | grep " /$"` (need >100GB free)
- [ ] Monitor log health: `tail -5 hpo_monitor.log`

**Every week, check:**
- [ ] Progress on track: Compare trial count to timeline expectations
- [ ] Best model improving: Check `*_topk.json` files
- [ ] No long-term issues: Review full `supermax_final.log`

---

## Contact Information

**Documentation:**
- Main guide: `SUPERMAX_READY_TO_RUN.md`
- Optimization details: `SUPERMAX_OPTIMIZATION.md`
- Fixes applied: `SUPERMAX_FIXES_APPLIED.md`

**Logs:**
- HPO output: `supermax_final.log`
- Resource monitoring: `hpo_monitor.log`
- MLflow: `mlruns/` directory

**Quick Help:**
```bash
# View all documentation
ls -1 *.md | grep -i supermax

# Search for specific error
grep -i "error_keyword" supermax_final.log -A 10
```

---

## Emergency Contacts

If critical issues occur:
1. Check `supermax_final.log` for errors
2. Review this monitoring guide
3. Check `SUPERMAX_FIXES_APPLIED.md` troubleshooting section
4. Document issue in a new `ISSUE_$(date +%Y%m%d).md` file

**Good luck monitoring! The run will take ~58 days total.**
