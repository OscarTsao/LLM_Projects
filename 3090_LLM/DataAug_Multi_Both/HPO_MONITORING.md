# HPO Production Monitoring Guide

**Started:** 2025-10-11 15:43
**Study:** mental_health_hpo_production
**Trials:** 500
**Status:** 🟢 RUNNING

---

## Quick Status Check

```bash
# View current status
make hpo-results

# Watch live logs
tail -f hpo_production_run.log

# Check GPU usage
watch -n 1 nvidia-smi
```

---

## Current Progress

### Trial 1
- **Model:** SpanBERT/spanbert-large-cased
- **Status:** Training (Epoch 1/8, 7% complete)
- **Loss:** 6.2983 → 5.3984 (decreasing)
- **Speed:** ~4.3 it/s

---

## Monitoring Commands

### Real-time Status
```bash
# Study statistics (refreshes every 5 seconds)
watch -n 5 'make hpo-results'

# Live training logs
tail -f hpo_production_run.log

# Live GPU monitoring
nvidia-smi -l 1
```

### HPO Progress
```python
import optuna

study = optuna.load_study(
    study_name='mental_health_hpo_production',
    storage='sqlite:///experiments/hpo_production.db'
)

print(f"Total trials: {len(study.trials)}")
print(f"COMPLETE: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
print(f"RUNNING: {len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])}")
print(f"PRUNED: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) > 0:
    print(f"\nBest value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
```

### Check Trial Outputs
```bash
# List trials
ls -lh experiments/trial_*

# View latest trial checkpoints
ls -lh experiments/trial_*/checkpoints/

# Check trial logs (replace UUID with actual trial ID)
tail -f experiments/trial_<uuid>/logs/train.log
```

---

## Expected Timeline

### Per Trial
- **Training:** 8 epochs × ~10-20 min/epoch = 80-160 min
- **Evaluation:** ~5-10 min
- **Total:** ~90-170 min per trial (1.5-3 hours)

### Full Study (500 trials)
- **Sequential:** ~750-1500 hours (31-62 days)
- **Note:** This is a very long run. Consider reducing to 50-100 trials for initial search

---

## Success Indicators

### ✅ Good Signs
- Loss decreasing consistently
- No CUDA OOM errors
- Progress bars updating
- Checkpoints being saved
- Trials completing without errors

### ⚠️ Warning Signs
- Loss not decreasing (may indicate bad hyperparameters - will be pruned)
- Repeated CUDA OOM errors (reduce batch_size in search space)
- Trials all failing (check logs for systematic errors)
- Disk space running out (check retention policy)

---

## Common Issues & Solutions

### 1. CUDA Out of Memory
```bash
# Check current batch sizes in failed trials
grep "batch_size" experiments/trial_*/config.json

# Solution: Reduce max batch_size in search_space.py
# Edit: src/dataaug_multi_both/hpo/search_space.py
# Change: params["batch_size"] = trial.suggest_categorical("batch_size", [8, 16])
```

### 2. Disk Space Running Out
```bash
# Check disk usage
df -h experiments/

# Clean old trials (keeps best K)
# Automatic via retention policy

# Manual cleanup if needed
rm -rf experiments/trial_<old_uuid>
```

### 3. Training Stuck
```bash
# Check if process is alive
ps aux | grep "python.*train.*hpo"

# Check recent log activity
tail -100 hpo_production_run.log

# If truly stuck, restart:
pkill -f "python.*train.*hpo"
make hpo-production  # Will resume from database
```

---

## Stopping & Resuming

### Stop HPO
```bash
# Graceful stop (wait for current trial to finish)
pkill -SIGINT -f "python.*train.*hpo"

# Force stop
pkill -9 -f "python.*train.*hpo"
```

### Resume HPO
```bash
# HPO automatically resumes from database
make hpo-production

# Or manually specify resume
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m dataaug_multi_both.cli.train hpo \
    --study-name mental_health_hpo_production \
    --n-trials 500 \
    --experiments-dir experiments \
    --keep-last-n 1 \
    --keep-best-k 5 \
    --study-db experiments/hpo_production.db \
    --mlflow-uri sqlite:///experiments/mlflow_db/mlflow.db \
    --dataset-id irlab-udc/redsm5
```

---

## Viewing Results

### MLflow UI
```bash
make mlflow-ui
# Navigate to: http://localhost:5000

# Features:
# - Compare trials side-by-side
# - View training curves
# - Download best model checkpoints
# - Export results
```

### Export Best Configuration
```bash
# Export to JSON
python -c "
import optuna, json
study = optuna.load_study(study_name='mental_health_hpo_production', storage='sqlite:///experiments/hpo_production.db')
with open('best_hpo_config.json', 'w') as f:
    json.dump(study.best_params, f, indent=2)
print(f'Best config saved to: best_hpo_config.json')
print(f'Best F1 score: {study.best_value:.4f}')
"
```

### Visualization
```python
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice
)

study = optuna.load_study(
    study_name='mental_health_hpo_production',
    storage='sqlite:///experiments/hpo_production.db'
)

# Optimization history
fig = plot_optimization_history(study)
fig.write_html('optimization_history.html')

# Parameter importances
fig = plot_param_importances(study)
fig.write_html('param_importances.html')

# Parallel coordinate plot
fig = plot_parallel_coordinate(study)
fig.write_html('parallel_coordinate.html')

# Slice plots
fig = plot_slice(study)
fig.write_html('slice_plots.html')
```

---

## Performance Optimization

### Reduce Trial Count (Recommended)
```bash
# Edit Makefile or run directly with fewer trials
make hpo  # 50 trials instead of 500
```

### Narrow Search Space
Edit `src/dataaug_multi_both/hpo/search_space.py`:
```python
# Focus on best models only
params["backbone"] = trial.suggest_categorical(
    "backbone",
    [
        "microsoft/deberta-v3-base",  # Usually performs well
        "google-bert/bert-base-uncased",
        "FacebookAI/xlm-roberta-base",
    ],
)

# Reduce hyperparameter ranges
params["batch_size"] = trial.suggest_categorical("batch_size", [16])  # Fixed
params["epochs"] = trial.suggest_categorical("epochs", [8])  # Fixed
```

---

## Estimated Completion

**Current Configuration:**
- Trials: 500
- Time per trial: ~90-170 min
- Total time: ~750-1500 hours (31-62 days)

**Recommended:**
- Start with `make hpo` (50 trials, ~3-7 days)
- Analyze results
- Run focused search on best configurations

---

**Monitor regularly and adjust based on initial results!**

**Log file:** `hpo_production_run.log`
**Database:** `experiments/hpo_production.db`
**MLflow:** `experiments/mlflow_db/`
