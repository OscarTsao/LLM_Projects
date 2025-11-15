# MLflow + Optuna Integration Plan

**Date**: 2025-10-27
**Goal**: Integrate MLflow logging alongside Optuna HPO tracking

---

## 🎯 Architecture Overview

### Dual Tracking System

**Two complementary databases**:

1. **Optuna DB** (`_optuna/dataaug.db`) - HPO Study Management
   - Study creation and trials
   - Hyperparameter sampling
   - Trial pruning decisions
   - Best trial selection
   - **Purpose**: HPO orchestration

2. **MLflow DB** (`mlflow.db`) - Experiment Tracking & Model Registry
   - Metrics per epoch
   - Hyperparameters logging
   - Model artifacts (checkpoints)
   - Model registration for production
   - **Purpose**: Experiment tracking and model versioning

---

## 🔧 Integration Strategy

### Why Both?

| Feature | Optuna | MLflow |
|---------|--------|--------|
| **HPO search** | ✅ Primary | ❌ |
| **Trial pruning** | ✅ Primary | ❌ |
| **Best params** | ✅ Primary | ✅ Copy |
| **Epoch metrics** | ❌ | ✅ Primary |
| **Model artifacts** | ❌ | ✅ Primary |
| **Model registry** | ❌ | ✅ Primary |
| **Visualization** | Web UI | ✅ Better UI |

**Conclusion**: Use both!
- Optuna: HPO orchestration
- MLflow: Rich metrics and model lifecycle

---

## 📋 Implementation Plan

### 1. Update Database References

**Files to update**:
- `scripts/tune_max.py` - Change `noaug.db` → `dataaug.db`
- Any other scripts referencing the old name

**Changes**:
```python
# OLD
storage = 'sqlite:///_optuna/noaug.db'

# NEW
storage = 'sqlite:///_optuna/dataaug.db'
```

---

### 2. Add MLflow Callbacks in tune_max.py

**Strategy**: Log to MLflow within each trial

**Key integration points**:

#### A. Trial Start
```python
def objective(trial):
    # Start MLflow run
    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        # Log Optuna trial info
        mlflow.set_tag("optuna_trial", trial.number)
        mlflow.set_tag("optuna_study", study_name)

        # Log hyperparameters
        mlflow.log_params(trial.params)

        # Train and log metrics
        ...
```

#### B. Epoch Callback
```python
def on_epoch(trial, epoch, metric, loss):
    # Log to MLflow
    mlflow.log_metric("val_metric", metric, step=epoch)
    mlflow.log_metric("val_loss", loss, step=epoch)

    # Report to Optuna (for pruning)
    trial.report(metric, epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()
```

#### C. Trial End
```python
# Log final results
mlflow.log_metric("final_metric", best_metric)

# Save model artifact (if best trial)
if is_best_trial:
    mlflow.pytorch.log_model(model, "model")
    mlflow.log_dict(config, "config.json")
```

---

### 3. MLflow Experiment Organization

**Experiment structure**:
```
Evidence-HPO/                    (MLflow Experiment)
├── trial_0/                     (MLflow Run)
│   ├── params: {aug.enabled: true, model.name: roberta, ...}
│   ├── metrics: {val_metric, val_loss} per epoch
│   └── artifacts: model checkpoint (if best)
├── trial_1/
├── ...
└── trial_106/

Criteria-HPO/                    (MLflow Experiment)
└── (similar structure)
```

---

### 4. Code Changes

**File**: `scripts/tune_max.py`

**Section 1: Setup** (add near imports)
```python
# MLflow setup
if _HAS_MLFLOW:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # Will create experiment per agent/study
```

**Section 2: Objective function** (wrap with MLflow)
```python
def objective(trial: optuna.Trial) -> float:
    # Start MLflow run for this trial
    experiment_name = f"{args.agent}-{args.study}"

    if _HAS_MLFLOW:
        mlflow.set_experiment(experiment_name)
        mlflow_run = mlflow.start_run(run_name=f"trial_{trial.number}")
        mlflow.set_tags({
            "optuna_trial": trial.number,
            "optuna_study": args.study,
            "agent": args.agent,
        })

    try:
        # Sample hyperparameters (Optuna)
        cfg = sample_config(trial)

        # Log params to MLflow
        if _HAS_MLFLOW:
            mlflow.log_params(trial.params)

        # Create callback for epoch logging
        def epoch_callback(epoch, metric, loss):
            # Log to MLflow
            if _HAS_MLFLOW:
                mlflow.log_metrics({
                    "val_metric": metric,
                    "val_loss": loss,
                    "epoch": epoch,
                }, step=epoch)

            # Report to Optuna (pruning)
            trial.report(metric, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Train model
        result = run_training_eval(cfg, {"on_epoch": epoch_callback})

        # Log final result
        if _HAS_MLFLOW:
            mlflow.log_metric("final_metric", result["primary"])
            mlflow.log_metric("runtime_s", result["runtime_s"])

            # Save config
            mlflow.log_dict(cfg, "config.json")

        return result["primary"]

    finally:
        if _HAS_MLFLOW and mlflow.active_run():
            mlflow.end_run()
```

---

### 5. Model Registration (Best Trial)

**After HPO completes**:
```python
# Get best trial from Optuna
best_trial = study.best_trial

# Register model in MLflow
if _HAS_MLFLOW:
    # Find corresponding MLflow run
    experiment = mlflow.get_experiment_by_name(f"{agent}-{study_name}")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.optuna_trial = '{best_trial.number}'"
    )

    if len(runs) > 0:
        best_run_id = runs.iloc[0].run_id

        # Register model
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(
            model_uri,
            name=f"{agent}-best-model",
            tags={
                "optuna_trial": best_trial.number,
                "value": best_trial.value,
                "study": study_name,
            }
        )
```

---

## 📊 Benefits

### For Each Trial:
✅ **Optuna**: Manages HPO, pruning, best trial selection
✅ **MLflow**: Logs detailed metrics, saves artifacts

### After HPO:
✅ **Optuna DB**: Query best hyperparameters
✅ **MLflow UI**: Visualize training curves, compare trials
✅ **MLflow Registry**: Deploy best model to production

### Workflow:
```
1. Optuna samples params → MLflow logs them
2. Training happens → MLflow logs epoch metrics
3. Optuna prunes poor trials → MLflow marks as pruned
4. Best trial found → MLflow registers model
5. Deploy from MLflow registry
```

---

## 🔍 Access Patterns

### During HPO:
```bash
# Monitor Optuna progress
python -c "
import optuna
study = optuna.load_study(
    study_name='aug-evidence-production',
    storage='sqlite:///_optuna/dataaug.db'
)
print(f'Trials: {len(study.trials)}, Best: {study.best_value}')
"

# Monitor MLflow metrics
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Visit http://localhost:5000
```

### After HPO:
```python
# Get best config from Optuna
import optuna
study = optuna.load_study(...)
best_params = study.best_trial.params

# Load best model from MLflow
import mlflow
model = mlflow.pytorch.load_model(
    f"models:/{agent}-best-model/production"
)
```

---

## 📝 Implementation Checklist

- [ ] Update database path: `noaug.db` → `dataaug.db`
- [ ] Add MLflow run wrapper in objective function
- [ ] Add epoch callback for metrics logging
- [ ] Log hyperparameters at trial start
- [ ] Log final metrics at trial end
- [ ] Add model artifact logging for best trials
- [ ] Add model registration after HPO
- [ ] Update documentation with dual-tracking workflow
- [ ] Test with 2-trial run
- [ ] Verify MLflow UI shows trials

---

## 🚀 Usage Example

```bash
# Run HPO with dual logging
python scripts/tune_max.py \
    --agent evidence \
    --study aug-evidence-mlflow-test \
    --n-trials 10 \
    --parallel 1 \
    --outdir outputs

# This will:
# 1. Store HPO data in _optuna/dataaug.db
# 2. Log all metrics/artifacts to mlflow.db
# 3. Create MLflow experiment: "evidence-aug-evidence-mlflow-test"
# 4. Register best model after completion

# View results
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## 💾 Database Roles Summary

| Database | Path | Role | Contents |
|----------|------|------|----------|
| **Optuna** | `_optuna/dataaug.db` | HPO orchestration | Studies, trials, sampling, pruning |
| **MLflow** | `mlflow.db` | Experiment tracking | Metrics, params, models, registry |

**Both work together**: Optuna orchestrates, MLflow records! 🎯
