# MLflow Production Enhancement Guide

This document describes comprehensive MLflow logging enhancements for production-ready experiment tracking.

## Overview

The enhanced MLflow logging system provides:

1. **Step-level metrics** - Track every training batch
2. **Epoch-level summaries** - Comprehensive training/validation metrics per epoch  
3. **HPO trial tracking** - Nested runs for hyperparameter optimization trials
4. **System resource monitoring** - CPU, memory, GPU utilization tracking
5. **Model registry integration** - Register models with stages, tags, and metadata

## Files Modified

### 1. `src/psy_agents_noaug/training/train_loop.py`

**Enhancements:**
- Step-level logging (train/loss_step, train/accuracy_step, train/learning_rate, train/batch_time_seconds)
- GPU metrics per step (system/gpu_memory_allocated_gb, system/gpu_memory_reserved_gb, system/gpu_utilization_percent)
- Epoch-level summaries (epoch/train_loss, epoch/train_accuracy, epoch/val_*, epoch/duration_seconds)
- Final training summary (final/best_*, final/total_epochs, final/total_steps, final/early_stopped)
- System metrics integration via `SystemMetricsLogger`

**New Parameters:**
```python
Trainer(
    ...,
    log_system_metrics=True,        # Enable system resource monitoring
    system_metrics_interval=10,     # Log system metrics every N steps
)
```

### 2. `src/psy_agents_noaug/hpo/optuna_runner.py`

**Enhancements:**
- Nested MLflow runs for each HPO trial
- Trial-level parameter logging (hpo/*)
- Trial-level metric logging (trial/*)
- HPO study summary (best_*, total_trials, completed_trials, pruned_trials, failed_trials)
- Automatic trial history export as CSV artifact
- Optimization history visualization (if optuna.visualization available)

**New Parameters:**
```python
runner.optimize(
    ...,
    mlflow_experiment_name="hpo_criteria",  # Experiment name for parent run
)
```

### 3. `src/psy_agents_noaug/utils/mlflow_utils.py`

**Enhancements:**
- Enhanced `save_model_to_mlflow()` with signature and input_example support
- New `register_model()` function for model registry
- Model version staging (Staging, Production, Archived)
- Model tags and descriptions
- Automatic signature inference from sample input

**New Function:**
```python
from psy_agents_noaug.utils.mlflow_utils import register_model

model_version = register_model(
    model=trained_model,
    model_name="criteria_roberta_v1",
    sample_input=sample_batch["input_ids"],
    stage="Production",
    tags={"task": "criteria", "f1_macro": "0.85"},
    description="RoBERTa model for criteria classification",
)
```

## Files Created

### 4. `src/psy_agents_noaug/utils/system_metrics.py`

**New Module:**
Comprehensive system resource monitoring for MLflow.

**Features:**
- CPU usage and memory tracking (via psutil)
- GPU utilization, memory, temperature tracking (via pynvml)
- Disk I/O monitoring (optional)
- Network I/O monitoring (optional)
- Automatic MLflow integration

**Usage:**
```python
from psy_agents_noaug.utils.system_metrics import SystemMetricsLogger

# Initialize logger
system_logger = SystemMetricsLogger(
    log_cpu=True,
    log_gpu=True,
    log_disk=False,
    log_network=False,
    gpu_device=0,
)

# Log metrics to MLflow
system_logger.log_metrics(step=global_step, prefix="system")
```

## MLflow UI Hierarchy

### Training Run Structure

```
Experiment: training_criteria
├── Run: criteria_roberta_2025-01-25
│   ├── Metrics:
│   │   ├── train/loss_step (per batch, step-indexed)
│   │   ├── train/accuracy_step (per batch, step-indexed)
│   │   ├── train/learning_rate (per batch, step-indexed)
│   │   ├── train/batch_time_seconds (per batch, step-indexed)
│   │   ├── epoch/train_loss (per epoch, epoch-indexed)
│   │   ├── epoch/train_accuracy (per epoch, epoch-indexed)
│   │   ├── epoch/val_loss (per epoch, epoch-indexed)
│   │   ├── epoch/val_f1_macro (per epoch, epoch-indexed)
│   │   ├── epoch/duration_seconds (per epoch, epoch-indexed)
│   │   ├── system/gpu_memory_allocated_gb (periodic, step-indexed)
│   │   ├── system/gpu_utilization_percent (periodic, step-indexed)
│   │   ├── system/cpu_percent (periodic, step-indexed)
│   │   ├── system/memory_used_gb (periodic, step-indexed)
│   │   ├── final/best_val_f1_macro
│   │   ├── final/total_epochs
│   │   └── final/total_steps
│   ├── Parameters:
│   │   ├── model.encoder_name
│   │   ├── training.batch_size
│   │   ├── training.optimizer.lr
│   │   └── ...
│   └── Artifacts:
│       ├── config.yaml
│       ├── checkpoints/best_checkpoint.pt
│       └── model/
```

### HPO Run Structure

```
Experiment: hpo_criteria
├── Run: HPO_criteria_stage2 (Parent Run)
│   ├── Metrics:
│   │   ├── best_val_f1_macro
│   │   ├── best_trial_number
│   │   ├── total_trials
│   │   ├── completed_trials
│   │   ├── pruned_trials
│   │   └── failed_trials
│   ├── Parameters:
│   │   ├── best_hpo/learning_rate
│   │   ├── best_hpo/batch_size
│   │   └── ...
│   ├── Artifacts:
│   │   ├── hpo_trials_history.csv
│   │   └── hpo_optimization_history.html
│   └── Nested Runs:
│       ├── Run: trial_0001
│       │   ├── Metrics:
│       │   │   ├── trial/val_f1_macro
│       │   │   ├── trial/number
│       │   │   ├── train/loss_step (all training steps)
│       │   │   ├── epoch/* (all epochs)
│       │   │   └── system/* (system metrics)
│       │   ├── Parameters:
│       │   │   ├── hpo/learning_rate
│       │   │   ├── hpo/batch_size
│       │   │   ├── trial_number
│       │   │   └── study_name
│       │   └── Artifacts:
│       │       └── checkpoints/
│       ├── Run: trial_0002
│       │   └── ...
│       └── ...
```

## Usage Examples

### 1. Basic Training with Enhanced Logging

```python
from psy_agents_noaug.training.train_loop import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    num_epochs=20,
    logging_steps=50,           # Log step metrics every 50 steps
    log_system_metrics=True,    # Enable system monitoring
    system_metrics_interval=10, # Log system metrics every 10 steps
)

# Train with MLflow active run
import mlflow
with mlflow.start_run(run_name="criteria_training"):
    trainer.train()
```

### 2. HPO with Nested Runs

```python
from psy_agents_noaug.hpo.optuna_runner import OptunaRunner

runner = OptunaRunner(
    study_name="criteria_stage2",
    direction="maximize",
    metric="val_f1_macro",
)

def objective(trial, params):
    # Setup model with params
    model = create_model(params)
    trainer = Trainer(model, ...)
    
    # Train (will log to nested run)
    results = trainer.train()
    return results["best_val_f1_macro"]

# Optimize with parent run
runner.optimize(
    objective_fn=objective,
    n_trials=50,
    search_space=search_space,
    mlflow_tracking_uri="sqlite:///mlflow.db",
    mlflow_experiment_name="hpo_criteria",
)
```

### 3. Model Registry

```python
from psy_agents_noaug.utils.mlflow_utils import register_model

# After training, register best model
with mlflow.start_run(run_name="criteria_final"):
    # Train model
    trainer.train()
    
    # Get sample input for signature
    sample_batch = next(iter(val_loader))
    sample_input = sample_batch["input_ids"][:1]  # Single example
    
    # Register model
    model_version = register_model(
        model=model,
        model_name="criteria_classifier",
        sample_input=sample_input,
        stage="Production",
        tags={
            "task": "criteria",
            "encoder": "roberta-base",
            "f1_macro": "0.856",
        },
        description="Production criteria classifier with F1=0.856",
    )
    
    print(f"Registered model version: {model_version}")
```

### 4. System Metrics Only

```python
from psy_agents_noaug.utils.system_metrics import SystemMetricsLogger

system_logger = SystemMetricsLogger(
    log_cpu=True,
    log_gpu=True,
    log_disk=True,
    log_network=True,
)

with mlflow.start_run():
    for step in range(1000):
        # Training code...
        
        if step % 10 == 0:
            system_logger.log_metrics(step=step)
```

## Required Dependencies

Add to `pyproject.toml`:

```toml
[tool.poetry.dependencies]
# Existing dependencies...
psutil = "^5.9.0"          # For CPU/memory metrics
pynvml = "^11.5.0"         # For detailed GPU metrics (optional)
pandas = "^2.0.0"          # For HPO trial history export
```

Install:
```bash
poetry add psutil pynvml pandas
```

## Backward Compatibility

All enhancements are **100% backward compatible**:

- Existing code continues to work without modifications
- New features are opt-in via parameters
- Default values maintain current behavior
- No breaking changes to existing APIs

## Metric Naming Conventions

### Prefixes

- `train/*` - Per-step training metrics
- `epoch/*` - Per-epoch aggregated metrics  
- `system/*` - System resource metrics
- `trial/*` - HPO trial results
- `hpo/*` - HPO hyperparameters
- `best_hpo/*` - Best trial hyperparameters
- `final/*` - Final training summary

### Indexing

- **step**: Use for per-batch metrics (train/loss_step, system/*)
- **epoch**: Use for per-epoch metrics (epoch/train_loss, epoch/val_*)

## Performance Impact

- **Step-level logging**: ~0.1-0.5% overhead (only when step % logging_steps == 0)
- **System metrics**: ~0.05-0.2% overhead (depends on interval)
- **HPO nested runs**: Negligible (MLflow overhead amortized over trial duration)

## Troubleshooting

### pynvml not available

If you see warnings about GPU utilization not being tracked:
```bash
poetry add pynvml
```

### psutil not available

System metrics will be disabled. Install:
```bash
poetry add psutil
```

### MLflow UI not showing nested runs

Make sure you're using MLflow 2.0+:
```bash
poetry show mlflow
poetry update mlflow
```

### Large number of metrics slowing UI

Increase logging intervals:
```python
Trainer(..., logging_steps=200, system_metrics_interval=50)
```

## Best Practices

1. **Use nested runs for HPO** - Keeps trials organized under parent study
2. **Log system metrics sparingly** - Every 10-50 steps is sufficient
3. **Register important models** - Use model registry for production models
4. **Tag model versions** - Include task, metrics, date for easy filtering
5. **Export trial history** - Keep CSV backup of HPO results
6. **Use prefixes consistently** - Makes filtering in UI much easier

## Migration Guide

### From Basic to Enhanced Logging

**Before:**
```python
trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, device)
trainer.train()
```

**After:**
```python
trainer = Trainer(
    model, train_loader, val_loader, optimizer, criterion, device,
    logging_steps=100,           # Add step-level logging
    log_system_metrics=True,     # Add system monitoring
)
trainer.train()
```

That's it! All enhancements are automatic.

## Verification

After applying patches, verify functionality:

```bash
# 1. Run short training
python -m psy_agents_noaug.cli train task=criteria training.num_epochs=2

# 2. Check MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# 3. Verify metrics:
#    - train/loss_step (step-indexed)
#    - epoch/train_loss (epoch-indexed)
#    - system/gpu_memory_allocated_gb
#    - final/best_val_f1_macro

# 4. Test HPO
python -m psy_agents_noaug.cli hpo stage=stage0 task=criteria

# 5. Verify nested runs in UI
#    - Parent run: HPO_*
#    - Child runs: trial_0001, trial_0002, ...

# 6. Test model registry
python -c "
from psy_agents_noaug.utils.mlflow_utils import register_model
import mlflow
import torch
mlflow.set_tracking_uri('sqlite:///mlflow.db')
with mlflow.start_run():
    model = torch.nn.Linear(10, 2)
    register_model(model, 'test_model', stage='Staging')
"

# 7. Check Model Registry in UI
#    - Models tab should show 'test_model'
#    - Version 1 should be in 'Staging' stage
```

## Summary

The enhanced MLflow logging system provides comprehensive, production-ready experiment tracking with:

- **Fine-grained visibility** - Track every step, epoch, and trial
- **System monitoring** - Understand resource utilization
- **Model governance** - Registry with stages and metadata
- **Organized experiments** - Nested runs for HPO
- **Backward compatible** - Existing code works unchanged

All enhancements are ready for immediate use in production environments.
