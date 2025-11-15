# MLflow Enhancement Quick Reference

## Installation

```bash
cd /media/cvrlab308/cvrlab308_4090/YuNing/NoAug_Criteria_Evidence
poetry add psutil pynvml
```

## 1. Enhanced Training

```python
from psy_agents_noaug.training.train_loop import Trainer

trainer = Trainer(
    model, train_loader, val_loader, optimizer, criterion, device,
    logging_steps=100,           # Log every 100 steps
    log_system_metrics=True,     # Enable system monitoring
    system_metrics_interval=50,  # Log system every 50 steps
)

with mlflow.start_run():
    trainer.train()
```

**Logs:**
- `train/loss_step`, `train/accuracy_step` (per batch)
- `epoch/train_loss`, `epoch/val_f1_macro` (per epoch)
- `system/gpu_memory_allocated_gb`, `system/cpu_percent`
- `final/best_val_f1_macro`, `final/total_steps`

## 2. HPO with Nested Runs

```python
from psy_agents_noaug.hpo.optuna_runner import OptunaRunner

runner = OptunaRunner("study_name", "maximize", "val_f1_macro")

runner.optimize(
    objective_fn=objective,
    n_trials=50,
    search_space=search_space,
    mlflow_tracking_uri="sqlite:///mlflow.db",
    mlflow_experiment_name="hpo_criteria",  # Creates parent run
)
```

**Creates:**
- Parent run: `HPO_criteria`
- Nested runs: `trial_0001`, `trial_0002`, ...
- Artifacts: `hpo_trials_history.csv`

## 3. Model Registry

```python
from psy_agents_noaug.utils.mlflow_utils import register_model

with mlflow.start_run():
    # Train model...
    
    model_version = register_model(
        model=model,
        model_name="criteria_classifier",
        sample_input=sample_batch["input_ids"][:1],
        stage="Production",
        tags={"f1": "0.85", "task": "criteria"},
        description="Production criteria classifier",
    )
```

## 4. System Metrics Only

```python
from psy_agents_noaug.utils.system_metrics import SystemMetricsLogger

logger = SystemMetricsLogger(log_cpu=True, log_gpu=True)

with mlflow.start_run():
    for step in range(1000):
        # Training code...
        if step % 10 == 0:
            logger.log_metrics(step=step)
```

## MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

**Navigate:**
- Experiments → View runs
- Metrics → Filter by prefix (train/, epoch/, system/)
- Models → Model Registry

## Metric Prefixes

| Prefix | When | Example |
|--------|------|---------|
| `train/*` | Per batch | `train/loss_step` |
| `epoch/*` | Per epoch | `epoch/val_f1_macro` |
| `system/*` | Periodic | `system/gpu_utilization_percent` |
| `trial/*` | Per HPO trial | `trial/val_f1_macro` |
| `final/*` | End of training | `final/best_val_f1_macro` |

## Verification

```bash
# 1. Test training
python -m psy_agents_noaug.cli train task=criteria training.num_epochs=2

# 2. Test HPO
python -m psy_agents_noaug.cli hpo stage=stage0 task=criteria

# 3. View UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Troubleshooting

**Missing pynvml:**
```bash
poetry add pynvml
```

**Missing psutil:**
```bash
poetry add psutil
```

**UI not showing metrics:**
- Check metric prefixes (train/, epoch/, system/)
- Verify mlflow.active_run() is True during logging
- Check step vs epoch indexing

## Documentation

- **Full Guide:** `MLFLOW_ENHANCEMENTS_GUIDE.md`
- **Summary:** `MLFLOW_ENHANCEMENT_SUMMARY.md`
- **Patches:** `mlflow_enhancement_*.patch`
