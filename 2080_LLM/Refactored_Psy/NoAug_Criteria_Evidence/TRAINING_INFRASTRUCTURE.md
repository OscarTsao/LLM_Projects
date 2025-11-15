# Training Infrastructure Documentation

## Overview

This repository is equipped with a production-ready training infrastructure integrating:
- **Hydra**: Hierarchical configuration management
- **MLflow**: Experiment tracking and artifact management  
- **Optuna**: Hyperparameter optimization with multi-stage search

## Directory Structure

```
NoAug_Criteria_Evidence/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main composition
│   ├── data/                  # Data source configs
│   ├── model/                 # Model architecture configs
│   ├── training/              # Training hyperparameters
│   ├── task/                  # Task-specific settings
│   └── hpo/                   # HPO stage configurations
├── src/psy_agents_noaug/
│   ├── utils/
│   │   ├── mlflow_utils.py   # MLflow integration
│   │   └── reproducibility.py # Seed and device utils
│   ├── hpo/
│   │   └── optuna_runner.py  # HPO orchestration
│   ├── training/
│   │   ├── train_loop.py     # Training loop with AMP
│   │   └── evaluate.py       # Comprehensive evaluation
│   └── cli.py                 # Command-line interface
├── scripts/
│   ├── run_hpo_stage.py      # Run HPO stage
│   └── train_best.py         # Train with best config
├── mlruns/                    # MLflow tracking data
└── outputs/                   # Training outputs
```

## Quick Start

### 1. Basic Training

```bash
# Train with default config
python -m psy_agents_noaug.cli train task=criteria model=bert_base

# Train with custom settings
python -m psy_agents_noaug.cli train task=evidence model=roberta_base training.num_epochs=15
```

### 2. Hyperparameter Optimization

Run multi-stage HPO:

```bash
# Stage 0: Sanity check (8 trials, 3 epochs)
python scripts/run_hpo_stage.py hpo=stage0_sanity task=criteria

# Stage 1: Coarse search (40 trials, 6 epochs)
python scripts/run_hpo_stage.py hpo=stage1_coarse task=criteria

# Stage 2: Fine search (24 trials, 10 epochs)
python scripts/run_hpo_stage.py hpo=stage2_fine task=criteria

# Stage 3: Refit with best params (1 trial, 12 epochs, multiple seeds)
python scripts/run_hpo_stage.py hpo=stage3_refit task=criteria
```

### 3. Train with Best Configuration

```bash
python scripts/train_best.py \
    task=criteria \
    best_config=outputs/hpo_stage2/best_config.yaml
```

## HPO Search Space

### Stage 0 (Sanity Check)
- 8 trials, 3 epochs
- Wide bounds to verify pipeline

### Stage 1 (Coarse Search)
- 40 trials, 6 epochs
- Unified search space:
  - `learning_rate`: loguniform(1e-6, 5e-5)
  - `weight_decay`: loguniform(1e-5, 1e-1)
  - `warmup_ratio`: uniform(0.0, 0.2)
  - `dropout`: uniform(0.0, 0.3)
  - `batch_size`: categorical [8, 16, 32]
  - `max_length`: categorical [256, 384, 512]
  - `scheduler`: categorical ["linear", "cosine", "cosine_with_restarts"]
  - `encoder`: categorical ["bert-base-uncased", "roberta-base", "microsoft/deberta-v3-base"]
  - `lora.enabled`: categorical [false, true]
  - `lora.r`: categorical [8, 16]
  - `lora.alpha`: categorical [16, 32]

### Stage 2 (Fine Search)
- 24 trials, 10 epochs
- Narrowed ranges around Stage 1 best

### Stage 3 (Refit)
- 1 trial, 12 epochs (+20%)
- Multiple seeds: [42, 123, 456, 789, 2023]
- Evaluate on test set

## Configuration Override

Hydra supports flexible overrides:

```bash
# Override single parameter
python scripts/train_best.py task=criteria training.batch_size=32

# Override nested parameter
python scripts/train_best.py model.head.dropout=0.2

# Override from file
python scripts/train_best.py --config-name=my_custom_config
```

## MLflow Tracking

### View Experiments

```bash
# Launch MLflow UI
mlflow ui --backend-store-uri ./mlruns

# Access at http://localhost:5000
```

### Experiment Naming

- Training: `noaug_baseline-train`
- HPO Stage 0: `noaug_baseline-hpo-stage0`
- HPO Stage 1: `noaug_baseline-hpo-stage1`
- HPO Stage 2: `noaug_baseline-hpo-stage2`
- HPO Stage 3: `noaug_baseline-hpo-stage3`

### Logged Artifacts

- Configuration YAML
- Model checkpoints (best and last)
- Evaluation reports (JSON)
- Confusion matrices
- Training history
- HPO study results

## Training Features

### Mixed Precision Training (AMP)
```yaml
training:
  amp:
    enabled: true
    dtype: "float16"  # or "bfloat16"
```

### Gradient Accumulation
```yaml
training:
  gradient_accumulation_steps: 2  # Effective batch_size *= 2
```

### Early Stopping
```yaml
training:
  early_stopping:
    enabled: true
    patience: 3
    metric: "val_f1_macro"  # or "val_loss"
    mode: "max"  # "max" for metrics, "min" for loss
    min_delta: 0.0001
```

### LoRA Fine-tuning
```yaml
training:
  lora:
    enabled: true
    r: 8
    alpha: 16
    dropout: 0.1
    target_modules: ["query", "value"]
```

## Evaluation Metrics

### Criteria Task
- F1 (macro)
- Accuracy
- AUROC (macro)
- Per-criterion F1
- Confusion matrix

### Evidence Task
- F1 (micro/macro) at sentence level
- Exact match
- Partial match
- Confusion matrix

## Reproducibility

All experiments use fixed seeds and deterministic operations:

```python
from psy_agents_noaug.utils.reproducibility import set_seed

set_seed(42, deterministic=True, cudnn_benchmark=False)
```

## Directory Outputs

### After Training
```
outputs/
├── checkpoints/
│   ├── best_checkpoint.pt
│   └── latest_checkpoint.pt
├── evaluation_report.json
└── config.yaml
```

### After HPO
```
outputs/
├── hpo_stage0/
│   ├── best_config.yaml
│   ├── trials_history.json
│   └── study.pkl
├── hpo_stage1/
├── hpo_stage2/
└── hpo_stage3/
```

## Best Practices

1. **Always run Stage 0** before full HPO to verify pipeline
2. **Use Stage 1** to explore broad hyperparameter ranges
3. **Use Stage 2** to refine around best Stage 1 results
4. **Use Stage 3** to retrain with multiple seeds for robustness
5. **Monitor MLflow** for experiment comparisons
6. **Set deterministic=true** for reproducibility
7. **Use AMP** to speed up training on modern GPUs

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `max_length`
- Enable gradient checkpointing

### Slow Training
- Enable AMP: `training.amp.enabled=true`
- Increase `batch_size` (if GPU allows)
- Use `num_workers > 0` for data loading

### Poor Convergence
- Increase `num_epochs`
- Adjust `learning_rate` (try loguniform search)
- Try different `scheduler` types
- Check for class imbalance (use `class_weights`)

## Contact

For issues or questions, please refer to the main project documentation.
