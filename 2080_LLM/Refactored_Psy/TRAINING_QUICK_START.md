# Training Infrastructure Quick Start Guide

## Installation Requirements

```bash
pip install hydra-core omegaconf mlflow optuna torch transformers scikit-learn tqdm
```

## Directory Overview

Both repositories now have:
- `/configs/`: Hydra configuration files
- `/src/{package}/utils/`: MLflow and reproducibility utilities
- `/src/{package}/hpo/`: Optuna HPO runner
- `/src/{package}/training/`: Training loop and evaluation
- `/scripts/`: Executable training scripts
- `/mlruns/`: MLflow tracking storage
- `/outputs/`: Training outputs and checkpoints

## Common Usage Patterns

### 1. Quick Training (NoAug)

```bash
cd /experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence

# Train criteria classification
python scripts/train_best.py task=criteria model=bert_base

# Train evidence extraction
python scripts/train_best.py task=evidence model=roberta_base

# Override hyperparameters
python scripts/train_best.py task=criteria model=bert_base \
    training.batch_size=32 \
    training.num_epochs=15 \
    training.optimizer.lr=3e-5
```

### 2. Quick Training (DataAug)

```bash
cd /experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence

# Train with NLPAug
python scripts/train_best.py task=criteria \
    augmentation=nlpaug_default

# Train with TextAttack
python scripts/train_best.py task=evidence \
    augmentation=textattack_default

# Adjust augmentation ratio
python scripts/train_best.py task=criteria \
    augmentation=nlpaug_default \
    augmentation.ratio=0.5
```

### 3. Hyperparameter Optimization

**Stage 0: Sanity Check (8 trials, 3 epochs)**
```bash
python scripts/run_hpo_stage.py hpo=stage0_sanity task=criteria
```

**Stage 1: Coarse Search (40 trials, 6/8 epochs)**
```bash
# NoAug: 6 epochs
python scripts/run_hpo_stage.py hpo=stage1_coarse task=criteria

# DataAug: 8 epochs
python scripts/run_hpo_stage.py hpo=stage1_coarse task=criteria \
    augmentation=nlpaug_default
```

**Stage 2: Fine Search (24 trials, 10/12 epochs)**
```bash
# NoAug: 10 epochs
python scripts/run_hpo_stage.py hpo=stage2_fine task=criteria

# DataAug: 12 epochs
python scripts/run_hpo_stage.py hpo=stage2_fine task=criteria \
    augmentation=nlpaug_default
```

**Stage 3: Refit (1 trial, 12/15 epochs, 5 seeds)**
```bash
# NoAug: 12 epochs
python scripts/run_hpo_stage.py hpo=stage3_refit task=criteria

# DataAug: 15 epochs
python scripts/run_hpo_stage.py hpo=stage3_refit task=criteria \
    augmentation=nlpaug_default
```

### 4. View Results in MLflow

```bash
# Launch MLflow UI
cd /experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence
mlflow ui --backend-store-uri ./mlruns --port 5000

# Or for DataAug
cd /experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence
mlflow ui --backend-store-uri ./mlruns --port 5001

# Open browser to http://localhost:5000 or http://localhost:5001
```

### 5. Train with Best HPO Results

```bash
# After running HPO stages, use best config
python scripts/train_best.py \
    task=criteria \
    best_config=outputs/hpo_stage2/best_config.yaml
```

## Configuration Files Quick Reference

### NoAug Repository

**Main config**: `/configs/config.yaml`
```yaml
defaults:
  - data: hf_redsm5          # or local_csv
  - model: bert_base          # or roberta_base, deberta_v3_base
  - training: default
  - task: criteria            # or evidence
  - hpo: stage0_sanity       # stage0-3
```

**Training config**: `/configs/training/default.yaml`
```yaml
num_epochs: 10
batch_size: 16
max_length: 512
optimizer:
  lr: 2.0e-5
  weight_decay: 0.01
scheduler:
  type: "cosine"
  warmup_ratio: 0.06
amp:
  enabled: true
early_stopping:
  patience: 3
  metric: "val_f1_macro"
```

### DataAug Repository

**Main config**: `/configs/config.yaml`
```yaml
defaults:
  - data: hf_redsm5_aug      # or local_csv_aug
  - model: mental_bert        # or bert_base, roberta_base
  - training: default_aug
  - augmentation: nlpaug_default  # or textattack, hybrid, disabled
  - task: criteria
  - hpo: stage0_sanity
```

**Training config**: `/configs/training/default_aug.yaml`
```yaml
num_epochs: 12              # Increased for augmented data
# All other settings same as NoAug
```

## HPO Search Space

All stages use unified search space:

| Parameter | Type | Range |
|-----------|------|-------|
| learning_rate | loguniform | 1e-6 to 5e-5 |
| weight_decay | loguniform | 1e-5 to 1e-1 |
| warmup_ratio | uniform | 0.0 to 0.2 |
| dropout | uniform | 0.0 to 0.3 |
| batch_size | categorical | [8, 16, 32] |
| max_length | categorical | [256, 384, 512] |
| scheduler | categorical | [linear, cosine, cosine_with_restarts] |
| encoder | categorical | [bert-base, roberta-base, deberta-v3-base] |
| lora_enabled | categorical | [false, true] |
| lora_r | categorical | [8, 16] |
| lora_alpha | categorical | [16, 32] |

## Outputs Structure

After training/HPO:
```
outputs/
├── checkpoints/
│   ├── best_checkpoint.pt      # Best model on validation
│   └── latest_checkpoint.pt    # Latest model
├── evaluation_report.json      # Comprehensive metrics
├── config.yaml                 # Training configuration
└── hpo_stage{0,1,2,3}/
    ├── best_config.yaml        # Best hyperparameters
    ├── trials_history.json     # All trial results
    └── study.pkl              # Optuna study object
```

## Evaluation Metrics

### Criteria Task
- F1 Macro (primary metric)
- Accuracy
- AUROC Macro
- Per-criterion F1 (A, B, D, G, I, J, K)
- Confusion Matrix

### Evidence Task
- F1 Macro (primary metric)
- F1 Micro
- Exact Match
- Partial Match
- Confusion Matrix

## Common Commands Cheat Sheet

```bash
# Basic training
python scripts/train_best.py task={criteria|evidence}

# With custom model
python scripts/train_best.py task=criteria model={bert_base|roberta_base|deberta_v3_base}

# Override batch size
python scripts/train_best.py training.batch_size=32

# Override learning rate
python scripts/train_best.py training.optimizer.lr=3e-5

# Run HPO stage
python scripts/run_hpo_stage.py hpo=stage{0|1|2|3}_{sanity|coarse|fine|refit}

# With augmentation (DataAug only)
python scripts/train_best.py augmentation={nlpaug_default|textattack_default|hybrid_default|disabled}

# View MLflow
mlflow ui --backend-store-uri ./mlruns

# Export metrics to CSV
python -m psy_agents_noaug.cli export_metrics
```

## Troubleshooting

**Out of Memory**:
```bash
python scripts/train_best.py training.batch_size=8 training.gradient_accumulation_steps=4
```

**Slow Training**:
```bash
python scripts/train_best.py training.amp.enabled=true training.num_workers=8
```

**Poor Performance**:
```bash
# Run full HPO pipeline
python scripts/run_hpo_stage.py hpo=stage1_coarse
python scripts/run_hpo_stage.py hpo=stage2_fine
```

## Next Steps

1. ✅ Infrastructure is set up
2. TODO: Integrate with existing data loaders
3. TODO: Connect to existing model architectures
4. TODO: Run end-to-end tests
5. TODO: Execute full HPO pipeline

## Documentation

- `/TRAINING_INFRASTRUCTURE.md`: Comprehensive documentation
- `/TRAINING_INFRASTRUCTURE_SETUP_SUMMARY.md`: Setup summary

## Support

For implementation details, see the source code in:
- `src/psy_agents_noaug/` (NoAug)
- `src/Project/Share/` (DataAug)
