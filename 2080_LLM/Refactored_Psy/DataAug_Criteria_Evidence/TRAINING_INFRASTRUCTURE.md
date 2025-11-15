# Training Infrastructure Documentation (Data Augmentation)

## Overview

This repository is equipped with a production-ready training infrastructure for **augmented data** integrating:
- **Hydra**: Hierarchical configuration management
- **MLflow**: Experiment tracking and artifact management  
- **Optuna**: Hyperparameter optimization with multi-stage search
- **Data Augmentation**: NLPAug, TextAttack, and hybrid pipelines

## Key Differences from NO-AUG

1. **Extended Training Epochs**: 
   - Stage 1: 8 epochs (vs 6)
   - Stage 2: 12 epochs (vs 10)
   - Stage 3: 15 epochs (vs 12)
   - Default: 12 epochs (vs 10)

2. **Augmentation Integration**: Data augmentation applied during training

3. **Configuration**: `configs/training/default_aug.yaml` with augmentation-specific settings

## Directory Structure

```
DataAug_Criteria_Evidence/
├── configs/
│   ├── config.yaml            # Main composition (includes augmentation)
│   ├── data/                  # Data source configs (with augmentation)
│   ├── model/                 # Model architecture configs
│   ├── training/              # Training hyperparameters (default_aug.yaml)
│   ├── task/                  # Task-specific settings
│   ├── augmentation/          # Augmentation pipeline configs
│   └── hpo/                   # HPO stage configurations
├── src/Project/Share/
│   ├── utils/
│   │   ├── mlflow_utils.py   # MLflow integration
│   │   └── reproducibility.py # Seed and device utils
│   ├── hpo/
│   │   └── optuna_runner.py  # HPO orchestration
│   └── training/
│       ├── train_loop.py     # Training loop with AMP
│       └── evaluate.py       # Comprehensive evaluation
├── scripts/
│   ├── run_hpo_stage.py      # Run HPO stage
│   └── train_best.py         # Train with best config
├── mlruns/                    # MLflow tracking data
└── outputs/                   # Training outputs
```

## Quick Start

### 1. Basic Training with Augmentation

```bash
# Train with default augmentation
python scripts/train_best.py task=criteria augmentation=nlpaug_default

# Train with TextAttack augmentation
python scripts/train_best.py task=evidence augmentation=textattack_default

# Train with hybrid augmentation
python scripts/train_best.py task=criteria augmentation=hybrid_default

# Disable augmentation
python scripts/train_best.py task=criteria augmentation=disabled
```

### 2. Hyperparameter Optimization

Run multi-stage HPO with augmented data:

```bash
# Stage 0: Sanity check (8 trials, 3 epochs)
python scripts/run_hpo_stage.py hpo=stage0_sanity task=criteria augmentation=nlpaug_default

# Stage 1: Coarse search (40 trials, 8 epochs - increased for augmentation)
python scripts/run_hpo_stage.py hpo=stage1_coarse task=criteria augmentation=nlpaug_default

# Stage 2: Fine search (24 trials, 12 epochs - increased for augmentation)
python scripts/run_hpo_stage.py hpo=stage2_fine task=criteria augmentation=nlpaug_default

# Stage 3: Refit (1 trial, 15 epochs - increased for augmentation)
python scripts/run_hpo_stage.py hpo=stage3_refit task=criteria augmentation=nlpaug_default
```

## HPO Search Space (Augmentation-Aware)

### Stage 0 (Sanity Check)
- 8 trials, 3 epochs
- Wide bounds to verify augmentation pipeline

### Stage 1 (Coarse Search - AUGMENTED)
- 40 trials, **8 epochs** (increased from NO-AUG's 6)
- Same unified search space as NO-AUG
- Accounts for larger augmented dataset

### Stage 2 (Fine Search - AUGMENTED)
- 24 trials, **12 epochs** (increased from NO-AUG's 10)
- Narrowed ranges around Stage 1 best
- More epochs to handle augmented data

### Stage 3 (Refit - AUGMENTED)
- 1 trial, **15 epochs** (increased from NO-AUG's 12, +20%)
- Multiple seeds: [42, 123, 456, 789, 2023]
- Full evaluation on test set

## Augmentation Pipelines

### NLPAug Pipeline
```yaml
augmentation:
  pipeline: "nlpaug"
  ratio: 0.3  # 30% augmentation ratio
  methods:
    - synonym_replacement
    - random_insertion
```

### TextAttack Pipeline
```yaml
augmentation:
  pipeline: "textattack"
  ratio: 0.3
  methods:
    - word_swap
    - back_translation
```

### Hybrid Pipeline
```yaml
augmentation:
  pipeline: "hybrid"
  ratio: 0.5  # 50% augmentation ratio
  methods:
    - nlpaug_synonym
    - textattack_swap
```

## Configuration Override

```bash
# Change augmentation ratio
python scripts/train_best.py augmentation.ratio=0.5

# Switch augmentation pipeline
python scripts/train_best.py augmentation=textattack_default

# Adjust training epochs for augmentation
python scripts/train_best.py training.num_epochs=15
```

## MLflow Tracking

### Experiment Naming (Augmentation-Aware)

- Training: `aug_criteria_evidence-train`
- HPO Stage 0: `aug_criteria_evidence-hpo-stage0`
- HPO Stage 1: `aug_criteria_evidence-hpo-stage1`
- HPO Stage 2: `aug_criteria_evidence-hpo-stage2`
- HPO Stage 3: `aug_criteria_evidence-hpo-stage3`

### Augmentation-Specific Tags

- `augmentation`: Pipeline name (nlpaug, textattack, hybrid)
- `aug_ratio`: Augmentation ratio
- `aug_methods`: Augmentation methods used

## Training Features (Same as NO-AUG + Augmentation)

### Default Training Config (Augmentation)
```yaml
training:
  num_epochs: 12  # Increased from 10
  batch_size: 16
  max_length: 512
  amp:
    enabled: true
  early_stopping:
    patience: 3
    metric: "val_f1_macro"
```

## Evaluation Metrics (Same as NO-AUG)

### Criteria Task
- F1 (macro), Accuracy, AUROC (macro)
- Per-criterion F1
- Confusion matrix

### Evidence Task
- F1 (micro/macro) at sentence level
- Exact/partial match
- Confusion matrix

## Best Practices for Augmented Training

1. **Use appropriate augmentation ratio**: Start with 0.3, adjust based on results
2. **Extended epochs**: Augmented data needs more epochs (12-15 vs 10)
3. **Monitor for overfitting**: Augmentation can introduce noise
4. **Compare with NO-AUG**: Run parallel experiments to validate augmentation benefit
5. **Augmentation-aware HPO**: Run full HPO pipeline with augmentation enabled
6. **Test different pipelines**: Try NLPAug, TextAttack, and Hybrid
7. **Validate on clean test set**: Ensure augmentation doesn't hurt real performance

## Comparison with NO-AUG

| Aspect | NO-AUG | DATA-AUG |
|--------|---------|----------|
| Default epochs | 10 | 12 |
| Stage 1 epochs | 6 | 8 |
| Stage 2 epochs | 10 | 12 |
| Stage 3 epochs | 12 | 15 |
| Data size | Original | Original + Augmented |
| Training time | Baseline | +20-30% |
| Convergence | Faster | Slower (more data) |

## Troubleshooting (Augmentation-Specific)

### Poor Performance with Augmentation
- Reduce `augmentation.ratio` (try 0.2 or 0.1)
- Try different augmentation pipeline
- Increase training epochs
- Check augmentation quality (inspect samples)

### Augmentation Too Slow
- Reduce `augmentation.ratio`
- Use simpler augmentation methods
- Consider pre-augmenting and saving to disk

### Overfitting Despite Augmentation
- Increase dropout rate
- Add more regularization (weight_decay)
- Reduce model capacity

## Contact

For issues or questions, please refer to the main project documentation.
