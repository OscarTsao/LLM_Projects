# PSY Agents NO-AUG: Baseline Models without Data Augmentation

This repository contains the **NO-AUG baseline** implementation for Criteria and Evidence extraction from clinical text. This serves as the control/baseline for comparing against augmentation-enhanced models.

## Overview

The NO-AUG repository implements:
- **Criteria Extraction**: Multi-class classification to identify DSM-5 criterion IDs (A, B, C, etc.)
- **Evidence Extraction**: Binary or multi-class classification for evidence presence
- **STRICT Data Validation**: Enforces `status → criteria` and `cases → evidence` mapping rules
- **No Augmentation**: Pure baseline without any data augmentation techniques

## Repository Structure

```
NoAug_Criteria_Evidence/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main composition config
│   ├── data/                  # Data loader configs
│   │   ├── hf_redsm5.yaml
│   │   ├── local_csv.yaml
│   │   └── field_map.yaml
│   ├── model/                 # Model configs
│   │   ├── bert_base.yaml
│   │   ├── roberta_base.yaml
│   │   └── deberta_v3_base.yaml
│   ├── training/
│   │   └── default.yaml
│   ├── hpo/                   # HPO stage configs
│   │   ├── stage0_sanity.yaml
│   │   ├── stage1_coarse.yaml
│   │   ├── stage2_fine.yaml
│   │   └── stage3_refit.yaml
│   └── task/
│       ├── criteria.yaml
│       └── evidence.yaml
├── data/
│   ├── raw/
│   │   └── redsm5/
│   │       └── dsm_criteria.json
│   └── processed/             # Generated splits and ground truth
├── src/psy_agents_noaug/
│   ├── data/
│   │   ├── loaders.py        # Data loading with strict validation
│   │   ├── splits.py         # Train/val/test splitting
│   │   └── groundtruth.py    # Ground truth generation
│   ├── models/
│   │   ├── encoders.py       # Transformer encoders (BERT, RoBERTa, DeBERTa)
│   │   ├── criteria_head.py  # Criteria classification head
│   │   └── evidence_head.py  # Evidence classification head
│   ├── training/
│   │   ├── train_loop.py     # Training loop with MLflow
│   │   └── evaluate.py       # Evaluation utilities
│   ├── hpo/
│   │   └── optuna_runner.py  # Hyperparameter optimization
│   └── utils/
│       ├── logging.py
│       ├── reproducibility.py
│       └── mlflow_utils.py
├── scripts/
│   ├── make_groundtruth.py   # Generate ground truth files
│   ├── run_hpo_stage.py      # Run HPO stages
│   ├── train_best.py         # Train with best hyperparameters
│   └── export_metrics.py     # Export MLflow metrics
├── tests/
│   ├── test_groundtruth.py   # Test strict validation rules
│   ├── test_loaders.py
│   ├── test_training_smoke.py
│   └── test_hpo_config.py
├── pyproject.toml            # Poetry dependencies
├── Makefile                  # Common tasks
└── README.md
```

## Installation

### Prerequisites
- Python 3.10+
- Poetry (for dependency management)

### Setup

```bash
# Install dependencies
make install-dev

# Install pre-commit hooks
make pre-commit-install

# Run tests
make test
```

## STRICT Data Validation Rules

This repository enforces **STRICT** field mapping rules to prevent data leakage:

1. **Criteria Task**: ONLY uses `status` field
2. **Evidence Task**: ONLY uses `cases` field
3. **NO Cross-Contamination**: Ground truth files are validated to ensure no overlap

### Generating Ground Truth

```bash
# Generate criteria ground truth (ONLY status field)
make groundtruth-criteria INPUT=./data/raw/redsm5/train.csv

# Generate evidence ground truth (ONLY cases field)
make groundtruth-evidence INPUT=./data/raw/redsm5/train.csv
```

## Usage

### 1. Hyperparameter Optimization (HPO)

We use a multi-stage HPO approach:

**Stage 0: Sanity Check** (3 trials)
```bash
make hpo-sanity TASK=criteria MODEL=bert_base
```

**Stage 1: Coarse Search** (20 trials, broad ranges)
```bash
make hpo-coarse TASK=criteria MODEL=bert_base
```

**Stage 2: Fine Search** (30 trials, narrow ranges around best)
```bash
make hpo-fine TASK=criteria MODEL=bert_base
```

**Stage 3: Refit with Best** (1 trial, longer training)
```bash
make hpo-refit TASK=criteria MODEL=bert_base
```

### 2. Train Best Model

After HPO, train with best hyperparameters:

```bash
make train-best STUDY=./outputs/hpo/study.pkl TASK=criteria
```

### 3. Export Metrics

Export MLflow metrics to CSV:

```bash
make export-metrics EXPERIMENT=noaug_baseline OUTPUT=./results.csv
```

## Configuration

All configuration is managed via Hydra. See `configs/` directory for:

- **Data configs**: `configs/data/`
- **Model configs**: `configs/model/`
- **Training configs**: `configs/training/`
- **HPO configs**: `configs/hpo/`
- **Task configs**: `configs/task/`

### Example: Changing Model

```yaml
# configs/model/custom.yaml
_target_: psy_agents_noaug.models.encoders.RoBERTaEncoder

variant: "large"
freeze_encoder: false
pooling_strategy: "mean"
max_length: 512
```

## Development

### Code Quality

```bash
# Format code
make format

# Check formatting
make format-check

# Run linters
make lint

# Run tests
make test
```

### Pre-commit Hooks

Pre-commit hooks are configured for:
- `ruff` (linting)
- `black` (formatting)
- `isort` (import sorting)
- Trailing whitespace removal
- YAML/JSON validation

## MLflow Tracking

All experiments are tracked with MLflow:

```bash
# View MLflow UI
mlflow ui --backend-store-uri ./mlruns
```

Metrics logged:
- Training loss and accuracy
- Validation loss and accuracy
- Per-class precision, recall, F1
- Confusion matrix
- Hyperparameters

## Testing

```bash
# Run all tests
make test

# Run specific test file
poetry run pytest tests/test_groundtruth.py -v

# Run with coverage
poetry run pytest --cov=src/psy_agents_noaug tests/
```

### Key Tests

- **test_groundtruth.py**: Validates STRICT field separation rules
- **test_loaders.py**: Tests data loading and validation
- **test_training_smoke.py**: Smoke tests for training pipeline
- **test_hpo_config.py**: Validates HPO configuration files

## Important Notes

### NO Augmentation

This repository is the **NO-AUG baseline**. It contains:
- ✅ Base transformer models (BERT, RoBERTa, DeBERTa)
- ✅ Standard training and evaluation
- ✅ Hyperparameter optimization
- ❌ NO data augmentation techniques
- ❌ NO augmentation code or dependencies (nlpaug is listed but NOT used)

### Field Mapping Rules

The codebase enforces STRICT rules:
- `status` field → Criteria task ONLY
- `cases` field → Evidence task ONLY
- Ground truth files are validated to prevent contamination
- Tests verify strict separation

## Reproducibility

All experiments use fixed random seeds:
- Default seed: 42
- Seeds set for: Python, NumPy, PyTorch, CUDA
- Deterministic CUDA operations enabled

## Citation

If you use this code, please cite:

```bibtex
@software{psy_agents_noaug,
  title = {PSY Agents NO-AUG: Baseline Models for Clinical Text Classification},
  author = {Research Team},
  year = {2024},
  version = {0.1.0}
}
```

## License

[Add your license here]

## Contact

[Add contact information]
