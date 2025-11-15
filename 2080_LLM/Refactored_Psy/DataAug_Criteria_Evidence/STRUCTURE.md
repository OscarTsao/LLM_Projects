# DataAug_Criteria_Evidence Repository Structure

This document provides a complete overview of the repository structure with augmentation capabilities.

## Package Structure

```
DataAug_Criteria_Evidence/
│
├── pyproject.toml                 # Poetry package configuration
├── README.md                      # Main documentation
├── STRUCTURE.md                   # This file
├── Makefile                       # Development targets
├── .gitignore                     # Git ignore patterns
├── .pre-commit-config.yaml        # Pre-commit hooks
│
├── configs/                       # Hydra configuration files
│   ├── config.yaml               # Main config
│   ├── data/
│   │   ├── hf_redsm5_aug.yaml   # HuggingFace dataset with augmentation
│   │   └── local_csv_aug.yaml   # Local CSV with augmentation
│   ├── model/
│   │   └── mental_bert.yaml      # MentalBERT configuration
│   ├── training/
│   │   └── default_aug.yaml      # Training config (12 epochs for AUG)
│   └── augmentation/
│       ├── nlpaug_default.yaml   # NLPAug configuration
│       ├── textattack_default.yaml # TextAttack configuration
│       ├── hybrid_default.yaml   # Hybrid configuration
│       └── disabled.yaml         # Disable augmentation (baseline)
│
├── src/psy_agents_aug/           # Main package
│   ├── __init__.py
│   │
│   ├── augment/                  # Augmentation pipelines (NEW)
│   │   ├── __init__.py
│   │   ├── base_augmentor.py    # Base interface for all augmentors
│   │   ├── nlpaug_pipeline.py   # NLPAug implementation
│   │   ├── textattack_pipeline.py # TextAttack implementation
│   │   ├── hybrid_pipeline.py   # Hybrid approach
│   │   └── backtranslation.py   # Back-translation (optional)
│   │
│   ├── data/                     # Data loading and processing
│   │   ├── __init__.py
│   │   ├── loaders.py           # Augmentation-aware data loaders
│   │   ├── groundtruth.py       # Ground truth generation
│   │   └── splits.py            # Data splitting utilities
│   │
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   ├── encoders.py          # Text encoders (BERT, etc.)
│   │   ├── criteria_head.py     # Criteria classification head
│   │   └── evidence_head.py     # Evidence extraction head
│   │
│   ├── training/                 # Training loops
│   │   ├── __init__.py
│   │   ├── train_loop.py        # Main training loop
│   │   └── evaluate.py          # Evaluation functions
│   │
│   ├── hpo/                      # Hyperparameter optimization
│   │   ├── __init__.py
│   │   └── optuna_runner.py     # Optuna HPO runner
│   │
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── logging.py           # Logging utilities
│   │   ├── mlflow_utils.py      # MLflow integration
│   │   └── reproducibility.py   # Seed setting, etc.
│   │
│   └── cli.py                    # Command-line interface
│
├── scripts/                      # Executable scripts
│   ├── make_groundtruth.py      # Generate ground truth files
│   ├── test_augmentation.py     # Test augmentation pipelines (NEW)
│   ├── run_hpo_stage.py         # Run hyperparameter optimization
│   ├── train_best.py            # Train with best hyperparameters
│   └── export_metrics.py        # Export MLflow metrics
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_augment_contract.py  # Test augmentation contracts (NEW)
│   ├── test_augment_pipelines.py # Test augmentation pipelines (NEW)
│   └── test_augment_no_leak.py   # Test no val/test leakage (NEW)
│
├── data/                         # Data directory (gitignored)
│   ├── raw/                     # Raw data files
│   ├── processed/               # Processed data
│   ├── groundtruth/             # Ground truth files
│   └── augmented/               # Augmented data cache
│
├── outputs/                      # Training outputs (gitignored)
├── mlruns/                       # MLflow runs (gitignored)
└── artifacts/                    # Model artifacts

```

## Key Files and Their Purposes

### Configuration Files

- **pyproject.toml**: Package configuration with all dependencies including nlpaug and textattack
- **configs/config.yaml**: Main Hydra configuration file
- **configs/augmentation/*.yaml**: Augmentation pipeline configurations

### Augmentation Module (NEW)

- **base_augmentor.py**: Base class for all augmentors with train-only enforcement
- **nlpaug_pipeline.py**: NLPAug-based augmentation (synonym, insert, swap)
- **textattack_pipeline.py**: TextAttack-based augmentation (WordNet, embedding)
- **hybrid_pipeline.py**: Mix multiple augmentation methods
- **backtranslation.py**: Optional back-translation augmentation

### Data Module (Enhanced)

- **loaders.py**: Enhanced with augmentation support, ONLY augments train split
- **groundtruth.py**: Generates ground truth with STRICT validation rules
- **splits.py**: Data splitting utilities

### Scripts (Enhanced)

- **test_augmentation.py**: NEW - Test augmentation determinism and train-only constraint
- **make_groundtruth.py**: Generate ground truth (augmentation-aware)
- **run_hpo_stage.py**: Run HPO with augmentation support
- **train_best.py**: Train with best hyperparameters
- **export_metrics.py**: Export MLflow metrics

### Tests (Enhanced)

- **test_augment_contract.py**: NEW - Test augmentation contracts
- **test_augment_pipelines.py**: NEW - Test each augmentation pipeline
- **test_augment_no_leak.py**: NEW - Test no val/test data leakage

## Critical Guarantees

### 1. Train-Only Augmentation (CRITICAL)

Augmentation ONLY applies to training data, NEVER to validation or test sets.

Enforced at multiple levels:
- `AugmentationConfig.train_only` defaults to `True`
- `BaseAugmentor.augment_batch()` checks split name
- `ReDSM5Loader.load_csv()` only augments when `split == "train"`
- Tests verify this guarantee

### 2. STRICT Data Validation

Same rules as NO-AUG:
- **status field** → ONLY for criteria task
- **cases field** → ONLY for evidence task
- NO cross-contamination allowed

### 3. Deterministic Augmentation

Same seed produces same augmentations for reproducibility.

## Augmentation Pipelines

### NLPAug Pipeline
- Synonym replacement using WordNet
- Random word insertion
- Random word swap

### TextAttack Pipeline
- WordNet-based synonym replacement
- Embedding-based word replacement

### Hybrid Pipeline
- Combines multiple methods with configurable proportions
- Example: 50% NLPAug + 50% TextAttack

### Back-translation (Optional)
- Translates to intermediate language and back
- Requires additional translation models

## Configuration Examples

### Enable NLPAug
```yaml
# configs/augmentation/nlpaug_default.yaml
enabled: true
pipeline: nlpaug_pipeline
ratio: 0.5
max_aug_per_sample: 1
seed: 42
train_only: true  # CRITICAL
```

### Enable TextAttack
```yaml
# configs/augmentation/textattack_default.yaml
enabled: true
pipeline: textattack_pipeline
ratio: 0.5
max_aug_per_sample: 1
seed: 42
train_only: true  # CRITICAL
```

### Disable Augmentation
```yaml
# configs/augmentation/disabled.yaml
enabled: false
```

## Training Configuration

Training with augmentation uses **12 epochs** (vs 10 for NO-AUG):

```yaml
# configs/training/default_aug.yaml
epochs: 12  # Increased from 10
batch_size: 16
learning_rate: 2e-5
log_augmentation_stats: true  # Log to MLflow
```

## Usage Examples

### Train with Augmentation
```bash
# NLPAug
psy-aug train task=criteria augmentation=nlpaug_default

# TextAttack
psy-aug train task=criteria augmentation=textattack_default

# Hybrid
psy-aug train task=criteria augmentation=hybrid_default

# No augmentation (baseline)
psy-aug train task=criteria augmentation=disabled
```

### Test Augmentation
```bash
# Test all pipelines
make verify-aug

# Test specific pipeline
python scripts/test_augmentation.py --pipeline nlpaug
```

### Run Tests
```bash
# All tests
make test

# Augmentation tests only
make test-aug

# Specific test category
make test-contract    # Determinism & train-only
make test-pipelines   # Pipeline functionality
make test-no-leak     # Val/test leakage prevention
```

## Comparison with NO-AUG

| Aspect | NO-AUG | AUG (this repo) |
|--------|--------|-----------------|
| Package name | `psy_agents_noaug` | `psy_agents_aug` |
| Augmentation | None | NLPAug, TextAttack, Hybrid |
| Training epochs | 10 | 12 |
| Dependencies | Basic ML | + nlpaug, textattack |
| Module structure | 6 modules | 7 modules (+ augment/) |
| Test files | Standard | + 3 augmentation tests |
| Scripts | 4 scripts | 5 scripts (+ test_augmentation.py) |

## Dependencies

### Core Dependencies
- Python 3.10+
- torch
- transformers
- hydra-core
- mlflow
- optuna
- pandas, numpy, scikit-learn
- datasets

### Augmentation Dependencies (NEW)
- nlpaug
- textattack

### Development Dependencies
- ruff
- black
- isort
- pytest
- pre-commit

## MLflow Tracking

Augmentation stats automatically logged:
- Augmentation method
- Ratio and max_aug_per_sample
- Original vs augmented sample counts
- Augmentation time

## Next Steps

1. Install dependencies: `poetry install`
2. Test augmentation: `make verify-aug`
3. Generate ground truth: `python scripts/make_groundtruth.py ...`
4. Train with augmentation: `psy-aug train task=criteria`
5. Run HPO: `python scripts/run_hpo_stage.py --task criteria`
