# DeBERTa-v3 Evidence Sentence Classification

5-fold cross-validation training pipeline for binary evidence classification using DeBERTa-v3-base on the ReDSM5 dataset. This implementation uses NSP-style criterion-sentence pairing to classify whether a sentence provides evidence for a DSM-5 depression criterion.

## Overview

**Task**: Binary sentence-level evidence classification
**Model**: `microsoft/deberta-v3-base` with sequence classification head
**Dataset**: ReDSM5 (Reddit DSM-5 Depression Detection)
**Input Format**: `[CLS] <criterion> [SEP] <sentence> [SEP]`
**Training**: 5-fold cross-validation with GroupKFold by post_id
**Framework**: Hugging Face Transformers + Hydra + MLflow

### Key Features

- Stratified negative sampling (1:3 positive:negative ratio)
- Weighted cross-entropy or focal loss for class imbalance
- Full MLflow experiment tracking with nested CV runs
- Automated metrics aggregation (macro-F1, ROC-AUC, PR-AUC)
- Reproducible with deterministic seeding
- Precision mode auto-detection (BF16 → FP16 → FP32)

## Quickstart

### 1. Installation

Python 3.10+ recommended:

```bash
make setup
```

Or run the individual steps manually:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

Required packages: `torch`, `transformers`, `hydra-core`, `omegaconf`, `mlflow`, `pandas`, `scikit-learn`

### 2. Data Preparation

Ensure data files are present:
- DSM-5 criteria: `data/DSM5/*.json`
- Posts: `data/redsm5/posts.csv`
- Annotations: `data/redsm5/annotations.csv`

See `data/redsm5/README.md` for dataset details.

### 3. Test Data Loading

Verify data pipeline before training:

```bash
python scripts/test_data_loading.py
```

Expected output: Data statistics, fold distribution, sample counts.

### 4. Training (5-Fold Cross-Validation)

Basic training with default config:

```bash
python scripts/train.py
```

With custom hyperparameters:

```bash
python scripts/train.py \
  model.name=microsoft/deberta-v3-base \
  training.args.num_train_epochs=5 \
  training.args.learning_rate=2e-5 \
  training.args.per_device_train_batch_size=16 \
  loss.type=weighted_ce
```

To use focal loss instead:

```bash
python scripts/train.py loss=focal
```

### 5. Inference

Run inference on a criterion-sentence pair:

```bash
python scripts/inference.py \
  --criterion "Depressed mood most of the day" \
  --sentence "I feel sad every single day" \
  --model-path outputs/models/fold_0/best_model
```

Expected output: `label=1, probability=0.87`

## Project Structure

```
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config with defaults
│   ├── model/                 # Model configs (deberta_v3.yaml)
│   ├── trainer/               # Training args (cv.yaml)
│   ├── loss/                  # Loss functions (weighted_ce, focal)
│   ├── cv/                    # Cross-validation settings
│   ├── data/                  # Data pipeline config
│   └── logger/                # MLflow config
├── data/                      # Dataset files
│   ├── DSM5/                  # DSM-5 criteria JSON files
│   └── redsm5/                # ReDSM5 posts and annotations
├── scripts/                   # Executable scripts
│   ├── train.py              # Main training script
│   ├── test_data_loading.py  # Data validation script
│   └── inference.py          # Inference script
├── src/Project/SubProject/    # Source code
│   ├── data/                  # Dataset loading & preprocessing
│   │   └── dataset.py        # ReDSM5Dataset, negative sampling, CV folds
│   ├── engine/                # Training & evaluation
│   │   ├── train_engine.py   # CustomTrainer, CV orchestration
│   │   └── eval_engine.py    # Metrics computation
│   ├── utils/                 # Utilities
│   │   ├── mlflow_utils.py   # MLflow helpers
│   │   ├── seed.py           # Reproducibility
│   │   └── log.py            # Logging config
│   └── models/                # Model wrappers
├── outputs/                   # Training outputs
│   ├── models/               # Model checkpoints per fold
│   └── datasets/             # Processed datasets
├── mlruns/                    # MLflow artifacts
├── specs/                     # Feature specifications
└── tests/                     # Test files (to be implemented)
```

## Configuration (Hydra)

All configuration is managed via Hydra. Override any parameter via CLI:

```bash
python scripts/train.py \
  training.seed=42 \
  training.n_folds=5 \
  data.pos_neg_ratio=3 \
  training.args.learning_rate=1e-5
```

Key config groups:
- `model=deberta_v3` - Model selection
- `loss=weighted_ce` or `loss=focal` - Loss function
- `trainer=cv` - Training arguments
- `data=evidence_pairs` - Data pipeline settings

See `configs/` directory for all available options.

## MLflow Experiment Tracking

### Starting MLflow UI

```bash
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Access at: http://127.0.0.1:5000

### What's Logged

**Per-Fold Runs** (child runs):
- Training/validation metrics (F1, accuracy, ROC-AUC, PR-AUC)
- Model checkpoints (best model per fold)
- Training curves
- Confusion matrices

**Parent Run** (cross-validation):
- Aggregate metrics (mean ± std across folds)
- Best fold selection (highest macro-F1)
- Complete configuration
- Environment info (git SHA, pip freeze, CUDA version)
- Dataset manifest with reproducibility metadata

## Implementation Status

**Completed** (85%):
- ✅ Data loading and preprocessing
- ✅ Stratified negative sampling
- ✅ 5-fold GroupKFold cross-validation
- ✅ DeBERTa-v3 training with custom loss functions
- ✅ MLflow integration with nested runs
- ✅ Metrics computation and aggregation
- ✅ Inference pipeline
- ✅ Complete Hydra configuration system

**Remaining**:
- ⚠️ Comprehensive unit tests (integration test exists)
- ⚠️ Extended documentation and examples

## Reproducibility

All runs are fully reproducible via deterministic seeding:

```python
from Project.SubProject.utils.seed import set_seed
set_seed(1337)
```

Seeds are logged to MLflow and embedded in dataset manifests for auditability.

## Development

Run code quality checks:

```bash
# Linting
ruff check src scripts

# Formatting
black src scripts tests

# Type checking
mypy src
```

## References

- **Dataset**: [ReDSM5](https://huggingface.co/datasets/irlab-udc/redsm5) (Bao et al., CIKM 2025)
- **Model**: [DeBERTa-v3](https://huggingface.co/microsoft/deberta-v3-base) (Microsoft)
- **Specs**: See `specs/001-debertav3-5fold-evidence/` for detailed specification

## License

See LICENSE file for details.
