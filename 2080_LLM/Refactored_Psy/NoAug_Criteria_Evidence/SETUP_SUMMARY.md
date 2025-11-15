# NO-AUG Repository Setup Summary

This document summarizes the complete setup of the NO-AUG baseline repository for Criteria and Evidence extraction.

## Created Files and Directories

### Package Structure (src/psy_agents_noaug/)

```
src/psy_agents_noaug/
├── __init__.py                  # Package initialization (v0.1.0)
├── cli.py                       # Command-line interface
├── data/
│   ├── __init__.py
│   ├── loaders.py              # ReDSM5Loader, DSMCriteriaLoader with STRICT validation
│   ├── splits.py               # DataSplitter with reproducibility
│   └── groundtruth.py          # GroundTruthValidator & GroundTruthGenerator
├── models/
│   ├── __init__.py
│   ├── encoders.py             # TransformerEncoder, BERTEncoder, RoBERTaEncoder, DeBERTaEncoder
│   ├── criteria_head.py        # CriteriaClassificationHead, CriteriaModel
│   └── evidence_head.py        # EvidenceClassificationHead, EvidenceModel
├── training/
│   ├── __init__.py
│   ├── train_loop.py           # Trainer with MLflow integration & early stopping
│   └── evaluate.py             # Evaluator with comprehensive metrics
├── hpo/
│   ├── __init__.py
│   └── optuna_runner.py        # OptunaHPO for hyperparameter optimization
└── utils/
    ├── __init__.py
    ├── reproducibility.py      # set_seed, get_device, count_parameters
    ├── logging.py              # setup_logger
    └── mlflow_utils.py         # MLflow integration utilities
```

### Configuration Files (configs/)

```
configs/
├── config.yaml                 # Main composition config
├── data/
│   ├── hf_redsm5.yaml         # HuggingFace dataset config
│   ├── local_csv.yaml         # Local CSV dataset config
│   └── field_map.yaml         # STRICT field mapping rules
├── model/
│   ├── bert_base.yaml         # BERT-base configuration
│   ├── roberta_base.yaml      # RoBERTa-base configuration
│   └── deberta_v3_base.yaml   # DeBERTa-v3-base configuration
├── training/
│   └── default.yaml           # Default training hyperparameters
├── hpo/
│   ├── stage0_sanity.yaml     # Stage 0: Sanity check (3 trials)
│   ├── stage1_coarse.yaml     # Stage 1: Coarse search (20 trials)
│   ├── stage2_fine.yaml       # Stage 2: Fine search (30 trials)
│   └── stage3_refit.yaml      # Stage 3: Refit best (1 trial)
└── task/
    ├── criteria.yaml          # Criteria task config (status field)
    └── evidence.yaml          # Evidence task config (cases field)
```

### Scripts (scripts/)

```
scripts/
├── make_groundtruth.py        # Generate ground truth with STRICT validation
├── run_hpo_stage.py           # Run HPO stages 0-3
├── train_best.py              # Train with best hyperparameters
└── export_metrics.py          # Export MLflow metrics to CSV/JSON
```

All scripts are executable (chmod +x).

### Tests (tests/)

```
tests/
├── __init__.py
├── test_groundtruth.py        # Test STRICT validation rules
├── test_loaders.py            # Test data loaders
├── test_training_smoke.py     # Smoke tests for training pipeline
└── test_hpo_config.py         # Validate HPO configurations
```

### Data Structure

```
data/
├── raw/
│   └── redsm5/
│       └── dsm_criteria.json  # Copied from source repository
└── processed/                 # For generated splits and ground truth
```

### Development Files

```
.
├── pyproject.toml             # Poetry dependencies & configuration
├── .gitignore                 # Standard Python ignores + mlruns/outputs/
├── .pre-commit-config.yaml    # Pre-commit hooks (ruff, black, isort)
├── Makefile                   # Common tasks & targets
├── README.md                  # Comprehensive documentation
└── setup.sh                   # Automated setup script
```

## Key Features Implemented

### 1. STRICT Data Validation

The repository enforces strict field mapping rules:

- **Criteria Task**: ONLY uses `status` field
- **Evidence Task**: ONLY uses `cases` field
- **Validation**: `GroundTruthValidator` checks for cross-contamination
- **Tests**: `test_groundtruth.py` verifies strict separation

### 2. Multi-Stage HPO

Four-stage hyperparameter optimization:

- **Stage 0**: Sanity check (3 trials, minimal search space)
- **Stage 1**: Coarse search (20 trials, broad ranges)
- **Stage 2**: Fine search (30 trials, narrow ranges around best)
- **Stage 3**: Refit best (1 trial, longer training)

### 3. Model Support

Implemented encoders for:

- BERT (base/large variants)
- RoBERTa (base/large variants)
- DeBERTa-v3 (base/large variants)

All models support:
- Frozen or fine-tunable encoders
- CLS token or mean pooling
- Configurable classification heads

### 4. Training Infrastructure

- **Trainer**: Training loop with MLflow logging, early stopping, gradient clipping
- **Evaluator**: Comprehensive metrics (accuracy, precision, recall, F1, confusion matrix)
- **Reproducibility**: Fixed seeds for Python, NumPy, PyTorch, CUDA

### 5. MLflow Integration

All experiments tracked with:
- Hyperparameters
- Training/validation metrics
- Model checkpoints
- Configuration files

### 6. Testing

Comprehensive test coverage:
- Unit tests for all core modules
- STRICT validation rule tests
- Smoke tests for training pipeline
- HPO configuration validation

## Dependencies (pyproject.toml)

### Core Dependencies

- torch >= 2.0.0
- transformers >= 4.30.0
- hydra-core >= 1.3.0
- mlflow >= 2.5.0
- optuna >= 3.2.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- datasets >= 2.13.0
- nlpaug >= 1.1.11 (for comparison only - NOT used)

### Development Dependencies

- ruff >= 0.1.0
- black >= 23.7.0
- isort >= 5.12.0
- pytest >= 7.4.0
- pre-commit >= 3.3.0

## Makefile Targets

### Installation
- `make install` - Install dependencies
- `make install-dev` - Install with dev dependencies
- `make pre-commit-install` - Install pre-commit hooks

### Testing
- `make test` - Run all tests
- `make test-groundtruth` - Run ground truth validation tests

### Code Quality
- `make lint` - Run linters
- `make format` - Format code
- `make format-check` - Check formatting
- `make pre-commit-run` - Run all pre-commit hooks

### Data Processing
- `make groundtruth-criteria` - Generate criteria ground truth
- `make groundtruth-evidence` - Generate evidence ground truth

### HPO
- `make hpo-sanity` - Run stage 0 (sanity check)
- `make hpo-coarse` - Run stage 1 (coarse search)
- `make hpo-fine` - Run stage 2 (fine search)
- `make hpo-refit` - Run stage 3 (refit best)

### Training
- `make train-best` - Train with best hyperparameters
- `make export-metrics` - Export MLflow metrics

### Cleanup
- `make clean` - Remove generated files and caches

## Usage Examples

### 1. Initial Setup

```bash
cd /experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence
./setup.sh
```

### 2. Generate Ground Truth

```bash
# For criteria task
make groundtruth-criteria INPUT=./data/raw/redsm5/train.csv

# For evidence task
make groundtruth-evidence INPUT=./data/raw/redsm5/train.csv
```

### 3. Run HPO Pipeline

```bash
# Stage 0: Sanity check
make hpo-sanity TASK=criteria MODEL=bert_base

# Stage 1: Coarse search
make hpo-coarse TASK=criteria MODEL=bert_base

# Stage 2: Fine search
make hpo-fine TASK=criteria MODEL=bert_base

# Stage 3: Refit best
make hpo-refit TASK=criteria MODEL=bert_base
```

### 4. Train Best Model

```bash
make train-best STUDY=./outputs/hpo/study.pkl TASK=criteria
```

### 5. Export Results

```bash
make export-metrics EXPERIMENT=noaug_baseline OUTPUT=./results.csv
```

## File Locations (Absolute Paths)

All files are located under:
```
/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/
```

Key locations:
- Source code: `/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/src/psy_agents_noaug/`
- Configs: `/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/configs/`
- Scripts: `/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/scripts/`
- Tests: `/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/tests/`
- Data: `/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/data/`

## Next Steps

1. **Add Training Data**: Place processed CSV files in `data/processed/`
2. **Generate Ground Truth**: Run `make groundtruth-*` commands
3. **Run HPO**: Execute multi-stage HPO pipeline
4. **Train Models**: Train with best hyperparameters
5. **Evaluate**: Compare against augmentation-enhanced models

## Important Notes

### NO Augmentation

This is the **NO-AUG baseline**. It contains:
- ✅ Base transformer models
- ✅ Standard training and evaluation
- ✅ Hyperparameter optimization
- ❌ NO data augmentation techniques
- ❌ NO augmentation code (nlpaug listed but NOT used)

### STRICT Validation

All ground truth files are validated to ensure:
- Criteria uses ONLY `status` field
- Evidence uses ONLY `cases` field
- NO cross-contamination between tasks

### Reproducibility

All random seeds are fixed:
- Default seed: 42
- Seeds set for: Python, NumPy, PyTorch, CUDA
- Deterministic CUDA operations enabled

## Verification

To verify the setup:

```bash
cd /experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence

# Check structure
ls -la src/psy_agents_noaug/
ls -la configs/
ls -la scripts/
ls -la tests/

# Run tests
make test

# Check DSM criteria file
cat data/raw/redsm5/dsm_criteria.json
```

## Troubleshooting

If you encounter issues:

1. **Missing dependencies**: Run `make install-dev`
2. **Test failures**: Check that DSM criteria file exists
3. **Import errors**: Ensure Poetry environment is activated: `poetry shell`
4. **Permission errors**: Make scripts executable: `chmod +x scripts/*.py`

## Contact & Support

For questions or issues, refer to:
- README.md for detailed documentation
- Individual module docstrings for API documentation
- Test files for usage examples
