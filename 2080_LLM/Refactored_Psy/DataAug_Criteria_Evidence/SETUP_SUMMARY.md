# AUG Repository Setup Summary

## What Was Created

This repository provides a **complete data augmentation pipeline** for clinical text extraction, building upon the NO-AUG baseline with the following enhancements:

### 1. Poetry Configuration (pyproject.toml)
- Package name: `psy_agents_aug`
- Version: 0.1.0
- Python 3.10+
- Added dependencies: nlpaug, textattack
- All dev dependencies: ruff, black, isort, pytest, pre-commit

### 2. Package Structure (src/psy_agents_aug/)

#### NEW: Augmentation Module (augment/)
- `base_augmentor.py`: Unified interface with train-only enforcement
- `nlpaug_pipeline.py`: Synonym, insertion, swap
- `textattack_pipeline.py`: WordNet, embedding
- `hybrid_pipeline.py`: Mix proportion control
- `backtranslation.py`: Optional en↔de/fr translation

#### Enhanced: Data Module (data/)
- `loaders.py`: Augmentation-aware, ONLY augments train split
- `groundtruth.py`: Same STRICT validation rules
- `splits.py`: Data splitting utilities

#### Copied from NO-AUG:
- models/ (encoders.py, criteria_head.py, evidence_head.py)
- training/ (train_loop.py, evaluate.py)
- hpo/ (optuna_runner.py)
- utils/ (logging.py, reproducibility.py, mlflow_utils.py)
- cli.py

### 3. Hydra Configurations (configs/)

#### Data Configs:
- `hf_redsm5_aug.yaml`: HuggingFace dataset with augmentation
- `local_csv_aug.yaml`: Local CSV with augmentation

#### Augmentation Configs (NEW):
- `nlpaug_default.yaml`: ratio=0.5, max_aug_per_sample=1, synonym method
- `textattack_default.yaml`: ratio=0.5, WordNet method
- `hybrid_default.yaml`: ratio=0.5, 50% NLPAug + 50% TextAttack
- `disabled.yaml`: Augmentation disabled (baseline)

#### Training Config:
- `default_aug.yaml`: 12 epochs (vs 10 for NO-AUG)

#### Model Config:
- `mental_bert.yaml`: MentalBERT configuration

### 4. Scripts (scripts/)

#### NEW:
- `test_augmentation.py`: Verify determinism and train-only constraint

#### Enhanced:
- `make_groundtruth.py`: Augmentation-aware ground truth generation
- `run_hpo_stage.py`: HPO with augmentation support
- `train_best.py`: Train with best hyperparameters
- `export_metrics.py`: Export MLflow metrics

### 5. Tests (tests/)

#### NEW Augmentation Tests:
- `test_augment_contract.py`: Guarantee deterministic counts
- `test_augment_pipelines.py`: Test each pipeline
- `test_augment_no_leak.py`: Verify no augmentation in val/test

### 6. Development Files

- `.pre-commit-config.yaml`: ruff, black, isort, trailing-whitespace
- `.gitignore`: Python ignores + mlruns/, outputs/, data/processed/
- `Makefile`: Augmentation-specific targets

### 7. Documentation

- `README.md`: Comprehensive usage guide
- `STRUCTURE.md`: Complete repository structure
- `SETUP_SUMMARY.md`: This file

## Critical Features

### 1. STRICT Train-Only Augmentation
**CRITICAL**: Augmentation ONLY applies to training data, NEVER to val/test

Enforced at:
- Config level: `train_only=True` (forced)
- Augmentor level: `augment_batch()` checks split name
- Loader level: `load_csv()` only augments train split
- Test level: Verified by `test_augment_no_leak.py`

### 2. Deterministic Augmentation
Same seed produces same augmentations for reproducibility.

### 3. STRICT Data Validation
Same rules as NO-AUG:
- status → criteria task
- cases → evidence task
- NO cross-contamination

### 4. Increased Training Epochs
12 epochs for augmented data (vs 10 for NO-AUG) to account for increased data volume.

## File Counts

```
Total Files Created/Modified:
- Python files: 17 (6 new augmentation files + 11 copied/adapted)
- YAML configs: 8 (4 new augmentation configs + 4 data/training configs)
- Scripts: 5 (1 new + 4 adapted)
- Tests: 3 (all new augmentation tests)
- Dev files: 3 (.pre-commit-config.yaml, .gitignore, Makefile)
- Documentation: 3 (README.md, STRUCTURE.md, SETUP_SUMMARY.md)
```

## Key Differences from NO-AUG

| Feature | NO-AUG | AUG (this repo) |
|---------|--------|-----------------|
| Package name | psy_agents_noaug | psy_agents_aug |
| Augmentation | None | NLPAug, TextAttack, Hybrid |
| Training epochs | 10 | 12 |
| augment/ module | No | Yes (5 files) |
| Augmentation tests | No | Yes (3 files) |
| Dependencies | Basic ML | + nlpaug, textattack |
| Config files | Standard | + augmentation/ |

## Usage Quick Start

```bash
# 1. Install dependencies
poetry install

# 2. Verify augmentation setup
make verify-aug

# 3. Generate ground truth
python scripts/make_groundtruth.py \
    --raw-data data/raw/redsm5.csv \
    --dsm-criteria data/dsm_criteria.json \
    --output-dir data/groundtruth \
    --task both

# 4. Train with augmentation
psy-aug train task=criteria augmentation=nlpaug_default

# 5. Run HPO
python scripts/run_hpo_stage.py \
    --task criteria \
    --n-trials 50 \
    --augmentation nlpaug_default

# 6. Train with best params
python scripts/train_best.py \
    --task criteria \
    --study-name criteria_hpo \
    --augmentation nlpaug_default

# 7. Export metrics
python scripts/export_metrics.py \
    --experiment-name aug_criteria_evidence \
    --output metrics.csv
```

## Testing

```bash
# Test all
make test

# Test augmentation only
make test-aug

# Test specific aspects
make test-contract     # Determinism & train-only
make test-pipelines    # Pipeline functionality
make test-no-leak      # Val/test leakage prevention
```

## Verification Checklist

- [x] pyproject.toml with correct package name and dependencies
- [x] Complete augment/ module with 5 pipeline implementations
- [x] Augmentation-aware data loaders
- [x] Hydra configs with augmentation settings
- [x] Scripts with augmentation support
- [x] Tests for augmentation contracts
- [x] Development files (.pre-commit, .gitignore, Makefile)
- [x] Comprehensive documentation (README, STRUCTURE, SETUP_SUMMARY)

## Next Steps

1. **Test Installation**: Run `poetry install` to verify dependencies
2. **Test Augmentation**: Run `make verify-aug` to test all pipelines
3. **Run Tests**: Run `make test-aug` to verify augmentation contracts
4. **Generate Data**: Create ground truth files with strict validation
5. **Train Models**: Train with different augmentation strategies
6. **Compare Results**: Compare AUG vs NO-AUG performance

## Support

For issues or questions:
1. Check README.md for usage examples
2. Check STRUCTURE.md for file locations
3. Run tests to verify setup: `make test-aug`
4. Review augmentation configs in configs/augmentation/
