# Absolute File Paths Reference

This document lists all important file paths in absolute form for easy reference.

## Root Directory
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/
```

## Configuration Files

### Main Config
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/pyproject.toml
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/configs/config.yaml
```

### Data Configs
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/configs/data/hf_redsm5_aug.yaml
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/configs/data/local_csv_aug.yaml
```

### Augmentation Configs
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/configs/augmentation/nlpaug_default.yaml
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/configs/augmentation/textattack_default.yaml
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/configs/augmentation/hybrid_default.yaml
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/configs/augmentation/disabled.yaml
```

### Model & Training Configs
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/configs/model/mental_bert.yaml
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/configs/training/default_aug.yaml
```

## Source Code

### Main Package
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/__init__.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/cli.py
```

### Augmentation Module (NEW)
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/augment/__init__.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/augment/base_augmentor.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/augment/nlpaug_pipeline.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/augment/textattack_pipeline.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/augment/hybrid_pipeline.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/augment/backtranslation.py
```

### Data Module
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/data/__init__.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/data/loaders.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/data/groundtruth.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/data/splits.py
```

### Models Module
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/models/__init__.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/models/encoders.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/models/criteria_head.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/models/evidence_head.py
```

### Training Module
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/training/__init__.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/training/train_loop.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/training/evaluate.py
```

### HPO Module
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/hpo/__init__.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/hpo/optuna_runner.py
```

### Utils Module
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/utils/__init__.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/utils/logging.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/utils/mlflow_utils.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/src/psy_agents_aug/utils/reproducibility.py
```

## Scripts
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/scripts/make_groundtruth.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/scripts/test_augmentation.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/scripts/run_hpo_stage.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/scripts/train_best.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/scripts/export_metrics.py
```

## Tests
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/tests/__init__.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/tests/test_augment_contract.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/tests/test_augment_pipelines.py
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/tests/test_augment_no_leak.py
```

## Development Files
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/.gitignore
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/.pre-commit-config.yaml
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/Makefile
```

## Documentation
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/README.md
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/STRUCTURE.md
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/SETUP_SUMMARY.md
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/ABSOLUTE_PATHS.md
```

## Data Directories (to be created)
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/data/raw/
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/data/processed/
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/data/groundtruth/
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/data/augmented/
```

## Output Directories (to be created)
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/outputs/
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/mlruns/
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/artifacts/
```

## Usage Examples with Absolute Paths

### Install Package
```bash
cd /experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence
poetry install
```

### Test Augmentation
```bash
python /experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/scripts/test_augmentation.py --pipeline nlpaug
```

### Generate Ground Truth
```bash
python /experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/scripts/make_groundtruth.py \
    --raw-data /experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/data/raw/train.csv \
    --dsm-criteria /experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/data/dsm_criteria.json \
    --output-dir /experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/data/groundtruth
```

### Run Tests
```bash
cd /experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence
pytest tests/test_augment_contract.py -v
```

### Run HPO
```bash
python /experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/scripts/run_hpo_stage.py \
    --task criteria \
    --n-trials 50 \
    --augmentation nlpaug_default \
    --config-dir /experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/configs
```
