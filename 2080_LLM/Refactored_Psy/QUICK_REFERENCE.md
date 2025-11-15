# Quick Reference Card

## NoAug Repository - Common Commands

### Setup
```bash
cd /experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence
make setup              # Install + pre-commit + sanity check
```

### Data
```bash
make groundtruth        # Generate ground truth (HF)
```

### Training
```bash
make train              # Train criteria/roberta_base
make train-evidence     # Train evidence task
make train TASK=criteria MODEL=bert_base  # Custom
```

### HPO
```bash
make hpo-s0            # Sanity check
make hpo-s1            # Coarse search
make hpo-s2            # Fine search
make refit             # Refit on train+val
make full-hpo          # Run all stages
```

### Evaluation
```bash
make eval              # Evaluate best model
make export            # Export metrics
```

### Development
```bash
make format            # Format code
make lint              # Run linters
make test              # Run tests
make clean             # Clean caches
```

---

## AUG Repository - Common Commands

### Setup
```bash
cd /experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence
make setup             # Install + pre-commit + sanity check
make verify-aug        # Test augmentation setup
```

### Data
```bash
make groundtruth       # Generate ground truth (HF)
```

### Training
```bash
make train             # Train without augmentation
make train-aug         # Train WITH augmentation
make train-aug AUG_PIPELINE=nlpaug_pipeline  # Specific pipeline
```

### HPO
```bash
make hpo-s0            # Sanity check (no aug)
make hpo-s1-aug        # Coarse search WITH aug
make hpo-s2-aug        # Fine search WITH aug
make full-hpo-aug      # Complete pipeline with aug
```

### Augmentation
```bash
make verify-aug        # Verify augmentation works
make test-aug          # Run augmentation tests
make test-contract     # Test contracts
make compare-aug       # Compare with/without aug
```

### Evaluation
```bash
make eval              # Evaluate best model
make export            # Export metrics
```

---

## CLI Direct Usage

### NoAug
```bash
# Generate ground truth
python -m psy_agents_noaug.cli make_groundtruth data=hf_redsm5

# Train
python -m psy_agents_noaug.cli train task=criteria model=roberta_base

# HPO
python -m psy_agents_noaug.cli hpo hpo=stage1_coarse task=criteria

# Evaluate
python -m psy_agents_noaug.cli evaluate_best checkpoint=outputs/best.pt

# Export
python -m psy_agents_noaug.cli export_metrics
```

### AUG
```bash
# Generate ground truth
python -m psy_agents_aug.cli make_groundtruth data=hf_redsm5

# Train with augmentation
python -m psy_agents_aug.cli train task=criteria augment.enabled=true

# HPO with augmentation
python -m psy_agents_aug.cli hpo hpo=stage1_coarse augment.enabled=true

# Test augmentation
python -m psy_agents_aug.cli test_augmentation --pipeline nlpaug --n 5

# Evaluate
python -m psy_agents_aug.cli evaluate_best checkpoint=outputs/best.pt
```

---

## Hydra Overrides

```bash
# Override single parameter
python -m psy_agents_noaug.cli train training.num_epochs=30

# Override nested parameter
python -m psy_agents_noaug.cli train training.optimizer.lr=2e-5

# Override multiple parameters
python -m psy_agents_noaug.cli train \
    training.num_epochs=30 \
    training.batch_size=16 \
    training.optimizer.lr=2e-5

# Multi-run
python -m psy_agents_noaug.cli train -m training.batch_size=16,32
```

---

## Common Workflows

### NoAug: Full Pipeline
```bash
make setup
make groundtruth
make full-hpo
make eval
make export
```

### AUG: With Augmentation
```bash
make setup
make groundtruth
make verify-aug
make full-hpo-aug
make eval
make export
```

### AUG: Comparison Study
```bash
make setup
make groundtruth
make compare-aug TASK=criteria
```

---

## Troubleshooting

### Module not found
```bash
# Use poetry run
poetry run python -m psy_agents_noaug.cli train task=criteria

# Or set PYTHONPATH
PYTHONPATH=src python -m psy_agents_noaug.cli train task=criteria
```

### Config not found
```bash
# Check available configs
ls configs/task/
ls configs/model/

# Use correct config name (no path, no .yaml)
python -m psy_agents_noaug.cli train task=criteria  # ✓
python -m psy_agents_noaug.cli train task=configs/task/criteria.yaml  # ✗
```

### MLflow errors
```bash
# Set tracking URI
export MLFLOW_TRACKING_URI=./mlruns
```

---

## File Locations

### NoAug
```
/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/
├── src/psy_agents_noaug/cli.py
├── Makefile
├── CLI_AND_MAKEFILE_GUIDE.md
└── configs/
```

### AUG
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/
├── src/psy_agents_aug/cli.py
├── Makefile
├── CLI_AND_MAKEFILE_GUIDE.md
└── configs/
```

---

## Help Commands

```bash
# CLI help
python -m psy_agents_noaug.cli --help
python -m psy_agents_aug.cli --help

# Makefile help
make help

# Project info
make info
```

---

## Key Differences: NoAug vs AUG

| Feature | NoAug | AUG |
|---------|-------|-----|
| CLI Commands | 6 | 7 (+ test_augmentation) |
| Makefile Targets | 29 | 40 (+ 11 augmentation) |
| Augmentation | ✗ | ✓ |
| test_augmentation | ✗ | ✓ |
| train-aug | ✗ | ✓ |
| hpo-s*-aug | ✗ | ✓ |
| verify-aug | ✗ | ✓ |
| compare-aug | ✗ | ✓ |

---

## Tips

1. **Use `make help`** for comprehensive list of targets
2. **Use `make info`** to check project status
3. **Use `make quick-start`** for initial setup
4. **Use `make clean`** before important runs
5. **Check logs** in `outputs/` directory
6. **Use poetry run** if CLI not found
7. **Read full guides** in `CLI_AND_MAKEFILE_GUIDE.md`

---

For detailed documentation, see:
- NoAug: `/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/CLI_AND_MAKEFILE_GUIDE.md`
- AUG: `/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/CLI_AND_MAKEFILE_GUIDE.md`
- Summary: `/experiment/YuNing/Refactored_Psy/CLI_IMPLEMENTATION_SUMMARY.md`
