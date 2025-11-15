# CLI and Makefile Guide - Augmentation Repository

This guide documents the unified command-line interface (CLI) and Makefile for the PSY Agents AUG repository with augmentation support.

## Table of Contents
- [CLI Interface](#cli-interface)
- [Makefile Targets](#makefile-targets)
- [Augmentation Features](#augmentation-features)
- [Quick Start](#quick-start)
- [Common Workflows](#common-workflows)
- [Advanced Usage](#advanced-usage)

---

## CLI Interface

The CLI provides a unified interface for all operations with full augmentation support using Hydra for configuration management.

### Available Commands

#### 1. make_groundtruth
Generate ground truth files from raw data (same as NoAug repository).

```bash
# From HuggingFace
python -m psy_agents_aug.cli make_groundtruth data=hf_redsm5

# From local CSV
python -m psy_agents_aug.cli make_groundtruth data=local_csv
```

#### 2. train
Train a model with optional augmentation support.

```bash
# Without augmentation (baseline)
python -m psy_agents_aug.cli train task=criteria model=roberta_base

# With augmentation enabled
python -m psy_agents_aug.cli train task=criteria augment.enabled=true

# Specify augmentation pipeline
python -m psy_agents_aug.cli train task=criteria \
    augment.enabled=true \
    augment.pipeline=nlpaug_pipeline

# Customize augmentation probability
python -m psy_agents_aug.cli train task=criteria \
    augment.enabled=true \
    augment.p=0.7

# Full customization
python -m psy_agents_aug.cli train task=criteria \
    augment.enabled=true \
    augment.pipeline=nlpaug_pipeline \
    augment.p=0.5 \
    augment.n_augmented_samples=2
```

**Augmentation Parameters:**
- `augment.enabled`: Enable/disable augmentation (default: false)
- `augment.pipeline`: Pipeline name (nlpaug_pipeline, textattack_pipeline, hybrid_pipeline)
- `augment.p`: Probability of augmentation (default: 0.5)
- `augment.train_only`: Apply only to training data (default: true)
- `augment.n_augmented_samples`: Number of augmented samples per original (default: 1)

#### 3. hpo
Run hyperparameter optimization with augmentation support.

```bash
# Stage 0: Sanity check without augmentation
python -m psy_agents_aug.cli hpo hpo=stage0_sanity task=criteria

# Stage 1: Coarse search with augmentation
python -m psy_agents_aug.cli hpo hpo=stage1_coarse task=criteria \
    augment.enabled=true \
    augment.pipeline=nlpaug_pipeline

# Stage 2: Fine search with augmentation
python -m psy_agents_aug.cli hpo hpo=stage2_fine task=criteria \
    augment.enabled=true
```

#### 4. test_augmentation
Test augmentation pipelines (augmentation-specific command).

```bash
# Test NLPAug pipeline
python -m psy_agents_aug.cli test_augmentation --pipeline nlpaug

# Test TextAttack pipeline
python -m psy_agents_aug.cli test_augmentation --pipeline textattack

# Test Hybrid pipeline
python -m psy_agents_aug.cli test_augmentation --pipeline hybrid

# Custom text and number of samples
python -m psy_agents_aug.cli test_augmentation \
    --pipeline nlpaug \
    --text "I feel sad and have no energy" \
    --n 5
```

#### 5. refit
Refit best model on full train+val dataset.

```bash
# Load best config from HPO stage 2
python -m psy_agents_aug.cli refit task=criteria best_config=outputs/hpo_stage2/best_config.yaml
```

#### 6. evaluate_best
Evaluate best model on test set.

```bash
# Evaluate with checkpoint
python -m psy_agents_aug.cli evaluate_best checkpoint=outputs/best_model.pt
```

#### 7. export_metrics
Export metrics from MLflow to CSV/JSON.

```bash
# Export default experiment
python -m psy_agents_aug.cli export_metrics

# Export specific experiment
python -m psy_agents_aug.cli export_metrics mlflow.experiment_name=aug_baseline
```

---

## Makefile Targets

The Makefile provides convenient shortcuts with augmentation-specific targets.

### Setup

```bash
make setup              # Full setup (install + pre-commit + sanity check)
make install            # Install dependencies with poetry
make install-dev        # Install with development dependencies
make sanity-check       # Run sanity checks (includes augment module)
```

### Data Generation

```bash
make groundtruth        # Generate ground truth from HuggingFace
make groundtruth-local  # Generate ground truth from local CSV
```

### Training

```bash
# Standard training (no augmentation)
make train              # Train default model (criteria, roberta_base)
make train-evidence     # Train evidence task

# Training with augmentation
make train-aug          # Train with augmentation enabled (default: nlpaug_pipeline)
make train-aug AUG_PIPELINE=textattack_pipeline  # Use TextAttack pipeline
make train-aug TASK=evidence  # Train evidence task with augmentation
```

### Hyperparameter Optimization

```bash
# Without augmentation
make hpo-s0             # Stage 0: Sanity check
make hpo-s1             # Stage 1: Coarse search
make hpo-s2             # Stage 2: Fine search

# With augmentation
make hpo-s0-aug         # Stage 0 with augmentation
make hpo-s1-aug         # Stage 1 with augmentation
make hpo-s2-aug         # Stage 2 with augmentation

# Custom pipeline
make hpo-s1-aug HPO_TASK=evidence AUG_PIPELINE=textattack_pipeline

# Complete pipeline
make refit              # Stage 3: Refit best model
```

### Augmentation Testing

```bash
# Run all augmentation tests
make test-aug           # Run all augmentation-related tests

# Specific test suites
make test-contract      # Test augmentation contracts (determinism, train-only)
make test-pipelines     # Test augmentation pipelines
make test-no-leak       # Test no data leakage in val/test

# Verification with examples
make verify-aug         # Verify augmentation setup with examples
```

### Evaluation

```bash
make eval               # Evaluate best model
make export             # Export metrics from MLflow
make eval CHECKPOINT=outputs/custom_model.pt  # Custom checkpoint
```

### Development

```bash
make lint               # Run linters (ruff + black --check)
make format             # Format code (ruff --fix + black)
make test               # Run all tests (including augmentation tests)
make test-cov           # Run tests with coverage
make test-groundtruth   # Run ground truth tests only
```

### Pre-commit

```bash
make pre-commit-install # Install pre-commit hooks
make pre-commit-run     # Run pre-commit on all files
```

### Cleaning

```bash
make clean              # Remove caches and temp files
make clean-all          # Clean everything (including data/mlruns)
```

### Quick Workflows

```bash
make quick-start        # setup + groundtruth + verify-aug
make full-hpo           # Run complete HPO pipeline (stages 0-3, no aug)
make full-hpo-aug       # Run complete HPO pipeline with augmentation
make compare-aug        # Run experiments comparing with/without augmentation
make info               # Show project information
```

---

## Augmentation Features

### Available Pipelines

1. **NLPAug Pipeline** (`nlpaug_pipeline`)
   - Synonym replacement using WordNet
   - Fast and deterministic
   - Best for general text augmentation

2. **TextAttack Pipeline** (`textattack_pipeline`)
   - More sophisticated transformations
   - Embedding-based augmentation
   - Requires additional models

3. **Hybrid Pipeline** (`hybrid_pipeline`)
   - Combines multiple techniques
   - More diverse augmentations
   - May be slower

### Testing Augmentation

```bash
# Quick verification
make verify-aug

# Run all augmentation tests
make test-aug

# Test specific aspects
make test-contract      # Contracts (determinism, train-only)
make test-pipelines     # Pipeline functionality
make test-no-leak       # Data leakage prevention
```

### Augmentation Constraints

The augmentation module enforces these contracts:

1. **Determinism**: Same input + same seed → same output
2. **Train-only**: Augmentation applied only to training data
3. **No leakage**: Val/test data never augmented
4. **Preservability**: Can disable augmentation without breaking code

---

## Quick Start

### 1. Initial Setup

```bash
# Complete setup
make setup

# Verify augmentation setup
make verify-aug
```

### 2. Generate Ground Truth

```bash
make groundtruth
```

### 3. Test Augmentation

```bash
# Verify augmentation works
make verify-aug

# Run augmentation tests
make test-aug
```

### 4. Train with Augmentation

```bash
# Train with default augmentation
make train-aug

# Or with specific pipeline
make train-aug AUG_PIPELINE=nlpaug_pipeline
```

### 5. HPO with Augmentation

```bash
# Run complete HPO pipeline with augmentation
make full-hpo-aug

# Or step by step
make hpo-s0-aug
make hpo-s1-aug
make hpo-s2-aug
make refit
```

---

## Common Workflows

### Workflow 1: Baseline vs Augmentation Comparison

```bash
# 1. Setup
make setup
make groundtruth

# 2. Train baseline (no augmentation)
make train TASK=criteria MODEL=roberta_base

# 3. Train with augmentation
make train-aug TASK=criteria MODEL=roberta_base

# 4. Compare results
make export
```

### Workflow 2: HPO with Augmentation

```bash
# 1. Setup
make quick-start

# 2. Run HPO with augmentation
make hpo-s1-aug HPO_TASK=criteria AUG_PIPELINE=nlpaug_pipeline
make hpo-s2-aug HPO_TASK=criteria
make refit

# 3. Evaluate
make eval
```

### Workflow 3: Test Different Augmentation Pipelines

```bash
# Test NLPAug
make train-aug AUG_PIPELINE=nlpaug_pipeline

# Test TextAttack
make train-aug AUG_PIPELINE=textattack_pipeline

# Test Hybrid
make train-aug AUG_PIPELINE=hybrid_pipeline

# Compare results
make export
```

### Workflow 4: Complete Comparison Study

```bash
# Automated comparison
make compare-aug TASK=criteria MODEL=roberta_base
```

---

## Advanced Usage

### Custom Augmentation Configuration

```bash
# Create custom augmentation config
# configs/augment/custom.yaml

# Use it
python -m psy_agents_aug.cli train \
    task=criteria \
    augment=custom \
    augment.enabled=true
```

### Multiple Augmentation Strategies

```bash
# Try different augmentation probabilities
python -m psy_agents_aug.cli train \
    task=criteria \
    -m augment.p=0.3,0.5,0.7 \
    augment.enabled=true
```

### Augmentation in HPO

The HPO process can also optimize augmentation hyperparameters:

```bash
# HPO will search over augmentation parameters if enabled
python -m psy_agents_aug.cli hpo \
    hpo=stage1_coarse \
    task=criteria \
    augment.enabled=true
```

### Debugging Augmentation

```bash
# Test augmentation manually
python -m psy_agents_aug.cli test_augmentation \
    --pipeline nlpaug \
    --text "Your text here" \
    --n 10

# Verify no leakage
make test-no-leak

# Check determinism
make test-contract
```

---

## Configuration Files

### Augmentation-specific Configs

- `configs/augment/` - Augmentation pipeline configs
  - `nlpaug_pipeline.yaml` - NLPAug configuration
  - `textattack_pipeline.yaml` - TextAttack configuration
  - `hybrid_pipeline.yaml` - Hybrid pipeline configuration

### Example: Custom Augmentation Config

Create `configs/augment/custom_pipeline.yaml`:

```yaml
pipeline: nlpaug_pipeline
enabled: true
p: 0.5
n_augmented_samples: 2
train_only: true
seed: 42
```

Use it:

```bash
python -m psy_agents_aug.cli train \
    task=criteria \
    augment=custom_pipeline
```

---

## Comparison with NoAug Repository

### Similarities

- Same CLI structure for core commands
- Same Makefile organization
- Same data processing pipeline
- Same training infrastructure

### Differences

**AUG Repository Additions:**

1. **CLI Command**: `test_augmentation`
2. **Makefile Targets**:
   - `train-aug` - Training with augmentation
   - `hpo-s0-aug`, `hpo-s1-aug`, `hpo-s2-aug` - HPO with augmentation
   - `test-aug` - Augmentation tests
   - `test-contract`, `test-pipelines`, `test-no-leak` - Specific augmentation tests
   - `verify-aug` - Augmentation verification
   - `compare-aug` - Automated comparison
   - `full-hpo-aug` - Complete HPO with augmentation

3. **Configuration Parameters**:
   - `augment.enabled`
   - `augment.pipeline`
   - `augment.p`
   - `augment.n_augmented_samples`
   - `augment.train_only`

---

## Troubleshooting

### Augmentation-specific Issues

1. **Pipeline not found**
   ```bash
   # Check available pipelines
   ls configs/augment/
   
   # Verify module imports
   poetry run python -c "from psy_agents_aug.augment.pipelines import create_nlpaug_pipeline; print('OK')"
   ```

2. **Augmentation too slow**
   ```bash
   # Use faster pipeline
   make train-aug AUG_PIPELINE=nlpaug_pipeline
   
   # Reduce augmentation probability
   python -m psy_agents_aug.cli train augment.p=0.3
   ```

3. **Augmentation contracts failing**
   ```bash
   # Run contract tests
   make test-contract
   
   # Check determinism
   poetry run pytest tests/test_augment_contract.py::test_determinism -v
   ```

4. **Data leakage concerns**
   ```bash
   # Verify no leakage
   make test-no-leak
   ```

---

## Additional Resources

- **NLPAug Documentation**: https://github.com/makcedward/nlpaug
- **TextAttack Documentation**: https://textattack.readthedocs.io/
- **Hydra Documentation**: https://hydra.cc/
- **MLflow Documentation**: https://mlflow.org/
- **Optuna Documentation**: https://optuna.org/

---

## Summary

The AUG repository extends the NoAug repository with full augmentation support:

**Key Features:**
- Drop-in augmentation support for all training operations
- Multiple augmentation pipelines (NLPAug, TextAttack, Hybrid)
- Comprehensive augmentation testing
- Easy comparison between augmented and non-augmented training
- Strict contracts ensuring data integrity

**Quick Commands:**
```bash
make setup              # Initial setup
make verify-aug         # Verify augmentation works
make train-aug          # Train with augmentation
make hpo-s1-aug         # HPO with augmentation
make compare-aug        # Compare with/without augmentation
```

Use augmentation to improve model performance on low-resource tasks!
