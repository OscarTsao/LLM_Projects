# CLI and Makefile Guide

This guide documents the unified command-line interface (CLI) and Makefile for the PSY Agents NO-AUG repository.

## Table of Contents
- [CLI Interface](#cli-interface)
- [Makefile Targets](#makefile-targets)
- [Quick Start](#quick-start)
- [Common Workflows](#common-workflows)
- [Advanced Usage](#advanced-usage)

---

## CLI Interface

The CLI provides a unified interface for all operations using Hydra for configuration management.

### Available Commands

#### 1. make_groundtruth
Generate ground truth files from raw data.

```bash
# From HuggingFace
python -m psy_agents_noaug.cli make_groundtruth data=hf_redsm5

# From local CSV
python -m psy_agents_noaug.cli make_groundtruth data=local_csv

# Custom data directory
python -m psy_agents_noaug.cli make_groundtruth data.data_dir=./custom/path
```

**What it does:**
- Loads posts, annotations, and DSM criteria
- Validates required columns per field_map.yaml
- Generates `criteria_groundtruth.csv` (uses ONLY status field)
- Generates `evidence_groundtruth.csv` (uses ONLY cases field)
- Generates `splits.json` with train/val/test post_ids
- Validates strict field separation

#### 2. train
Train a model with specified configuration.

```bash
# Default training (criteria task, roberta_base)
python -m psy_agents_noaug.cli train task=criteria model=roberta_base

# Evidence task
python -m psy_agents_noaug.cli train task=evidence

# Custom hyperparameters
python -m psy_agents_noaug.cli train task=criteria training.num_epochs=20 training.batch_size=32

# Override learning rate
python -m psy_agents_noaug.cli train task=criteria training.optimizer.lr=2e-5
```

**Configuration:**
- Task: criteria, evidence
- Model: roberta_base, bert_base, etc.
- All training parameters from configs/training/

#### 3. hpo
Run hyperparameter optimization stage.

```bash
# Stage 0: Sanity check (2 trials)
python -m psy_agents_noaug.cli hpo hpo=stage0_sanity task=criteria

# Stage 1: Coarse search (20 trials)
python -m psy_agents_noaug.cli hpo hpo=stage1_coarse task=criteria model=roberta_base

# Stage 2: Fine search (50 trials)
python -m psy_agents_noaug.cli hpo hpo=stage2_fine task=criteria

# Custom number of trials
python -m psy_agents_noaug.cli hpo hpo=stage1_coarse hpo.n_trials=100
```

**HPO Stages:**
- Stage 0: Sanity check (2 trials) - verify setup
- Stage 1: Coarse search (20 trials) - explore broad space
- Stage 2: Fine search (50 trials) - refine best region
- Stage 3: Refit (1 trial) - train on train+val

#### 4. refit
Refit best model on full train+val dataset.

```bash
# Load best config from HPO stage 2
python -m psy_agents_noaug.cli refit task=criteria best_config=outputs/hpo_stage2/best_config.yaml
```

#### 5. evaluate_best
Evaluate best model on test set.

```bash
# Evaluate with checkpoint
python -m psy_agents_noaug.cli evaluate_best checkpoint=outputs/best_model.pt

# With specific task
python -m psy_agents_noaug.cli evaluate_best task=criteria checkpoint=outputs/checkpoints/best_checkpoint.pt
```

#### 6. export_metrics
Export metrics from MLflow to CSV/JSON.

```bash
# Export default experiment
python -m psy_agents_noaug.cli export_metrics

# Export specific experiment
python -m psy_agents_noaug.cli export_metrics mlflow.experiment_name=custom_exp

# Custom output directory
python -m psy_agents_noaug.cli export_metrics output_dir=./results
```

---

## Makefile Targets

The Makefile provides convenient shortcuts for common operations.

### Setup

```bash
make setup              # Full setup (install + pre-commit + sanity check)
make install            # Install dependencies with poetry
make install-dev        # Install with development dependencies
make sanity-check       # Run sanity checks
```

### Data Generation

```bash
make groundtruth        # Generate ground truth from HuggingFace
make groundtruth-local  # Generate ground truth from local CSV
```

### Training

```bash
make train              # Train default model (criteria, roberta_base)
make train-evidence     # Train evidence task
make train TASK=criteria MODEL=roberta_base  # Custom task/model
```

### Hyperparameter Optimization

```bash
make hpo-s0             # Stage 0: Sanity check
make hpo-s1             # Stage 1: Coarse search
make hpo-s2             # Stage 2: Fine search
make refit              # Stage 3: Refit best model

# With custom task
make hpo-s1 HPO_TASK=evidence HPO_MODEL=bert_base
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
make test               # Run all tests
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
make quick-start        # setup + groundtruth + hpo-s0
make full-hpo           # Run complete HPO pipeline (stages 0-3)
make info               # Show project information
```

---

## Quick Start

### 1. Initial Setup

```bash
# Clone repository and navigate to it
cd /path/to/NoAug_Criteria_Evidence

# Complete setup
make setup

# Or manually:
make install
make pre-commit-install
make sanity-check
```

### 2. Generate Ground Truth

```bash
# From HuggingFace (recommended)
make groundtruth

# Or from local CSV files
make groundtruth-local
```

### 3. Run Sanity Check

```bash
# Quick HPO sanity check (2 trials)
make hpo-s0
```

### 4. Full Training Pipeline

```bash
# Option A: Using Makefile
make hpo-s1              # Coarse search
make hpo-s2              # Fine search
make refit               # Refit best model
make eval                # Evaluate on test

# Option B: Using CLI directly
python -m psy_agents_noaug.cli hpo hpo=stage1_coarse task=criteria
python -m psy_agents_noaug.cli hpo hpo=stage2_fine task=criteria
python -m psy_agents_noaug.cli refit task=criteria best_config=outputs/hpo_stage2/best_config.yaml
python -m psy_agents_noaug.cli evaluate_best checkpoint=outputs/best_model.pt
```

---

## Common Workflows

### Workflow 1: Complete HPO Pipeline

```bash
# 1. Setup (one time)
make setup
make groundtruth

# 2. Run full HPO
make full-hpo

# 3. Export results
make export
```

### Workflow 2: Quick Experimentation

```bash
# 1. Setup
make quick-start

# 2. Train a model directly (skip HPO)
make train TASK=criteria MODEL=roberta_base

# 3. Evaluate
make eval CHECKPOINT=outputs/checkpoints/best_checkpoint.pt
```

### Workflow 3: Custom Configuration

```bash
# Train with custom hyperparameters
python -m psy_agents_noaug.cli train \
    task=criteria \
    model=roberta_base \
    training.num_epochs=30 \
    training.batch_size=16 \
    training.optimizer.lr=3e-5 \
    training.gradient_clip=1.0
```

### Workflow 4: Development Cycle

```bash
# 1. Make code changes
# ... edit files ...

# 2. Format and lint
make format
make lint

# 3. Run tests
make test

# 4. Pre-commit check
make pre-commit-run

# 5. Commit changes
git add .
git commit -m "Your message"
```

---

## Advanced Usage

### Hydra Configuration Overrides

The CLI uses Hydra for configuration management. You can override any configuration parameter:

```bash
# Override nested parameters
python -m psy_agents_noaug.cli train \
    task=criteria \
    model.hidden_size=768 \
    training.optimizer.lr=2e-5 \
    training.scheduler.warmup_steps=500

# Use different config groups
python -m psy_agents_noaug.cli train \
    task=evidence \
    model=bert_base \
    data=local_csv
```

### Hydra Multirun

Run experiments with different parameters:

```bash
# Try multiple learning rates
python -m psy_agents_noaug.cli train \
    task=criteria \
    -m training.optimizer.lr=1e-5,2e-5,3e-5

# Combine multiple parameters
python -m psy_agents_noaug.cli train \
    task=criteria \
    -m training.batch_size=16,32 training.optimizer.lr=1e-5,2e-5
```

### Environment Variables

```bash
# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=sqlite:///$PWD/mlflow.db
export MLFLOW_ARTIFACT_URI=file://$PWD/mlruns

# Set device
export CUDA_VISIBLE_DEVICES=0

# Run training
make train
```

### Debugging

```bash
# Verbose logging
python -m psy_agents_noaug.cli train task=criteria -v

# Hydra debug mode
python -m psy_agents_noaug.cli train task=criteria hydra.verbose=true

# Check configuration without running
python -m psy_agents_noaug.cli train task=criteria --cfg job
```

### Custom Output Directories

```bash
# Change output directory
python -m psy_agents_noaug.cli train \
    task=criteria \
    output_dir=./custom_outputs

# Use timestamp in output directory
python -m psy_agents_noaug.cli train \
    task=criteria \
    hydra.run.dir=outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

---

## Configuration Files

### Main Configuration
- `configs/config.yaml` - Main configuration file
- `configs/task/` - Task-specific configs (criteria.yaml, evidence.yaml)
- `configs/model/` - Model configs (roberta_base.yaml, bert_base.yaml)
- `configs/training/` - Training configs
- `configs/hpo/` - HPO stage configs
- `configs/data/` - Data source configs

### Example: Custom Task Config

Create `configs/task/custom_task.yaml`:

```yaml
task_name: custom_task
task_type: classification
num_classes: 3
class_names: [class_0, class_1, class_2]
```

Use it:

```bash
python -m psy_agents_noaug.cli train task=custom_task
```

---

## Troubleshooting

### Common Issues

1. **Module not found**: Ensure you're in the repository root or using poetry run
   ```bash
   poetry run python -m psy_agents_noaug.cli train task=criteria
   ```

2. **Config not found**: Check config path is relative to configs/
   ```bash
   # Correct
   python -m psy_agents_noaug.cli train task=criteria
   
   # Incorrect
   python -m psy_agents_noaug.cli train task=configs/task/criteria.yaml
   ```

3. **MLflow errors**: Check tracking URI
   ```bash
   export MLFLOW_TRACKING_URI=sqlite:///$PWD/mlflow.db
   export MLFLOW_ARTIFACT_URI=file://$PWD/mlruns
   ```

4. **CUDA out of memory**: Reduce batch size
   ```bash
   make train TASK=criteria training.batch_size=8
   ```

---

## Additional Resources

- **Hydra Documentation**: https://hydra.cc/
- **MLflow Documentation**: https://mlflow.org/
- **Optuna Documentation**: https://optuna.org/

---

## Summary

The CLI and Makefile provide a unified, powerful interface for all operations:

**CLI Advantages:**
- Hydra-based configuration management
- Easy parameter overrides
- Multi-run support
- Structured logging

**Makefile Advantages:**
- Quick shortcuts for common tasks
- Memorable commands
- Workflow automation
- Colored output

Use the CLI for precise control and the Makefile for convenience!
