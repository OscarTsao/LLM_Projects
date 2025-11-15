# CLI Contracts: Storage-Optimized Training & HPO Pipeline

**Feature**: Storage-Optimized Training & HPO Pipeline  
**Phase**: 1 (Design & Contracts)  
**Date**: 2025-01-14

## Overview

This document defines the command-line interface contracts for training, HPO, evaluation, and management operations.

## CLI Commands

### 1. Train Single Model

**Command**: `python -m dataaug_multi_both.cli.train`

**Purpose**: Train a dual-agent model with storage-optimized checkpointing.

**Arguments**:
```bash
--config CONFIG_PATH          # Path to Hydra config file (default: configs/train.yaml)
--model MODEL_ID              # Hugging Face model ID (e.g., "mental-bert")
--dataset DATASET_ID          # Hugging Face dataset ID (default: "irlab-udc/redsm5")
--optimization-metric METRIC  # Metric to optimize (e.g., "val_accuracy")
--max-epochs INT              # Maximum training epochs (default: 10)
--checkpoint-dir PATH         # Checkpoint output directory
--resume-from PATH            # Path to checkpoint for resuming
--seed INT                    # Random seed (default: 42)
--dry-run                     # Skip checkpointing, only log metrics
```

**Example**:
```bash
python -m dataaug_multi_both.cli.train \
  --model "mental-bert" \
  --dataset "irlab-udc/redsm5" \
  --optimization-metric "val_f1" \
  --max-epochs 20 \
  --seed 42
```

**Output**:
- Checkpoints saved to `experiments/trial_{uuid}/checkpoints/`
- Metrics logged to remote MLflow tracking server
- Structured JSON logs: `experiments/trial_{uuid}/logs/train.jsonl`
- Human-readable stdout progress

**Exit Codes**:
- 0: Success
- 1: Training error (e.g., OOM, convergence failure)
- 2: Configuration error (invalid parameters)
- 3: Data loading error (dataset/model not found)
- 4: Storage error (disk full, cannot save checkpoint)
- 5: Tracking error (MLflow unreachable after retries)

---

### 2. Run HPO Study

**Command**: `python -m dataaug_multi_both.cli.hpo`

**Purpose**: Execute hyperparameter optimization study with Optuna.

**Arguments**:
```bash
--study-name NAME             # Study name (default: auto-generated)
--config CONFIG_PATH          # Path to Hydra HPO config (default: configs/hpo.yaml)
--num-trials INT              # Number of trials (default: 100)
--optimization-metric METRIC  # Metric to optimize (e.g., "val_accuracy")
--direction DIRECTION         # Optimization direction (only "maximize" supported)
--study-dir PATH              # Study output directory
--resume-study STUDY_ID       # Resume existing study by ID
--seed INT                    # Random seed for reproducibility
```

**Example**:
```bash
python -m dataaug_multi_both.cli.hpo \
  --study-name "redsm5-hpo-001" \
  --num-trials 1000 \
  --optimization-metric "val_f1" \
  --direction maximize \
  --seed 42
```

**Output**:
- Study directory: `experiments/study_{uuid}/`
- Trial artifacts: `experiments/trial_{uuid}/` for each trial
- Optuna database: `experiments/study_{uuid}/optuna_study.db`
- Progress logged to stdout and `experiments/study_{uuid}/logs/hpo.jsonl`

**Exit Codes**:
- 0: Success (all trials completed)
- 1: Partial success (some trials failed, study resumed)
- 2: Configuration error
- 3: Study creation error
- 4: Storage exhaustion (cannot continue trials)
- 5: Tracking error (MLflow unreachable)

---

### 3. Evaluate Study on Test Set

**Command**: `python -m dataaug_multi_both.cli.evaluate_study`

**Purpose**: Evaluate the best model from an HPO study on the test set.

**Arguments**:
```bash
--study-dir PATH              # Study directory containing optuna_study.db
--output-path PATH            # Output path for evaluation report JSON (optional)
--test-split NAME             # Test split name (default: "test")
```

**Example**:
```bash
python -m dataaug_multi_both.cli.evaluate_study \
  --study-dir "experiments/study_abc123/" \
  --output-path "experiments/study_abc123/evaluation_report.json"
```

**Output**:
- Evaluation report JSON: `{study_dir}/evaluation_report.json`
- Test metrics logged to MLflow under best trial run
- Progress logged to stdout

**Exit Codes**:
- 0: Success
- 1: Evaluation error (model loading, inference failure)
- 2: Study not found or incomplete
- 3: Best trial checkpoint not found
- 4: Test dataset loading error

---

### 4. Resume Training

**Command**: `python -m dataaug_multi_both.cli.train --resume-from CHECKPOINT_PATH`

**Purpose**: Resume training from a saved checkpoint.

**Arguments**:
```bash
--resume-from PATH            # Path to checkpoint file
--config CONFIG_PATH          # Original training config (must match checkpoint)
--max-epochs INT              # New maximum epochs (extends training)
```

**Example**:
```bash
python -m dataaug_multi_both.cli.train \
  --resume-from "experiments/trial_xyz/checkpoints/epoch_005.pt" \
  --max-epochs 20
```

**Behavior**:
- Validates checkpoint integrity (SHA256 hash)
- Loads model state, optimizer state, epoch counter
- Continues training from next epoch
- Does not duplicate metrics in MLflow (uses same run ID)
- Falls back to previous checkpoint if validation fails

**Exit Codes**: Same as `train` command

---

### 5. Cleanup Artifacts

**Command**: `python -m dataaug_multi_both.cli.cleanup`

**Purpose**: Remove orphaned checkpoints and apply retention policies.

**Arguments**:
```bash
--study-dir PATH              # Study directory to clean (optional)
--trial-dir PATH              # Trial directory to clean (optional)
--dry-run                     # Show what would be deleted without deleting
--aggressive                  # Apply aggressive pruning (keep only best)
--orphaned-only               # Only remove orphaned artifacts (not referenced)
```

**Example**:
```bash
# Dry run to preview cleanup
python -m dataaug_multi_both.cli.cleanup \
  --study-dir "experiments/study_abc123/" \
  --dry-run

# Aggressive cleanup to free space
python -m dataaug_multi_both.cli.cleanup \
  --study-dir "experiments/study_abc123/" \
  --aggressive
```

**Output**:
- List of artifacts to delete (dry run) or deleted
- Space freed in GB
- Preserved checkpoints list

**Exit Codes**:
- 0: Success
- 1: Cleanup error (permission denied, I/O error)
- 2: Study/trial directory not found

---

### 6. List Studies

**Command**: `python -m dataaug_multi_both.cli.list_studies`

**Purpose**: List all HPO studies with summary statistics.

**Arguments**:
```bash
--experiments-dir PATH        # Experiments root directory (default: "experiments/")
--format FORMAT               # Output format: table, json, csv (default: table)
--filter-status STATUS        # Filter by status: completed, in_progress, failed
```

**Example**:
```bash
python -m dataaug_multi_both.cli.list_studies \
  --format table \
  --filter-status completed
```

**Output**:
```
Study ID       Name               Trials  Best Metric  Status      Created
─────────────────────────────────────────────────────────────────────────────
abc123...      redsm5-hpo-001     1000    0.8542       completed   2025-01-10
def456...      redsm5-hpo-002     500     0.8301       in_progress 2025-01-12
```

**Exit Codes**:
- 0: Success
- 1: Experiments directory not found

---

### 7. List Trial Checkpoints

**Command**: `python -m dataaug_multi_both.cli.list_checkpoints`

**Purpose**: List checkpoints for a trial with retention status.

**Arguments**:
```bash
--trial-dir PATH              # Trial directory
--format FORMAT               # Output format: table, json, csv (default: table)
--show-metrics                # Include metrics in output
```

**Example**:
```bash
python -m dataaug_multi_both.cli.list_checkpoints \
  --trial-dir "experiments/trial_xyz/" \
  --show-metrics
```

**Output**:
```
Checkpoint         Epoch  Step   Val F1   Is Best  Is Latest  Retained  Size (MB)
──────────────────────────────────────────────────────────────────────────────────
epoch_001.pt       1      500    0.7234   False    False      False     1200.5
epoch_005.pt       5      2500   0.8542   True     False      True      1205.3
epoch_010.pt       10     5000   0.8301   False    True       True      1208.7
```

**Exit Codes**:
- 0: Success
- 1: Trial directory not found

---

## Configuration Contracts

### Training Config Schema (configs/train.yaml)

```yaml
model:
  model_id: str                    # Hugging Face model ID
  revision: Optional[str]          # Model revision/tag
  num_criteria_classes: int        # Number of criteria categories
  task_weights:
    criteria_matcher: float        # Loss weight (default: 1.0)
    evidence_binder: float         # Loss weight (default: 1.0)

dataset:
  dataset_id: str                  # Hugging Face dataset ID
  revision: str                    # Dataset revision (default: "main")
  splits:
    train: str                     # Train split name (default: "train")
    validation: str                # Validation split name (default: "validation")
    test: str                      # Test split name (default: "test")

trainer:
  max_epochs: int                  # Maximum training epochs
  batch_size: int                  # Training batch size
  learning_rate: float             # Initial learning rate
  optimization_metric: str         # Metric to optimize
  gradient_accumulation_steps: int # Gradient accumulation (default: 1)
  mixed_precision: bool            # Use mixed precision (default: true)

retention:
  keep_last_n: int                 # Keep last N checkpoints (default: 1)
  keep_best_k: int                 # Keep best K checkpoints (default: 1)
  keep_best_k_max: int             # Max best checkpoints (default: 2)
  max_total_size_gb: float         # Max checkpoint storage GB (default: 10)
  min_interval_epochs: int         # Min epochs between checkpoints (default: 1)
  disk_space_threshold_percent: float  # Disk threshold % (default: 10.0)

tracking:
  mlflow_tracking_uri: str         # MLflow server URI (e.g., https://mlflow.example.com)
  vault_addr: str                  # Vault server address (e.g., https://vault.example.com)
  vault_token_path: str            # Path to Vault token secret
  experiment_name: str             # MLflow experiment name

seeds:
  python: int                      # Python random seed
  numpy: int                       # NumPy seed
  torch: int                       # PyTorch seed
  torch_cuda: int                  # PyTorch CUDA seed
```

---

### HPO Config Schema (configs/hpo.yaml)

```yaml
study:
  name: str                        # Study name
  num_trials: int                  # Number of trials
  optimization_metric: str         # Metric to optimize
  direction: str                   # "maximize" only
  storage: str                     # Optuna storage URI (e.g., "sqlite:///{study_dir}/optuna.db")
  sampler:
    type: str                      # Sampler type (e.g., "TPESampler")
    seed: int                      # Sampler seed
  pruner:
    type: str                      # Pruner type (e.g., "MedianPruner")
    n_startup_trials: int          # Warmup trials before pruning
    n_warmup_steps: int            # Warmup steps within trial

search_space:
  model_id:
    type: categorical
    choices: [mental-bert, psychbert, clinicalbert, bert-base-uncased, roberta-base]
  
  learning_rate:
    type: loguniform
    low: 1e-5
    high: 1e-3
  
  batch_size:
    type: categorical
    choices: [8, 16, 32]
  
  task_weights.criteria_matcher:
    type: uniform
    low: 0.5
    high: 2.0
  
  task_weights.evidence_binder:
    type: uniform
    low: 0.5
    high: 2.0

# Inherit base training config
defaults:
  - train
  - _self_
```

---

## Makefile Contracts

### Common Operations

```makefile
# Training
.PHONY: train
train:
	poetry run python -m dataaug_multi_both.cli.train --config configs/train.yaml

# HPO
.PHONY: hpo
hpo:
	poetry run python -m dataaug_multi_both.cli.hpo --config configs/hpo.yaml --num-trials 100

# Resume training from last checkpoint
.PHONY: resume
resume:
	poetry run python -m dataaug_multi_both.cli.train --resume-from $(CHECKPOINT_PATH)

# Evaluate study on test set
.PHONY: evaluate
evaluate:
	poetry run python -m dataaug_multi_both.cli.evaluate_study --study-dir $(STUDY_DIR)

# Cleanup orphaned artifacts
.PHONY: clean-artifacts
clean-artifacts:
	poetry run python -m dataaug_multi_both.cli.cleanup --orphaned-only --dry-run

# List all studies
.PHONY: list-studies
list-studies:
	poetry run python -m dataaug_multi_both.cli.list_studies --format table

# Run tests
.PHONY: test
test:
	poetry run pytest -v --cov=src/dataaug_multi_both

# Lint and format
.PHONY: lint
lint:
	poetry run ruff check .
	poetry run black --check .
	poetry run mypy src

.PHONY: format
format:
	poetry run black .
	poetry run ruff check --fix .

# Docker build
.PHONY: docker-build
docker-build:
	docker build -t dataaug-multi-both:latest -f docker/Dockerfile .

# Docker run (interactive)
.PHONY: docker-run
docker-run:
	docker run -it --gpus all -v $(PWD):/workspace dataaug-multi-both:latest

# Help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  train           - Train a single model"
	@echo "  hpo             - Run HPO study"
	@echo "  resume          - Resume training (set CHECKPOINT_PATH)"
	@echo "  evaluate        - Evaluate study on test set (set STUDY_DIR)"
	@echo "  clean-artifacts - Cleanup orphaned artifacts"
	@echo "  list-studies    - List all HPO studies"
	@echo "  test            - Run test suite"
	@echo "  lint            - Run linters and type checks"
	@echo "  format          - Format code with black and ruff"
	@echo "  docker-build    - Build Docker image"
	@echo "  docker-run      - Run Docker container interactively"
```

---

## MLflow Tracking Contracts

### Logged Metrics (Per Epoch)

**Criteria Matcher**:
- `train/criteria_matcher/loss`: Training loss
- `train/criteria_matcher/accuracy`: Training accuracy
- `val/criteria_matcher/loss`: Validation loss
- `val/criteria_matcher/accuracy`: Validation accuracy
- `val/criteria_matcher/precision`: Validation precision
- `val/criteria_matcher/recall`: Validation recall
- `val/criteria_matcher/f1`: Validation F1 score
- `val/criteria_matcher/auc`: Validation AUC

**Evidence Binder**:
- `train/evidence_binder/loss`: Training loss
- `val/evidence_binder/exact_match`: Validation exact match
- `val/evidence_binder/has_answer`: Validation has-answer score
- `val/evidence_binder/char_f1`: Validation character-level F1
- `val/evidence_binder/char_precision`: Validation character-level precision
- `val/evidence_binder/char_recall`: Validation character-level recall

**System Metrics**:
- `system/checkpoint_size_mb`: Checkpoint size in MB
- `system/disk_free_gb`: Available disk space in GB
- `system/epoch_duration_sec`: Epoch training duration

### Logged Parameters

```python
{
  "model_id": "mental-bert",
  "model_revision": "main",
  "dataset_id": "irlab-udc/redsm5",
  "dataset_revision": "abc123",
  "learning_rate": 5e-5,
  "batch_size": 16,
  "max_epochs": 10,
  "optimization_metric": "val_f1",
  "seed_python": 42,
  "seed_numpy": 42,
  "seed_torch": 42,
  "task_weight_criteria": 1.0,
  "task_weight_evidence": 1.0
}
```

### Logged Tags

```python
{
  "study_id": "study_abc123",
  "trial_id": "trial_xyz789",
  "trial_number": 42,
  "framework": "pytorch",
  "mlflow.source.type": "LOCAL",
  "mlflow.runName": "trial_042_mental-bert"
}
```

---

## Summary

This contract document defines:

1. **7 CLI commands**: train, hpo, evaluate_study, resume, cleanup, list_studies, list_checkpoints
2. **Configuration schemas**: train.yaml and hpo.yaml with Hydra
3. **Makefile targets**: Common operations for training, evaluation, testing, and Docker
4. **MLflow tracking contracts**: Metrics, parameters, and tags logged for each trial

All contracts follow REST-like principles with clear inputs, outputs, and exit codes for programmatic integration and automation.
