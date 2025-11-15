# Contracts: Storage-Optimized Training & HPO Pipeline

**Branch**: `002-storage-optimized-training` | **Date**: 2025-10-11

This directory contains API contracts and interface specifications for the storage-optimized training and HPO pipeline.

## Overview

Since this is a CLI-driven ML training pipeline (not a web service), contracts define:
1. **CLI Command Interface**: Command-line interface specifications for training, HPO, evaluation, and cleanup operations
2. **Configuration Schemas**: JSON schemas for trial configurations and study configurations
3. **Output Schemas**: JSON schemas for evaluation reports and metrics logs

## Contract Files

### 1. CLI Interface (`cli_interface.md`)
Specifies command-line interface contracts for all user-facing operations.

### 2. Trial Configuration Schema (`trial_config_schema.json`)
JSON Schema for validating trial configuration files used in HPO and standalone training.

### 3. Evaluation Report Schema (`evaluation_report_schema.json`)
JSON Schema for per-study evaluation reports generated after HPO completion.

## CLI Command Interface

### Training Commands

#### `make train`
**Purpose**: Run standalone training (single trial, no HPO)

**Usage**:
```bash
make train CONFIG=configs/example_trial.yaml
```

**Parameters** (via config file):
- See `trial_config_schema.json` for full schema
- All TrialConfig fields from data-model.md

**Outputs**:
- Checkpoints: `experiments/trial_<uuid>/checkpoints/*.pt`
- Logs: `experiments/trial_<uuid>/logs/training.jsonl`
- MLflow run logged to local database

**Exit Codes**:
- `0`: Training completed successfully
- `1`: Configuration validation failed
- `2`: Training failed (error logged)
- `3`: Out of disk space (pruning exhausted)

---

#### `make train-resume`
**Purpose**: Resume interrupted training from latest checkpoint

**Usage**:
```bash
make train-resume TRIAL_ID=<uuid>
```

**Parameters**:
- `TRIAL_ID`: UUID of trial to resume

**Behavior**:
- Loads latest valid checkpoint from `experiments/trial_<uuid>/checkpoints/`
- Validates checkpoint integrity (checksum)
- Restores optimizer, scheduler, RNG states
- Continues training from next epoch
- Does not duplicate metrics in MLflow

**Exit Codes**:
- Same as `make train`
- `4`: No valid checkpoint found

---

### HPO Commands

#### `make hpo`
**Purpose**: Run hyperparameter optimization study (multiple trials)

**Usage**:
```bash
make hpo STUDY_CONFIG=configs/hpo_study.yaml N_TRIALS=1000
```

**Parameters** (via study config file):
- `study_name`: Human-readable study identifier
- `n_trials`: Number of trials to run (default: 100)
- `sampler`: Optuna sampler ("tpe", "random", "grid")
- `pruner`: Optional early stopping pruner
- `search_space`: Dict mapping TrialConfig fields to distributions
- `optimization_metric`: Metric to optimize (e.g., "val_f1_macro")
- `timeout_per_trial_hours`: Optional per-trial timeout

**Outputs**:
- Study directory: `experiments/study_<uuid>/`
- Trial directories: `experiments/study_<uuid>/trial_<uuid>/` (one per trial)
- Study-level evaluation report: `experiments/study_<uuid>/evaluation_report.json`
- Optuna study database: `<study_db_path>` (SQLite)
- MLflow experiment with all trial runs

**Behavior**:
- Executes trials sequentially (FR-021)
- Each trial evaluates on validation set during training
- After all trials complete, evaluates best model on test set (per-study evaluation, FR-007)
- Generates study-level JSON report with test metrics (FR-008)

**Exit Codes**:
- `0`: Study completed successfully
- `1`: Study configuration validation failed
- `2`: One or more trials failed (partial results available)

---

#### `make hpo-resume`
**Purpose**: Resume interrupted HPO study

**Usage**:
```bash
make hpo-resume STUDY_ID=<uuid>
```

**Parameters**:
- `STUDY_ID`: UUID of study to resume

**Behavior**:
- Loads Optuna study from database
- Identifies completed, failed, and pending trials
- Resumes from next pending trial
- Skips already-completed trials (idempotent)

---

### Evaluation Commands

#### `make evaluate`
**Purpose**: Evaluate trained model on test set

**Usage**:
```bash
make evaluate TRIAL_ID=<uuid> CHECKPOINT=<path_or_best>
```

**Parameters**:
- `TRIAL_ID`: UUID of trial to evaluate
- `CHECKPOINT`: Path to checkpoint file, or "best" to auto-select best checkpoint

**Outputs**:
- Console output with metrics table
- JSON file: `experiments/trial_<uuid>/test_metrics.json`

**Metrics Computed**:
- Criteria matching: accuracy, F1 (macro/micro), precision, recall, per-criterion breakdown
- Evidence binding: exact match, token-level F1, character-level F1

---

### Cleanup Commands

#### `make cleanup`
**Purpose**: Remove orphaned checkpoints and logs

**Usage**:
```bash
make cleanup [TRIAL_ID=<uuid>] [DRY_RUN=true]
```

**Parameters**:
- `TRIAL_ID`: Optional, clean specific trial (default: all trials)
- `DRY_RUN`: If true, only show what would be deleted (default: true)

**Behavior**:
- Detects orphaned checkpoints (not referenced by any trial metadata)
- Removes old logs and temporary files
- Preserves all MLflow tracking data
- Reports freed disk space

---

## Configuration Schema

### Trial Configuration (`trial_config_schema.json`)

**Purpose**: Validates trial configuration files for training and HPO.

**Schema** (JSON Schema Draft-07):
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": [
    "trial_id", "model_id", "input_format", "criteria_head_type",
    "evidence_head_type", "task_coupling", "loss_function",
    "activation", "learning_rate", "batch_size", "epochs",
    "optimization_metric", "seed"
  ],
  "properties": {
    "trial_id": {"type": "string", "format": "uuid"},
    "model_id": {
      "type": "string",
      "description": "Hugging Face model identifier",
      "examples": ["mental/mental-bert-base-uncased", "google-bert/bert-base-uncased"]
    },
    "input_format": {
      "type": "string",
      "enum": ["binary_pairs", "multi_label"]
    },
    ... (full schema omitted for brevity, see data-model.md for complete TrialConfig fields)
  }
}
```

**Validation**: Use `jsonschema` library to validate config files before training.

---

## Output Schema

### Evaluation Report (`evaluation_report_schema.json`)

**Purpose**: Validates per-study evaluation reports generated after HPO.

**Schema** (JSON Schema Draft-07):
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": [
    "report_id", "study_id", "best_trial_id", "generated_at",
    "config", "optimization_metric_name", "best_validation_score",
    "evaluated_checkpoints", "test_metrics", "report_file_path"
  ],
  "properties": {
    "report_id": {"type": "string", "format": "uuid"},
    "study_id": {"type": "string", "format": "uuid"},
    "best_trial_id": {"type": "string", "format": "uuid"},
    "generated_at": {"type": "string", "format": "date-time"},
    "config": {
      "type": "object",
      "description": "Full TrialConfig for the best trial"
    },
    "optimization_metric_name": {"type": "string"},
    "best_validation_score": {"type": "number"},
    "evaluated_checkpoints": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["checkpoint_id", "path", "epoch", "val_metric"],
        "properties": {
          "checkpoint_id": {"type": "string", "format": "uuid"},
          "path": {"type": "string"},
          "epoch": {"type": "integer"},
          "val_metric": {"type": "number"},
          "co_best": {"type": "boolean"}
        }
      }
    },
    "test_metrics": {
      "type": "object",
      "required": ["criteria_matching", "evidence_binding"],
      "properties": {
        "criteria_matching": {
          "type": "object",
          "required": ["accuracy", "f1_macro", "f1_micro", "per_criterion"],
          "properties": {
            "accuracy": {"type": "number", "minimum": 0, "maximum": 1},
            "f1_macro": {"type": "number", "minimum": 0, "maximum": 1},
            "f1_micro": {"type": "number", "minimum": 0, "maximum": 1},
            "precision_macro": {"type": "number", "minimum": 0, "maximum": 1},
            "recall_macro": {"type": "number", "minimum": 0, "maximum": 1},
            "per_criterion": {
              "type": "array",
              "items": {
                "type": "object",
                "required": ["id", "f1", "precision", "recall", "support"],
                "properties": {
                  "id": {"type": "string"},
                  "f1": {"type": "number"},
                  "precision": {"type": "number"},
                  "recall": {"type": "number"},
                  "support": {"type": "integer"}
                }
              }
            }
          }
        },
        "evidence_binding": {
          "type": "object",
          "required": ["exact_match", "f1"],
          "properties": {
            "exact_match": {"type": "number", "minimum": 0, "maximum": 1},
            "f1": {"type": "number", "minimum": 0, "maximum": 1},
            "char_f1": {"type": "number", "minimum": 0, "maximum": 1}
          }
        }
      }
    }
  }
}
```

---

## Usage Examples

### Example 1: Standalone Training
```bash
# Create trial config
cat > configs/trial_example.yaml <<EOF
trial_id: "550e8400-e29b-41d4-a716-446655440000"
model_id: "mental/mental-bert-base-uncased"
input_format: "multi_label"
criteria_head_type: "mlp_residual"
criteria_pooling: "cls"
criteria_hidden_dim: 512
evidence_head_type: "start_end_linear"
task_coupling: "independent"
loss_function: "weighted_bce"
learning_rate: 2e-5
batch_size: 16
epochs: 10
optimization_metric: "val_f1_macro"
seed: 42
EOF

# Run training
make train CONFIG=configs/trial_example.yaml

# Evaluate on test set
make evaluate TRIAL_ID=550e8400-e29b-41d4-a716-446655440000 CHECKPOINT=best
```

### Example 2: HPO Study
```bash
# Create HPO study config
cat > configs/hpo_study.yaml <<EOF
study_name: "mental_health_hpo_2025_10"
n_trials: 100
sampler: "tpe"
optimization_metric: "val_f1_macro"
search_space:
  model_id:
    type: categorical
    choices: ["mental/mental-bert-base-uncased", "google-bert/bert-base-uncased"]
  learning_rate:
    type: float
    low: 1e-6
    high: 1e-4
    log: true
  batch_size:
    type: categorical
    choices: [8, 16, 32]
  ... (full search space)
EOF

# Run HPO
make hpo STUDY_CONFIG=configs/hpo_study.yaml N_TRIALS=100

# Inspect results
cat experiments/study_<uuid>/evaluation_report.json
```

### Example 3: Resume Interrupted HPO
```bash
# If HPO interrupted, resume from latest state
make hpo-resume STUDY_ID=<uuid>

# Or resume specific trial
make train-resume TRIAL_ID=<uuid>
```

---

## Contract Validation

All contracts are validated at runtime:
1. **CLI argument validation**: `argparse` with type checking
2. **Config schema validation**: `jsonschema` library validates YAML configs against JSON schemas
3. **Output schema validation**: Generated JSON reports validated before writing to disk

**Testing**:
- Contract tests in `tests/contract/test_config_schema.py`
- Contract tests in `tests/contract/test_output_formats.py`

---

## Versioning

Contracts follow semantic versioning:
- **MAJOR**: Breaking changes to CLI interface or schema structure
- **MINOR**: Backward-compatible additions (new optional fields, new commands)
- **PATCH**: Clarifications, documentation fixes

Current version: **1.0.0**

---

## References

- Data Model: `../data-model.md`
- Feature Spec: `../spec.md`
- Research: `../research.md`
