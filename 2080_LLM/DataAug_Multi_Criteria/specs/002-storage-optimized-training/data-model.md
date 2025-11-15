# Data Model: Storage-Optimized Training & HPO Pipeline

**Feature**: Storage-Optimized Training & HPO Pipeline  
**Phase**: 1 (Design & Contracts)  
**Date**: 2025-01-14

## Overview

This document defines the core entities and their relationships for the storage-optimized training and HPO pipeline.

## Core Entities

### 1. Trial

Represents a single hyperparameter optimization trial with its complete lifecycle.

**Fields**:
- `trial_id`: str (UUID) - Unique identifier for the trial
- `study_id`: str (UUID) - Parent study identifier
- `trial_number`: int - Sequential trial number within study
- `status`: TrialStatus (enum) - Current trial state
- `params`: Dict[str, Any] - Hyperparameters for this trial
- `optimization_metric_name`: str - Metric name to optimize (e.g., "val_accuracy")
- `optimization_metric_value`: Optional[float] - Best value achieved for optimization metric
- `best_checkpoint_paths`: List[str] - Paths to best checkpoint(s) (multiple if co-best)
- `latest_checkpoint_path`: Optional[str] - Path to most recent checkpoint
- `artifact_dir`: str - Trial artifact directory (e.g., "experiments/trial_{uuid}")
- `mlflow_run_id`: str - MLflow run identifier
- `created_at`: datetime
- `started_at`: Optional[datetime]
- `completed_at`: Optional[datetime]
- `error_message`: Optional[str] - Error details if status == failed

**Validation Rules**:
- `trial_id` must be valid UUID v4
- `optimization_metric_name` must exist in logged metrics
- `status` transitions: queued → preparing → running → finishing → (completed | failed)
- `best_checkpoint_paths` must be non-empty if status == completed
- `artifact_dir` must follow pattern: `experiments/trial_{trial_id}/`

**State Transitions**:
```
queued: Initial state when trial is scheduled
  ↓
preparing: Loading model, datasets, initializing checkpoint manager
  ↓
running: Training in progress, checkpoints being created
  ↓
finishing: Final evaluation, artifact cleanup in progress
  ↓
completed: Trial finished successfully, best checkpoint(s) retained
  or
failed: Trial encountered error, partial artifacts may be retained
```

---

### 2. Study

Represents a complete HPO study with multiple trials.

**Fields**:
- `study_id`: str (UUID) - Unique identifier for the study
- `study_name`: str - Human-readable study name
- `direction`: str - Optimization direction ("maximize" only, per A-008)
- `optimization_metric_name`: str - Metric to optimize across trials
- `num_trials`: int - Total number of trials to execute
- `trials_completed`: int - Number of completed trials
- `trials_failed`: int - Number of failed trials
- `best_trial_id`: Optional[str] - Trial ID with best optimization metric
- `best_value`: Optional[float] - Best optimization metric value across all trials
- `study_dir`: str - Study artifact directory (e.g., "experiments/study_{uuid}")
- `evaluation_report_path`: Optional[str] - Path to test set evaluation report JSON
- `optuna_storage`: str - Optuna storage backend URI (e.g., "sqlite:///study.db")
- `created_at`: datetime
- `completed_at`: Optional[datetime]

**Validation Rules**:
- `study_id` must be valid UUID v4
- `direction` must be "maximize" (per A-008)
- `num_trials` must be > 0
- `study_dir` must follow pattern: `experiments/study_{study_id}/`
- `best_trial_id` must reference a valid completed trial if set
- `evaluation_report_path` must exist if study is completed (large-scale studies only)

**Relationships**:
- Study → Trials (one-to-many): A study contains multiple trials
- Study → EvaluationReport (one-to-one): A completed study has one test evaluation report

---

### 3. Checkpoint

Represents a saved model state at a specific training step.

**Fields**:
- `checkpoint_id`: str (UUID) - Unique identifier
- `trial_id`: str (UUID) - Parent trial identifier
- `checkpoint_path`: str - Filesystem path to checkpoint file
- `epoch`: int - Training epoch when checkpoint was saved
- `step`: int - Global training step
- `metrics`: Dict[str, float] - Metric values at checkpoint time
- `optimization_metric_value`: float - Value of the optimization metric
- `is_best`: bool - Whether this is a best checkpoint by optimization metric
- `is_co_best`: bool - Whether this ties for best with other checkpoints
- `is_latest`: bool - Whether this is the most recent checkpoint
- `retained`: bool - Whether this checkpoint is retained by policy
- `integrity_hash`: str (SHA256) - Checksum for corruption detection
- `size_bytes`: int - File size in bytes
- `created_at`: datetime

**Validation Rules**:
- `checkpoint_path` must be absolute path
- `epoch` and `step` must be non-negative
- `optimization_metric_value` must be present in `metrics`
- `integrity_hash` must be valid SHA256 hex string (64 chars)
- If `is_best` or `is_co_best` is True, `retained` must be True
- If `is_latest` is True, `retained` must be True

**Retention Logic**:
- Best checkpoints (`is_best` or `is_co_best`) are always retained
- Latest checkpoint (`is_latest`) is retained for resume capability
- Other checkpoints are pruned based on retention policy
- Under aggressive pruning (<10% disk), only single best checkpoint may be retained

---

### 4. RetentionPolicy

Configuration for checkpoint retention strategy.

**Fields**:
- `keep_last_n`: int - Number of most recent checkpoints to retain (default: 1)
- `keep_best_k`: int - Number of best checkpoints to retain (default: 1)
- `keep_best_k_max`: int - Maximum best checkpoints (cap for co-best, default: 2)
- `max_total_size_gb`: float - Maximum total checkpoint storage in GB (default: 10)
- `min_interval_epochs`: int - Minimum epochs between checkpoints (default: 1)
- `disk_space_threshold_percent`: float - Disk space threshold for aggressive pruning (default: 10.0)
- `pruning_strategy`: str - Pruning strategy ("tiered" or "aggressive")

**Validation Rules**:
- `keep_last_n` must be >= 0
- `keep_best_k` must be >= 1
- `keep_best_k_max` must be >= `keep_best_k`
- `max_total_size_gb` must be > 0
- `min_interval_epochs` must be >= 1
- `disk_space_threshold_percent` must be in range (0, 100)
- `pruning_strategy` must be "tiered" or "aggressive"

**Pruning Strategies**:
- **Tiered** (default): Retain both `keep_last_n` and `keep_best_k` checkpoints
- **Aggressive** (triggered at <10% disk): Reduce to single best checkpoint across entire study

---

### 5. EvaluationReport

JSON report for test set evaluation results.

**Fields**:
- `study_id`: str (UUID) - Parent study identifier
- `best_trial_id`: str (UUID) - Trial with best validation performance
- `test_metrics`: Dict[str, float] - Test set metric values
  - For criteria matcher: accuracy, precision, recall, f1, auc
  - For evidence binder: exact_match, has_answer, char_f1, char_precision, char_recall
- `config`: Dict[str, Any] - Complete resolved configuration of best trial
- `checkpoint_references`: List[str] - Paths to best checkpoint(s) (multiple if co-best)
- `optimization_metric_name`: str - Metric name that was optimized
- `optimization_metric_value`: float - Validation metric value of best trial
- `dataset_info`: DatasetInfo - Dataset metadata
- `evaluated_at`: datetime - Timestamp of test evaluation
- `evaluation_duration_seconds`: float - Time taken for evaluation

**Validation Rules**:
- `study_id` and `best_trial_id` must be valid UUIDs
- `test_metrics` must contain all required metrics for both agents
- `checkpoint_references` must be non-empty
- `config` must include: model_id, dataset_id, seeds, all hyperparameters
- `dataset_info.revision` must be set if dataset is pinned

**File Location**:
- Path pattern: `experiments/study_{study_id}/evaluation_report.json`
- Schema validated at creation time

---

### 6. DatasetInfo

Metadata for Hugging Face dataset configuration.

**Fields**:
- `dataset_id`: str - Hugging Face dataset identifier (default: "irlab-udc/redsm5")
- `revision`: str - Git revision/tag/commit (default: "main")
- `splits`: Dict[str, str] - Split name mappings (e.g., {"train": "train", "validation": "validation", "test": "test"})
- `split_sizes`: Dict[str, int] - Number of examples per split
- `dataset_hash`: str - Resolved dataset hash for reproducibility
- `cache_dir`: str - Local cache directory

**Validation Rules**:
- `dataset_id` must be valid Hugging Face dataset identifier
- `splits` must contain keys: "train", "validation", "test"
- `split_sizes` must match actual dataset sizes
- `revision` must be resolvable Git reference

**Relationships**:
- Dataset → Trials (one-to-many): Same dataset used across all trials in a study

---

### 7. ModelSource

Metadata for Hugging Face model configuration.

**Fields**:
- `model_id`: str - Hugging Face model identifier (e.g., "mental-bert")
- `revision`: Optional[str] - Model revision/tag/commit
- `model_hash`: str - Resolved model hash for reproducibility
- `cache_dir`: str - Local cache directory
- `num_parameters`: int - Total model parameters
- `architecture`: str - Model architecture type (e.g., "BertForSequenceClassification")

**Validation Rules**:
- `model_id` must be valid Hugging Face model identifier
- `revision` must be resolvable if specified
- `num_parameters` must be > 0
- `architecture` must match expected dual-agent architecture

**Catalog**:
Initial validated models (per A-001):
- mental-bert
- psychbert
- clinicalbert
- bert-base-uncased
- roberta-base

Expandable to 30+ models after validation.

---

### 8. LogEvent

Structured log entry for training/HPO events.

**Fields**:
- `timestamp`: datetime (ISO 8601)
- `severity`: str (DEBUG | INFO | WARNING | ERROR | CRITICAL)
- `message`: str - Human-readable message
- `component`: str - Source component (e.g., "checkpoint_manager", "trial_manager")
- `context`: Dict[str, Any] - Structured context fields
  - `trial_id`: Optional[str]
  - `step`: Optional[int]
  - `epoch`: Optional[int]
  - Additional component-specific fields
- `sanitized`: bool - Whether log was sanitized for secrets

**Validation Rules**:
- `severity` must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `message` must be non-empty
- `context` must not contain sensitive data (tokens, passwords)
- All log entries written to both JSON log file and stdout

**Log Sanitization**:
Regex patterns for secret masking (per FR-032):
- Hugging Face tokens: `hf_[A-Za-z0-9]{20,}` → `hf_***REDACTED***`
- API keys: `[A-Za-z0-9_-]{32,}` → `***REDACTED***`
- Bearer tokens: `Bearer [A-Za-z0-9_-]+` → `Bearer ***REDACTED***`
- Passwords in URLs: `://[^:]+:([^@]+)@` → `://user:***@`
- Email addresses: `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}` → `***@***.***`

---

### 9. MetricsBuffer

Disk buffer for metrics during MLflow tracking outages.

**Fields**:
- `buffer_path`: str - Path to JSONL buffer file
- `buffer_size_bytes`: int - Current buffer size
- `entry_count`: int - Number of buffered entries
- `first_entry_timestamp`: Optional[datetime] - Timestamp of oldest entry
- `last_entry_timestamp`: Optional[datetime] - Timestamp of newest entry

**Buffer Entry Format** (JSONL):
```json
{
  "run_id": "mlflow_run_uuid",
  "key": "metric_name",
  "value": 0.95,
  "step": 100,
  "timestamp": "2025-01-14T12:00:00Z"
}
```

**Validation Rules**:
- `buffer_path` must have `.jsonl` extension
- `buffer_size_bytes` must be non-negative
- Warning emitted when `buffer_size_bytes` > 100 MB
- No hard limit on buffer size (training continues)

**Replay Logic**:
- Read buffer line-by-line (JSONL format)
- Retry each metric with exponential backoff (1s, 2s, 4s, 8s, 16s)
- Delete buffer file only after successful upload confirmation
- Preserve buffer file on disk until replay completes

---

### 10. DualAgentModel

The dual-agent architecture with shared encoder.

**Fields**:
- `encoder`: HuggingFaceEncoder - Shared transformer encoder
- `criteria_matcher_head`: nn.Linear - Classification head for criteria matching
- `evidence_binder_head`: nn.Linear - Span extraction head for evidence binding
- `num_criteria_classes`: int - Number of criteria categories
- `task_weights`: Dict[str, float] - Loss weights for multi-task learning
  - `criteria_matcher`: float (default: 1.0)
  - `evidence_binder`: float (default: 1.0)

**Architecture**:
```
Input (text) → Tokenizer → Encoder (BERT/RoBERTa) → Task-specific heads
                                      ↓
                           ┌──────────┴──────────┐
                           ↓                      ↓
                  Criteria Matcher Head   Evidence Binder Head
                  (Classification)         (Span Extraction)
                           ↓                      ↓
                  Criteria Logits         Start/End Logits
```

**Metrics**:
- **Criteria Matcher**: accuracy, precision, recall, f1, auc
- **Evidence Binder**: exact_match, has_answer, char_f1, char_precision, char_recall

**Training**:
- Joint training with weighted multi-task loss
- Shared encoder updated by gradients from both tasks
- Separate validation evaluation for each task
- Optimization metric can be from either task (user-specified)

---

## Entity Relationships

```
Study (1) ──────→ (N) Trial
  ↓
  └─→ (1) EvaluationReport

Trial (1) ──────→ (N) Checkpoint
  ↓
  ├─→ (1) DatasetInfo (shared across trials)
  ├─→ (1) ModelSource
  ├─→ (1) RetentionPolicy
  └─→ (N) LogEvent

Trial ──────→ MLflow Run (external system)
  ↓
  └─→ (1) MetricsBuffer (created on tracking outage)

DualAgentModel ──────→ ModelSource (Hugging Face)
  ↓
  └─→ Checkpoint (serialized state)
```

## Validation Invariants

1. **Trial uniqueness**: Each trial must have a unique `trial_id` within a study
2. **Checkpoint consistency**: Latest checkpoint path must match a checkpoint with `is_latest=True`
3. **Best checkpoint retention**: All checkpoints with `is_best=True` or `is_co_best=True` must have `retained=True`
4. **Study completion**: Study can only transition to completed if `trials_completed + trials_failed == num_trials`
5. **Evaluation report**: Large-scale studies (1000+ trials) must have evaluation report if completed
6. **Metric existence**: `optimization_metric_name` must exist in trial metrics
7. **Disk space invariant**: Sum of retained checkpoint sizes must not exceed `max_total_size_gb` except under aggressive pruning
8. **State transition validity**: Trial status transitions must follow defined state machine
9. **Checkpoint integrity**: All retained checkpoints must have valid `integrity_hash` that matches file contents
10. **Log sanitization**: All log entries must have sensitive data masked before writing

## Persistence

### Filesystem Structure
```
experiments/
├── study_{study_uuid}/
│   ├── evaluation_report.json    # EvaluationReport
│   ├── study_metadata.json        # Study config and state
│   └── optuna_study.db           # Optuna storage
│
└── trial_{trial_uuid}/
    ├── checkpoints/
    │   ├── epoch_001.pt          # Checkpoint files
    │   ├── epoch_002.pt
    │   └── best_checkpoint.pt
    ├── logs/
    │   ├── train.jsonl           # Structured JSON logs
    │   └── metrics_buffer.jsonl  # MetricsBuffer (if tracking outage)
    └── trial_metadata.json       # Trial config and state
```

### MLflow Tracking
- Metrics: Time series data (step, value, timestamp)
- Parameters: Trial hyperparameters and config
- Tags: Study ID, trial ID, optimization metric, model ID
- Artifacts: Evaluation reports, config files (not checkpoints)

### Optuna Storage
- SQLite database per study: `experiments/study_{uuid}/optuna_study.db`
- Stores trial history, parameter search space, best trials
- Used for HPO optimization logic (Optuna algorithms)

---

## Summary

This data model defines 10 core entities that capture the complete lifecycle of storage-optimized HPO training:

1. **Trial**: Individual HPO trial execution
2. **Study**: HPO study with multiple trials
3. **Checkpoint**: Saved model state with retention metadata
4. **RetentionPolicy**: Checkpoint retention configuration
5. **EvaluationReport**: Test set evaluation results
6. **DatasetInfo**: Hugging Face dataset metadata
7. **ModelSource**: Hugging Face model metadata
8. **LogEvent**: Structured logging entries
9. **MetricsBuffer**: Disk buffer for tracking outages
10. **DualAgentModel**: Multi-task model architecture

All entities include validation rules, state transitions, and relationships to ensure data integrity and compliance with constitutional principles (reproducibility, storage optimization, resume capability).
