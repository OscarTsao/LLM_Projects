# Data Model

**Feature**: Storage-Optimized Training & HPO Pipeline
**Date**: 2025-10-10
**Phase**: 1 (Design & Contracts)

## Overview

This document defines the data models and entity relationships for the storage-optimized multi-task NLP training pipeline. Models are organized by domain: HPO management, training artifacts, experiment tracking, and dataset structures.

---

## 1. HPO & Trial Management

### 1.1 TrialConfig

**Purpose**: Encapsulates all hyperparameters and configuration for a single trial.

**Note**: This schema merges Feature 002 (storage-optimized training) and Feature 001 (threshold tuning). Threshold fields are optional and will be populated when Feature 001 is integrated.

**Fields**:
```python
trial_id: UUID  # Unique trial identifier
timestamp: datetime  # Trial creation time

# Model Architecture
model_id: str  # Hugging Face model identifier (e.g., "mental/mental-bert-base-uncased")
input_format: Literal["binary_pairs", "multi_label"]  # Data formatting strategy

# Criteria Matching Head
criteria_head_type: Literal["linear", "mlp", "mlp_residual", "gated", "multi_sample_dropout"]
criteria_pooling: Literal["cls", "mean", "attention", "last_2_layer_mix"]
criteria_hidden_dim: int  # For MLP variants (range: 128-1024)
criteria_dropout: float  # Dropout rate (range: 0.1-0.5)

# Evidence Binding Head
evidence_head_type: Literal["start_end_linear", "start_end_mlp", "biaffine", "bio_crf", "sentence_reranker"]
evidence_dropout: float

# Multi-Task Coupling
task_coupling: Literal["independent", "coupled"]
coupling_method: Optional[Literal["concat", "add", "mean", "weighted_sum", "weighted_mean"]]  # If coupled
coupling_pooling: Optional[Literal["mean", "max", "attention"]]  # If coupled, pool before combination
pool_before_combination: Optional[bool]  # If coupled

# Loss Function
loss_function: Literal["bce", "weighted_bce", "focal", "adaptive_focal", "hybrid"]
hybrid_weight_alpha: Optional[float]  # If hybrid (range: 0.1-0.9)
focal_gamma: Optional[float]  # If focal or adaptive focal (range: 1.0-5.0)
label_smoothing: float  # Range: 0.0-0.2

# Data Augmentation
augmentation_methods: List[Literal["synonym", "insert", "swap", "back_translation", "char_perturb"]]  # Can be empty
augmentation_prob: float  # Per-example probability (range: 0.0-0.5)

# Activation Function
activation: Literal["gelu", "silu", "swish", "relu", "leakyrelu", "mish", "tanh"]

# Regularization
layer_wise_lr_decay: float  # Range: 0.8-0.99
differential_lr_ratio: float  # Encoder LR / Head LR (range: 0.1-1.0)
warmup_ratio: float  # Range: 0.0-0.2
adversarial_epsilon: float  # Perturbation magnitude (range: 0.0-0.01)
class_weights: bool  # Whether to apply class weighting

# Training Hyperparameters
learning_rate: float  # Range: 1e-6 to 1e-4 (log scale)
batch_size: int  # Choices: 8, 16, 32
accumulation_steps: int  # Gradient accumulation (choices: 1, 2, 4)
epochs: int  # Choices: 10, 20, 30
optimizer: Literal["adam", "adamw"]
weight_decay: float  # Range: 0.0-0.1

# Optimization Metric
optimization_metric: str  # e.g., "val_f1_macro", "val_accuracy"

# Checkpoint Retention Policy
keep_last_n: int  # Keep last N checkpoints (default: 1)
keep_best_k: int  # Keep best K checkpoints (default: 1)
keep_best_k_max: int  # Maximum best checkpoints to retain (default: 2, co-best ties may exceed)
max_checkpoint_size_gb: float  # Maximum total checkpoint disk usage per trial (default: 10GB)

# Random Seeds
seed: int  # For reproducibility

# Decision Thresholds (Feature 001 - Optional, populated during post-training calibration)
criteria_threshold_strategy: Optional[Literal["global", "per_class"]]  # Default: "per_class"
criteria_thresholds: Optional[List[float]]  # Length C (num criteria), each in [0.0, 1.0], default all 0.5
evidence_null_threshold: Optional[float]  # Probability threshold for "no evidence" (range: 0.0-1.0, default 0.5)
evidence_min_span_score: Optional[float]  # Minimum span score to emit (range: 0.0-1.0, default 0.0)
hpo_tune_thresholds: Optional[bool]  # Whether to tune thresholds (Feature 001, default false)

# UI Configuration (Feature 001 - Optional)
ui_progress: Optional[bool]  # Enable tqdm progress bars (default: true)
ui_stdout_level: Optional[Literal["INFO", "DEBUG", "WARNING"]]  # Stdout logging level (default: "INFO")
```

**Validation Rules**:
- `trial_id` must be unique across all trials in a study
- `model_id` must be one of the supported Hugging Face models (initially 5, expandable to 30+)
- If `task_coupling == "coupled"`, `coupling_method` and `coupling_pooling` must be specified
- If `loss_function == "hybrid"`, `hybrid_weight_alpha` must be specified
- If `loss_function` in ["focal", "adaptive_focal", "hybrid"], `focal_gamma` must be specified
- `keep_last_n >= 1` and `keep_best_k >= 1` (must retain at least one checkpoint)
- If `criteria_threshold_strategy == "per_class"`, `criteria_thresholds` must have length equal to number of criteria (9 for RedSM5)
- If `criteria_threshold_strategy == "global"`, `criteria_thresholds` must have length 1 (broadcast to all criteria)
- Threshold fields are optional; if not specified, defaults apply (all 0.5)

---

### 1.2 Trial

**Purpose**: Tracks execution state and outcomes of a single HPO trial.

**Fields**:
```python
trial_id: UUID  # Foreign key to TrialConfig
config: TrialConfig  # Embedded configuration
status: Literal["pending", "running", "completed", "failed", "pruned"]  # Execution status
start_time: datetime
end_time: Optional[datetime]

# Directory Structure
trial_dir: Path  # e.g., experiments/trial_<uuid>/
checkpoint_dir: Path  # trial_dir / "checkpoints"
log_dir: Path  # trial_dir / "logs"

# Metrics
best_checkpoint_ids: List[UUID]  # References to Checkpoint entities (can be multiple if co-best)
final_test_metrics: Optional[Dict[str, float]]  # Test set evaluation results
validation_history: List[Dict[str, Any]]  # Per-epoch validation metrics

# Storage Tracking
total_checkpoint_size_bytes: int  # Current disk usage by this trial's checkpoints
num_checkpoints_pruned: int  # Count of pruned checkpoints for observability

# Error Handling
error_message: Optional[str]  # If status == "failed", capture traceback
```

**State Transitions**:
```
pending → running → completed
pending → running → failed
pending → running → pruned (early stopped by Optuna)
```

**Relationships**:
- One `Trial` has one `TrialConfig`
- One `Trial` has many `Checkpoint` entities
- One `HPOStudy` produces one `EvaluationReport` (per-study, not per-trial)

---

## 2. Training Artifacts

### 2.1 Checkpoint

**Purpose**: Represents a saved model checkpoint during training.

**Fields**:
```python
checkpoint_id: UUID
trial_id: UUID  # Foreign key to Trial
epoch: int  # Epoch number when checkpoint was saved
step: int  # Global training step

# File Metadata
file_path: Path  # Absolute path to checkpoint .pt file
file_size_bytes: int
created_at: datetime

# Metrics Snapshot
metrics: Dict[str, float]  # Validation metrics at this checkpoint (e.g., {"val_f1": 0.8, "val_loss": 0.3})

# Retention Policy Flags
retained: bool  # True if checkpoint is currently retained (not pruned)
co_best: bool  # True if tied for maximum optimization metric with other checkpoints
is_last_n: bool  # True if in last N checkpoints
is_best_k: bool  # True if in best K checkpoints

# Optimization Metric Value
optimization_metric_value: float  # Value of the trial's optimization metric (for ranking)
```

**Validation Rules**:
- `epoch >= 0` and `step >= 0`
- `file_path` must exist on disk if `retained == True`
- If `co_best == True`, then `is_best_k == True`
- Atomic checkpoint saves: Write to temp file, then rename to `file_path`

**Lifecycle**:
1. Created after each epoch (minimum interval: 1 epoch)
2. Evaluated on validation set, `metrics` populated
3. Retention policy applied:
   - Mark as `is_last_n` if in last N
   - Mark as `is_best_k` if in top K by `optimization_metric_value`
   - Mark as `co_best` if tied with other best checkpoints
4. If not retained by any policy, `retained = False` and file deleted (pruned)
5. Never prune if disk space sufficient; only prune when threshold violated

---

### 2.2 CheckpointMetadata

**Purpose**: Minimal metadata stored inside each checkpoint .pt file for self-contained resume.

**Fields** (serialized in checkpoint dict):
```python
checkpoint_id: UUID
trial_id: UUID
epoch: int
step: int
model_state_dict: Dict[str, Tensor]  # PyTorch model weights
optimizer_state_dict: Dict  # Optimizer state
scheduler_state_dict: Optional[Dict]  # LR scheduler state (if used)
rng_state: Dict  # RNG states (Python, NumPy, PyTorch CPU/CUDA) for reproducibility
config: TrialConfig  # Full configuration for self-contained resume
metrics: Dict[str, float]  # Validation metrics at this checkpoint
created_at: datetime
```

**Usage**:
- Loaded via `torch.load(file_path)` to resume training
- `rng_state` ensures deterministic resume

---

## 3. Experiment Tracking

### 3.1 ExperimentRun

**Purpose**: MLflow run metadata (maps to MLflow's internal schema but abstracted here).

**Fields**:
```python
run_id: str  # MLflow run ID (unique)
experiment_id: str  # MLflow experiment ID (groups related trials)
trial_id: UUID  # Foreign key to Trial

# Logged Data
params: Dict[str, Any]  # TrialConfig serialized as MLflow params
metrics: Dict[str, List[Tuple[int, float]]]  # Metric name → [(step, value), ...]
tags: Dict[str, str]  # Metadata tags (e.g., {"model_id": "bert-base", "status": "completed"})

# Artifacts
artifact_uri: str  # MLflow artifact storage location (e.g., file://experiments/trial_<uuid>/artifacts)
```

**Relationships**:
- One `Trial` has one `ExperimentRun`
- `ExperimentRun` stores references to logged artifacts (checkpoints, plots, JSON reports)

---

### 3.2 MetricsBuffer

**Purpose**: Disk-backed buffer for MLflow metrics when tracking backend is unavailable.

**Fields**:
```python
buffer_id: UUID
trial_id: UUID
buffer_file_path: Path  # JSONL file storing buffered metrics
buffer_size_bytes: int  # Current file size
created_at: datetime
last_flush_attempt: Optional[datetime]
flush_status: Literal["pending", "flushing", "flushed", "failed"]
```

**Buffer File Format** (JSONL):
```json
{"step": 100, "epoch": 1, "val_f1": 0.75, "val_loss": 0.42, "timestamp": "2025-10-10T12:00:00Z"}
{"step": 200, "epoch": 2, "val_f1": 0.78, "val_loss": 0.38, "timestamp": "2025-10-10T12:05:00Z"}
```

**Behavior**:
- If `buffer_size_bytes > 100MB`, emit WARNING (FR-017)
- Periodic background task attempts flush to MLflow
- On successful flush, delete buffer file and update `flush_status = "flushed"`

---

## 4. Dataset Models

### 4.1 RedSM5Post

**Purpose**: Represents a single mental health post from the RedSM5 dataset.

**Fields** (parsed from `Data/redsm5/redsm5_posts.csv`):
```python
post_id: str  # Unique post identifier
post_text: str  # Full post text content
author_id: Optional[str]  # Anonymized author ID (if available)
timestamp: Optional[datetime]  # Post creation time (if available)
metadata: Dict[str, Any]  # Additional metadata fields from CSV
```

**Validation Rules**:
- `post_id` must be unique
- `post_text` must be non-empty

---

### 4.2 RedSM5Annotation

**Purpose**: Represents criteria labels and evidence annotations for a post.

**Fields** (parsed from `Data/redsm5/redsm5_annotations.csv`):
```python
annotation_id: UUID
post_id: str  # Foreign key to RedSM5Post
criterion_id: int  # Criterion index (0-8 for 9 criteria)
criterion_name: str  # Human-readable criterion name (e.g., "depression_symptoms")
label: int  # Binary label: 0 (not matched) or 1 (matched)
evidence_sentence: Optional[str]  # Evidence text from "sentence" column (if label == 1)
evidence_char_start: Optional[int]  # Character offset in post_text where evidence starts
evidence_char_end: Optional[int]  # Character offset where evidence ends
```

**Validation Rules**:
- If `label == 1`, `evidence_sentence` must be non-null
- `evidence_char_start` and `evidence_char_end` must be valid offsets in parent post's `post_text`
- `criterion_id` must be in range [0, 8]

**Relationships**:
- Many `RedSM5Annotation` belong to one `RedSM5Post` (9 annotations per post, one per criterion)

---

### 4.3 ProcessedExample

**Purpose**: A single training example after preprocessing and augmentation.

**Fields**:
```python
example_id: UUID
post_id: str  # Foreign key to RedSM5Post
split: Literal["train", "val", "test"]  # Dataset split
input_format: Literal["binary_pairs", "multi_label"]

# Input Text
input_text: str  # Formatted input (e.g., "[CLS] post [SEP] criterion [SEP]")
tokenized_input_ids: List[int]  # Tokenized input IDs
attention_mask: List[int]
token_type_ids: Optional[List[int]]  # For models supporting segment IDs

# Labels
criteria_labels: Union[int, List[int]]  # Binary (0/1) or multi-label ([0,1,1,0,1,0,1,0,0])
evidence_start: Optional[int]  # Token index for span start (if evidence binding enabled)
evidence_end: Optional[int]  # Token index for span end
evidence_bio_tags: Optional[List[str]]  # BIO tags for sequence labeling (if using BIO+CRF head)

# Augmentation Metadata
is_augmented: bool  # True if augmentation applied
augmentation_method: Optional[str]  # Method used (if augmented)
original_evidence_sentence: Optional[str]  # Original evidence before augmentation
augmented_evidence_sentence: Optional[str]  # Augmented evidence
```

**Validation Rules**:
- If `input_format == "binary_pairs"`, `criteria_labels` must be int (single label)
- If `input_format == "multi_label"`, `criteria_labels` must be List[int] of length 9
- `len(tokenized_input_ids) == len(attention_mask) == len(token_type_ids or [])`
- If `is_augmented == True`, both `original_evidence_sentence` and `augmented_evidence_sentence` must be non-null

---

### 4.4 DatasetSplit

**Purpose**: Metadata about train/val/test splits.

**Fields**:
```python
split_id: UUID
split_name: Literal["train", "val", "test"]
post_ids: List[str]  # Post IDs in this split (post-level split to avoid leakage)
num_examples: int  # Total examples (may differ from len(post_ids) if using binary pairs format)
split_ratio: float  # Proportion of total data (e.g., 0.70 for train)
created_at: datetime
cache_path: Path  # Path to cached processed examples (e.g., Data/redsm5/processed/train.pkl)
```

**Validation Rules**:
- Sum of `split_ratio` across train/val/test should equal 1.0
- `post_ids` must be disjoint across splits (no post appears in both train and val)
- Cache file at `cache_path` must contain list of `ProcessedExample` matching `split_name`

---

## 5. Evaluation & Reporting

### 5.1 EvaluationReport (Per-Study)

**Purpose**: Per-study JSON report with test set evaluation results for the best model from the entire HPO study; saved to disk and logged to MLflow.

**Evaluation Timing**: Generated **after all trials in the HPO study complete** (hybrid approach: trials evaluate on validation set during training, test set evaluation occurs once per study)**.

**Fields**:
```python
report_id: UUID
study_id: UUID  # Identifier for the HPO study
best_trial_id: UUID  # Trial that produced the best model
generated_at: datetime

# Configuration Snapshot
config: TrialConfig  # Full trial config for the best trial (for reproducibility)

# Optimization Metric
optimization_metric_name: str  # e.g., "val_f1_macro"
best_validation_score: float  # Best validation score achieved across all trials

# Checkpoint References
evaluated_checkpoints: List[Dict]  # List of checkpoint metadata from best trial
# Example: [{"checkpoint_id": uuid, "path": "...", "epoch": 10, "val_metric": 0.85, "co_best": False}]

# Decision Thresholds (Feature 001 - Optional, populated if threshold tuning enabled)
decision_thresholds: Optional[Dict[str, Any]]
# Schema:
# {
#   "criteria": List[float],  # Length C (num criteria), per-class thresholds
#   "evidence": {
#     "null_threshold": float,
#     "min_span_score": float
#   }
# }

# Test Metrics
test_metrics: Dict[str, Any]
# Schema:
# {
#   "criteria_matching": {
#     "accuracy": float,
#     "f1_macro": float,
#     "f1_micro": float,
#     "precision_macro": float,
#     "recall_macro": float,
#     "macro_pr_auc": float,  # Feature 001: Macro-averaged PR AUC
#     "per_criterion": [
#       {
#         "id": str,  # Criterion identifier
#         "f1": float,
#         "precision": float,
#         "recall": float,
#         "pr_auc": float,  # Feature 001: Per-criterion PR AUC
#         "confusion_matrix": [[tn, fp], [fn, tp]],  # Feature 001: 2x2 confusion matrix
#         "support": int
#       },
#       ...
#     ]
#   },
#   "evidence_binding": {
#     "exact_match": float,  # Proportion of exact span matches
#     "f1": float,  # Token-level F1 (span overlap)
#     "char_f1": float,  # Character-level F1
#     "null_span_accuracy": float  # Accuracy of predicting no evidence
#   }
# }

# File Paths
report_file_path: Path  # Path to JSON file (e.g., experiments/study_<uuid>/evaluation_report.json)
```

**Validation Rules**:
- `evaluated_checkpoints` must contain at least one checkpoint (the best from best trial)
- If multiple co-best checkpoints in best trial, all must be included
- `test_metrics` must conform to schema above
- `decision_thresholds` is optional; populated only if Feature 001 threshold tuning is enabled
- Report saved as JSON file at `report_file_path` for human inspection and automated analysis
- **Per-study evaluation**: Only one report generated per HPO study (after all trials complete), not per trial

---

### 5.2 RetentionPolicy

**Purpose**: Configuration for checkpoint retention and pruning behavior.

**Fields**:
```python
policy_id: UUID
trial_id: UUID  # Foreign key to Trial (policy is trial-specific)

# Retention Parameters
keep_last_n: int  # Keep last N checkpoints (minimum: 1)
keep_best_k: int  # Keep best K checkpoints by optimization metric (minimum: 1)
keep_best_k_max: int  # Hard cap on the number of best checkpoints to keep (default: 2)
max_total_size_gb: float  # Maximum total checkpoint disk usage for this trial
min_interval_epochs: int  # Minimum epochs between checkpoints (default: 1)

# Disk Space Monitoring
disk_space_threshold_percent: float  # Trigger pruning when available disk < this % (default: 10.0)
current_disk_usage_percent: float  # Most recent disk usage check (updated periodically)
last_pruning_timestamp: Optional[datetime]

# Pruning Strategy
pruning_strategy: Literal["oldest_first", "smallest_first"]  # How to select checkpoints to prune
```

**Validation Rules**:
- `keep_last_n >= 1` and `keep_best_k >= 1`
- `keep_best_k_max >= keep_best_k`
- `min_interval_epochs >= 1` (enforced from FR-022)
- `disk_space_threshold_percent` should be in range (0, 100), typical value 10.0

**Behavior**:
- After each checkpoint save, evaluate:
  1. Total checkpoint size exceeds `max_total_size_gb`?
  2. Available disk space < `disk_space_threshold_percent`?
- If either true, trigger pruning:
  - Compute set of "protected" checkpoints: union of last N, best K, all co-best
  - Prune oldest (or smallest, per `pruning_strategy`) non-protected checkpoints
  - Repeat until constraints satisfied or only protected checkpoints remain
- If constraints still violated after pruning all non-protected, ERROR and abort trial (FR-014)

---

## 6. Logging & Observability

### 6.1 LogEvent

**Purpose**: Structured log event for dual logging (JSON + stdout).

**Fields**:
```python
event_id: UUID
timestamp: datetime
level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
message: str  # Human-readable message
component: str  # Source component (e.g., "checkpoint_manager", "trainer", "hpo_executor")

# Context
trial_id: Optional[UUID]
epoch: Optional[int]
step: Optional[int]
extra: Dict[str, Any]  # Additional structured context (e.g., {"disk_usage_gb": 450, "threshold_gb": 500})
```

**Output Formats**:
1. **JSON** (written to `trial_dir/logs/training.jsonl`):
   ```json
   {"event_id": "...", "timestamp": "2025-10-10T12:00:00Z", "level": "WARNING", "message": "Metrics buffer exceeds 100MB", "component": "metrics_buffer", "trial_id": "...", "extra": {"buffer_size_mb": 105}}
   ```

2. **Human-Readable Stdout**:
   ```
   [2025-10-10 12:00:00] WARNING [metrics_buffer] Metrics buffer exceeds 100MB (buffer_size_mb=105)
   ```

**Validation Rules**:
- `level` must be one of standard logging levels
- `timestamp` must be in ISO 8601 format
- If `trial_id` is null, event is global (not trial-specific, e.g., HPO orchestrator events)

---

## 7. Configuration Models

### 7.1 HPOStudyConfig

**Purpose**: Global configuration for an HPO study (collection of trials).

**Fields**:
```python
study_id: UUID
study_name: str  # Human-readable study name (e.g., "mental_health_hpo_2025_10")
created_at: datetime

# Optuna Configuration
sampler: Literal["tpe", "random", "grid"]  # TPE (Tree-structured Parzen Estimator) recommended
pruner: Optional[Literal["median", "percentile", "hyperband"]]  # Early stopping pruner
n_trials: int  # Total number of trials to run (e.g., 1000)
timeout_per_trial_hours: Optional[float]  # Max time per trial (optional)

# Search Space
search_space: Dict[str, Any]  # Optuna search space definition (maps to TrialConfig fields)
# Example:
# {
#   "model_id": {"type": "categorical", "choices": ["bert-base-uncased", "mental-bert", ...]},
#   "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-4, "log": True},
#   "batch_size": {"type": "categorical", "choices": [8, 16, 32]},
#   ...
# }

# Optimization Objective
optimization_metric: str  # Metric to optimize (e.g., "val_f1_macro")
optimization_direction: Literal["maximize", "minimize"]  # Always "maximize" per spec A-008

# Storage
study_db_path: Path  # Optuna study database (SQLite) for persistence
mlflow_experiment_name: str  # MLflow experiment name for grouping trials
```

**Validation Rules**:
- `n_trials > 0`
- `search_space` must define all required TrialConfig fields
- `optimization_direction` must be "maximize" (per spec assumption A-008)

---

## 8. Entity Relationships Summary

```
HPOStudyConfig (1) ──< (many) Trial
Trial (1) ──< (many) Checkpoint
Trial (1) ─── (1) TrialConfig
Trial (1) ─── (1) ExperimentRun
Trial (1) ─── (1) RetentionPolicy
Trial (1) ──< (many) LogEvent

HPOStudy (1) ─── (1) EvaluationReport  # Per-study evaluation (hybrid approach)

RedSM5Post (1) ──< (many) RedSM5Annotation
RedSM5Post (1) ──< (many) ProcessedExample
DatasetSplit (1) ──< (many) ProcessedExample

MetricsBuffer (1) ─── (1) Trial
```

---

## 9. Data Persistence Layers

### 9.1 File System
- **Checkpoints**: `experiments/trial_<uuid>/checkpoints/*.pt` (PyTorch state dicts)
- **Logs**: `experiments/trial_<uuid>/logs/training.jsonl` (structured JSON logs)
- **Reports**: `experiments/study_<uuid>/evaluation_report.json` (JSON evaluation reports, per-study)
- **Processed Data**: `Data/redsm5/processed/{train,val,test}.pkl` (cached PyTorch datasets)
- **Metrics Buffer**: `experiments/trial_<uuid>/logs/metrics_buffer.jsonl` (temporary buffer)

### 9.2 Databases
- **Optuna Study**: SQLite database at `study_db_path` (stores trial history, hyperparameters, results)
- **MLflow Tracking**: SQLite database at `experiments/mlflow_db/mlflow.db` (stores runs, metrics, params, artifacts)

### 9.3 In-Memory Caches
- **Hugging Face Models**: Cached at `~/.cache/huggingface/` (auto-managed by transformers library)
- **Tokenized Datasets**: Cached in memory during training (PyTorch DataLoader prefetch)

---

## 10. Data Validation & Integrity

### Validation Checkpoints

1. **Config Validation** (before trial start):
   - Validate `TrialConfig` against schema
   - Check model_id exists on Hugging Face
   - Verify conditional dependencies (e.g., if coupled, coupling_method specified)

2. **Dataset Validation** (after preprocessing):
   - Verify train/val/test splits are disjoint (no post_id appears in multiple splits)
   - Check label distribution (warn if extreme imbalance, e.g., <1% positive class)
   - Validate augmented text preserves non-evidence regions

3. **Checkpoint Validation** (after save):
   - Verify checkpoint file exists and is readable
   - Confirm checkpoint contains all required keys (model_state_dict, optimizer_state_dict, rng_state)
   - Check file size matches expected model size (catch truncated writes)

4. **Metrics Validation** (during logging):
   - Ensure metrics are finite (catch NaN/Inf from unstable training)
   - Verify metric keys match expected names (e.g., "val_f1_macro")

5. **Report Validation** (after generation):
   - Validate JSON schema against contract
   - Check all required fields present (config, test_metrics, checkpoint references)

### Integrity Constraints

- **Atomicity**: Checkpoint saves use atomic write (temp file → rename) to prevent partial writes
- **Idempotency**: Re-running a trial with same `trial_id` resumes from latest checkpoint (doesn't restart)
- **Consistency**: Retention policy never deletes last or best checkpoints; pruning respects invariants
- **Durability**: All critical state (checkpoints, configs, metrics) persisted to disk before trial completion

---

## Summary

This data model provides:
- **Comprehensive HPO tracking**: TrialConfig, Trial, Checkpoint entities capture full experiment lifecycle
- **Multi-task NLP support**: ProcessedExample, RedSM5Annotation models handle dual tasks (criteria matching + evidence binding)
- **Storage optimization**: RetentionPolicy, CheckpointMetadata enable intelligent pruning
- **Fault tolerance**: MetricsBuffer, checkpoint resume support robust long-running experiments
- **Observability**: LogEvent, EvaluationReport provide dual JSON + human-readable output

All entities are validated at creation and state transitions are well-defined to ensure data integrity across 1000-trial HPO workloads.
