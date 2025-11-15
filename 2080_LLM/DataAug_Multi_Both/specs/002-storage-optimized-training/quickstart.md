# Quickstart Guide: Storage-Optimized Training & HPO Pipeline

**Feature**: Storage-Optimized Training & HPO Pipeline for Mental Health Text Classification
**Date**: 2025-10-10
**Audience**: ML Engineers and Researchers

## Overview

This guide walks through setting up and running the storage-optimized multi-task NLP training pipeline with hyperparameter optimization (HPO) on the RedSM5 mental health dataset.

**Key Capabilities**:
- Dual-task learning: Criteria matching + evidence span extraction
- 30+ pretrained transformer models from Hugging Face
- Extensive HPO search space (model architecture, loss functions, augmentation, regularization)
- Intelligent checkpoint retention (60%+ storage reduction)
- Sequential trial execution with MLflow tracking
- Containerized environment for portability

---

## Prerequisites

### System Requirements
- Linux server with Docker installed
- (Optional) NVIDIA GPU with CUDA 11.8+ for GPU acceleration
- Minimum 50GB free disk space (for models, checkpoints, and data)
- 16GB+ RAM recommended

### Authentication
- **Hugging Face**: Pre-login via `huggingface-cli login` to access models
  ```bash
  pip install huggingface_hub
  huggingface-cli login
  # Follow prompts to enter your HF token
  ```

---

## Step 1: Environment Setup

### Option A: Docker (Recommended)

1. **Build the Docker image**:
   ```bash
   cd /experiment/YuNing/DataAug_Multi_Both
   docker build -t mental-health-hpo:latest -f docker/Dockerfile .
   ```

2. **Run the container**:
   ```bash
   docker run --gpus all \  # Omit --gpus if CPU-only
     -v $(pwd)/Data:/workspace/Data:ro \
     -v $(pwd)/experiments:/workspace/experiments \
     -v $HOME/.cache/huggingface:/workspace/.cache/huggingface \
     -it mental-health-hpo:latest bash
   ```

### Option B: Local Python Environment

1. **Create conda environment**:
   ```bash
   conda create -n mental-hpo python=3.10
   conda activate mental-hpo
   ```

2. **Install dependencies**:
   ```bash
   pip install -r docker/requirements.txt
   ```

---

## Step 2: Data Configuration (Hugging Face Datasets)

Data is loaded strictly from Hugging Face Datasets with explicit splits (constitution requirement).

1. Set the dataset ID in Hydra config `configs/data/dataset.yaml`:
   ```yaml
   dataset:
     id: irlab-udc/redsm5
     revision: null      # optional: pin to a specific revision/tag
     splits:
       train: train
       validation: validation
       test: test
     streaming: false
     cache_dir: ${oc.env:HF_HOME,~/.cache/huggingface}
   ```

2. Ensure you are logged into Hugging Face and have access to the dataset. No local CSVs are required. The datasets library will download and cache data under the Hugging Face cache directory.

---

## Step 3: Configure HPO Study

Create an HPO configuration file `config/hpo_study.yaml`:

```yaml
# HPO Study Configuration
study_name: mental_health_hpo_2025_10
n_trials: 100  # Start with 100 trials, scale to 1000 for full search

# Optuna Settings
sampler: tpe  # Tree-structured Parzen Estimator
pruner: median  # Early stop unpromising trials
timeout_per_trial_hours: 4.0  # Max 4 hours per trial

# Optimization Objective
optimization_metric: val_f1_macro
optimization_direction: maximize

# MLflow Tracking
mlflow_experiment_name: mental_health_multi_task

# Search Space (subset shown; see contracts/config_schema.yaml for full space)
search_space:
  model_id:
    type: categorical
    choices:
      - mental/mental-bert-base-uncased
      - mnaylor/psychbert-cased
      - medicalai/ClinicalBERT
      - google-bert/bert-base-uncased
      # Add more models from the 30+ catalog as needed

  input_format:
    type: categorical
    choices: [binary_pairs, multi_label]

  criteria_head_type:
    type: categorical
    choices: [linear, mlp, mlp_residual, gated]

  criteria_pooling:
    type: categorical
    choices: [cls, mean, attention, last_2_layer_mix]

  criteria_hidden_dim:
    type: int
    low: 128
    high: 1024
    step: 128

  evidence_head_type:
    type: categorical
    choices: [start_end_linear, start_end_mlp, biaffine, bio_crf]

  task_coupling:
    type: categorical
    choices: [independent, coupled]

  loss_function:
    type: categorical
    choices: [bce, weighted_bce, focal, hybrid]

  learning_rate:
    type: float
    low: 0.000001  # 1e-6
    high: 0.0001   # 1e-4
    log: true

  batch_size:
    type: categorical
    choices: [8, 16, 32]

  epochs:
    type: categorical
    choices: [10, 20, 30]

  # Add augmentation, regularization, etc. as needed

# Checkpoint Retention Policy
keep_last_n: 3
keep_best_k: 5
max_checkpoint_size_gb: 50.0
```

---

## Step 4: Launch HPO

### Single Command Execution

```bash
python src/cli/train.py \
  --config config/hpo_study.yaml \
  --mode hpo \
  --study-db experiments/hpo_study.db \
  --mlflow-uri file://experiments/mlflow_db
```

**Flags**:
- `--config`: Path to HPO study configuration
- `--mode hpo`: Run hyperparameter optimization (vs single trial)
- `--study-db`: Optuna study database path (for resume)
- `--mlflow-uri`: MLflow tracking URI

### Monitoring

**Terminal Output**:
- Real-time progress bars (tqdm) for epochs and trials
- Epoch summaries with metrics tables
- Warnings/errors logged to stdout

**MLflow UI** (optional):
```bash
# In a separate terminal
mlflow ui --backend-store-uri file://experiments/mlflow_db --port 5000
# Open browser: http://localhost:5000
```

**Log Files**:
- Structured JSON logs: `experiments/trial_<uuid>/logs/training.jsonl`
- Human-readable trial logs: `experiments/trial_<uuid>/logs/stdout.log`

---

## Step 5: Resume Interrupted HPO

If HPO is interrupted (crash, manual stop), resume from the last completed trial:

```bash
python src/cli/train.py \
  --config config/hpo_study.yaml \
  --mode hpo \
  --study-db experiments/hpo_study.db \  # Same DB path
  --resume  # Add resume flag
```

**Behavior**:
- Loads existing Optuna study from `--study-db`
- Skips completed trials
- Continues from trial `n+1`

---

## Step 6: Analyze Results

### Best Trial Query

**Using MLflow UI**:
1. Navigate to experiment `mental_health_multi_task`
2. Sort runs by `val_f1_macro` descending
3. Inspect best trial's hyperparameters and metrics

**Using Python**:
```python
import mlflow
import pandas as pd

mlflow.set_tracking_uri("file://experiments/mlflow_db")
experiment = mlflow.get_experiment_by_name("mental_health_multi_task")

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.val_f1_macro DESC"],
    max_results=10
)

print("Top 10 Trials:")
print(runs[["params.model_id", "metrics.val_f1_macro", "metrics.test_f1_macro"]])
```

### Trial Reports

Each trial generates a JSON evaluation report:

```bash
# Example: View trial report
cat experiments/trial_<uuid>/evaluation_report.json | jq .
```

**Report Contents**:
- Full configuration (hyperparameters)
- Test set metrics (criteria matching + evidence binding)
- Checkpoint references (path, epoch, validation score)

### Evaluate a Trial via CLI

Run the evaluator for a specific trial (per-trial evaluation):

```bash
python -m src.cli.evaluate --trial-id <trial_uuid>
```

This command generates or regenerates `experiments/trial_<uuid>/evaluation_report.json` validated against the per-trial schema.

### Aggregate Analysis

```python
import json
import glob

# Load all trial reports
reports = []
for report_file in glob.glob("experiments/trial_*/evaluation_report.json"):
    with open(report_file) as f:
        reports.append(json.load(f))

# Create DataFrame for analysis
df = pd.DataFrame([
    {
        "trial_id": r["trial_id"],
        "model_id": r["config"]["model_id"],
        "input_format": r["config"]["input_format"],
        "test_f1_macro": r["test_metrics"]["criteria_matching"]["f1_macro"],
        "evidence_f1": r["test_metrics"]["evidence_binding"]["f1"]
    }
    for r in reports
])

# Analyze: Which model architectures performed best?
print(df.groupby("model_id")["test_f1_macro"].agg(["mean", "std", "max"]))
```

---

## Step 7: Deploy Best Model

### Load Best Checkpoint

```python
import torch
from src.models.multi_task import MultiTaskModel

# Load best trial config
best_trial_id = "..."  # From MLflow/Optuna query
report_path = f"experiments/trial_{best_trial_id}/evaluation_report.json"
with open(report_path) as f:
    report = json.load(f)

best_checkpoint_path = report["evaluated_checkpoints"][0]["path"]

# Load model
checkpoint = torch.load(best_checkpoint_path)
config = checkpoint["config"]
model = MultiTaskModel.from_config(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Run inference
inputs = tokenizer(["Post text here..."], return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    criteria_predictions = outputs["criteria_logits"].sigmoid()  # Binary probabilities
    evidence_spans = outputs["evidence_start"], outputs["evidence_end"]

print("Criteria Matching Predictions:", criteria_predictions)
print("Evidence Span:", evidence_spans)
```

### Production Deployment

For production use, export the best model checkpoint and serve via:
- **TorchServe**: PyTorch model serving
- **Triton Inference Server**: NVIDIA's inference server
- **FastAPI**: Custom REST API wrapper

---

## Common Workflows

### Run a Single Trial (No HPO)

Test a specific configuration without HPO:

```bash
python src/cli/train.py \
  --config config/single_trial.yaml \
  --mode single \
  --output-dir experiments/trial_debug
```

**`config/single_trial.yaml`**:
```yaml
trial_id: debug_trial_001
model_id: mental/mental-bert-base-uncased
input_format: binary_pairs
criteria_head_type: mlp
learning_rate: 0.00002
batch_size: 16
epochs: 10
# ... other fixed hyperparameters
```

### Dry-Run (Validation Only)

Validate configuration and data loading without training:

```bash
python src/cli/train.py \
  --config config/hpo_study.yaml \
  --dry-run
```

### Export HPO Results to CSV

```bash
# Export Optuna study results
python -c "
import optuna
study = optuna.load_study(
    study_name='mental_health_hpo_2025_10',
    storage='sqlite:///experiments/hpo_study.db'
)
df = study.trials_dataframe()
df.to_csv('hpo_results.csv', index=False)
"
```

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms**: `CUDA out of memory` or system freeze

**Solutions**:
1. Reduce `batch_size` in config (try 8)
2. Enable gradient accumulation: Set `accumulation_steps: 4` to simulate larger batches
3. Enable gradient checkpointing:
   ```yaml
   gradient_checkpointing: true  # Trade compute for memory
   ```
4. Use smaller model variants (e.g., `bert-base` instead of `bert-large`)

### Disk Space Exhaustion

**Symptoms**: Trial fails with `ERROR: Cannot write checkpoint, disk space < 10%`

**Solutions**:
1. Check retention policy settings:
   ```yaml
   keep_last_n: 2  # Reduce from 3
   keep_best_k: 3  # Reduce from 5
   max_checkpoint_size_gb: 30.0  # Lower limit
   ```
2. Manually delete old trial directories: `rm -rf experiments/trial_<old_uuid>`
3. Prune Optuna study: Delete failed/pruned trials from DB

### MLflow Tracking Unavailable

**Symptoms**: `WARNING: Metrics buffer exceeds 100MB`

**Solutions**:
1. Check MLflow DB file permissions: `ls -l experiments/mlflow_db/mlflow.db`
2. Flush buffered metrics manually:
   ```python
   from src.hpo.metrics_buffer import MetricsBuffer
   buffer = MetricsBuffer.load("experiments/trial_<uuid>/logs/metrics_buffer.jsonl")
   buffer.flush_to_mlflow(mlflow_run_id="...")
   ```
3. Restart MLflow server if using remote tracking

### Slow Data Loading

**Symptoms**: Epoch start delay >1 minute

**Solutions**:
1. Increase DataLoader workers: `num_workers: 8` (in config)
2. Enable pin memory: `pin_memory: true`
3. Preload augmented dataset cache:
   ```python
   # Warm up Hugging Face datasets cache by iterating once
   from datasets import load_dataset
   ds = load_dataset("irlab-udc/redsm5")
   _ = [x for x in ds["train"].select(range(100))]
   ```

---

## Advanced Configuration

### Custom Search Space

To add a new hyperparameter to the HPO search space:

1. Update `config/hpo_study.yaml`:
   ```yaml
   search_space:
     my_new_param:
       type: float
       low: 0.0
       high: 1.0
   ```

2. Modify `src/hpo/search_space.py` to map to `TrialConfig`:
   ```python
   def suggest_hyperparameters(trial: optuna.Trial) -> TrialConfig:
       return TrialConfig(
           my_new_param=trial.suggest_float("my_new_param", 0.0, 1.0),
           ...
       )
   ```

3. Use in model/training code:
   ```python
   config.my_new_param  # Access in src/models/ or src/training/
   ```

### Multi-Objective Optimization

Optimize for both F1 and storage efficiency:

```yaml
optimization_objectives:
  - metric: val_f1_macro
    direction: maximize
    weight: 0.7
  - metric: checkpoint_size_mb
    direction: minimize
    weight: 0.3
```

Requires modifying `src/hpo/trial_executor.py` to return tuple of objectives to Optuna.

---

## Next Steps

1. **Scale to Full Search**: Increase `n_trials` to 1000 in `config/hpo_study.yaml`
2. **Distributed Execution**: Split trials across multiple machines (share `--study-db` via network storage)
3. **Ensemble Models**: Train ensemble of top-5 trials and average predictions
4. **Error Analysis**: Inspect misclassified examples from best model:
   ```bash
   python src/analysis/error_analysis.py \
     --trial-id <best_trial_uuid> \
     --split test \
     --output errors.csv
   ```

---

## Reference

- **Spec**: [spec.md](./spec.md)
- **Implementation Plan**: [plan.md](./plan.md)
- **Data Model**: [data-model.md](./data-model.md)
- **Contracts**: [contracts/](./contracts/)
- **Task List**: [tasks.md](./tasks.md) (generated by `/speckit.tasks`)

For issues or questions, consult the project documentation or open a GitHub issue.
