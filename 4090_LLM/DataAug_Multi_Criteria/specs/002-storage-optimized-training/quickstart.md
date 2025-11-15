# Quickstart: Storage-Optimized Training & HPO Pipeline

**Feature**: Storage-Optimized Training & HPO Pipeline  
**Version**: 1.0  
**Last Updated**: 2025-01-14

## Prerequisites

- Docker installed with GPU support (NVIDIA Container Toolkit)
- Git configured with repository access
- Hugging Face account with CLI authentication
- Remote MLflow tracking server with TLS (authenticated)
- HashiCorp Vault or equivalent token service (for MLflow auth)
- Minimum 50GB free disk space for experiments

## Quick Start (5 minutes)

### 1. Clone and Setup

```bash
# Clone repository
git clone <repository-url>
cd DataAug_Multi_Criteria

# Authenticate with Hugging Face
huggingface-cli login
# Paste your HF token when prompted

# Build Docker image
make docker-build
```

### 2. Configure Environment

Create `.env` file in project root:

```bash
# MLflow tracking
MLFLOW_TRACKING_URI=https://mlflow.example.com
VAULT_ADDR=https://vault.example.com
VAULT_TOKEN_PATH=/mlflow/token

# Hugging Face
HF_HOME=/workspace/.cache/huggingface

# Experiment settings
EXPERIMENTS_DIR=/workspace/experiments
```

### 3. Run Your First Training

```bash
# Interactive Docker container
make docker-run

# Inside container: Train a single model
make train

# Monitor progress
tail -f experiments/trial_*/logs/train.jsonl
```

**Expected output**: Model trains for 10 epochs, checkpoints saved to `experiments/trial_{uuid}/checkpoints/`, metrics logged to MLflow.

### 4. Run Small HPO Study

```bash
# Inside container: Run 10-trial HPO study
poetry run python -m dataaug_multi_both.cli.hpo \
  --study-name "quickstart-hpo" \
  --num-trials 10 \
  --optimization-metric "val_f1" \
  --seed 42

# Evaluate best model on test set
poetry run python -m dataaug_multi_both.cli.evaluate_study \
  --study-dir experiments/study_*/
```

**Expected output**: 10 trials complete in ~30 minutes (depending on hardware), test evaluation report generated.

---

## Common Workflows

### Workflow 1: Train and Resume

Train a model, interrupt it, and resume from the latest checkpoint.

```bash
# Start training
poetry run python -m dataaug_multi_both.cli.train \
  --model "mental-bert" \
  --max-epochs 20 \
  --seed 42

# Interrupt with Ctrl+C after a few epochs

# Resume from latest checkpoint
CHECKPOINT=$(ls -t experiments/trial_*/checkpoints/epoch_*.pt | head -1)
poetry run python -m dataaug_multi_both.cli.train \
  --resume-from $CHECKPOINT \
  --max-epochs 20
```

**Validation**: Check MLflow UI - metrics should be continuous without gaps or duplicates.

---

### Workflow 2: Large-Scale HPO with Storage Management

Run a 1000-trial HPO study with aggressive storage optimization.

```bash
# Configure aggressive retention policy
cat > configs/retention/aggressive.yaml <<EOF
keep_last_n: 0
keep_best_k: 1
keep_best_k_max: 2
max_total_size_gb: 5
min_interval_epochs: 2
disk_space_threshold_percent: 10.0
pruning_strategy: tiered
EOF

# Run 1000-trial study
poetry run python -m dataaug_multi_both.cli.hpo \
  --study-name "large-scale-hpo" \
  --num-trials 1000 \
  --optimization-metric "val_f1" \
  --config configs/hpo.yaml \
  --seed 42

# Monitor disk usage during run
watch -n 60 'df -h | grep /workspace && du -sh experiments/'

# After completion, evaluate on test set
poetry run python -m dataaug_multi_both.cli.evaluate_study \
  --study-dir experiments/study_large-scale-hpo/
```

**Expected behavior**:
- Disk usage stays under 5GB for checkpoints
- Only best checkpoint from each trial retained
- When disk < 10%, aggressive pruning activates
- All 1000 trials complete without storage errors
- Test evaluation report generated with best trial metrics

---

### Workflow 3: Multi-Model Comparison

Compare 5 different transformer models on the same task.

```bash
# Create model comparison config
cat > configs/model_comparison.yaml <<EOF
models:
  - mental-bert
  - psychbert
  - clinicalbert
  - bert-base-uncased
  - roberta-base
EOF

# Run comparison study (1 model = 1 trial, 5 trials total)
poetry run python -m dataaug_multi_both.cli.hpo \
  --study-name "model-comparison" \
  --num-trials 5 \
  --optimization-metric "val_f1" \
  --config configs/model_comparison.yaml \
  --seed 42

# List results
poetry run python -m dataaug_multi_both.cli.list_studies \
  --format table \
  --filter-status completed
```

**Analysis**: Use MLflow UI to compare validation curves across models and identify best-performing architecture.

---

### Workflow 4: Cleanup and Maintenance

Manage storage by cleaning orphaned artifacts.

```bash
# Dry run: Preview what would be deleted
poetry run python -m dataaug_multi_both.cli.cleanup \
  --study-dir experiments/study_*/  \
  --dry-run

# Clean orphaned artifacts (not referenced by any trial)
poetry run python -m dataaug_multi_both.cli.cleanup \
  --study-dir experiments/study_*/  \
  --orphaned-only

# Aggressive cleanup: Keep only best checkpoint per study
poetry run python -m dataaug_multi_both.cli.cleanup \
  --study-dir experiments/study_*/  \
  --aggressive

# Verify disk space freed
du -sh experiments/
```

---

## Configuration Customization

### Custom Training Configuration

Create `configs/my_training.yaml`:

```yaml
defaults:
  - train
  - _self_

model:
  model_id: "mental-bert"
  num_criteria_classes: 20
  task_weights:
    criteria_matcher: 1.5
    evidence_binder: 1.0

dataset:
  dataset_id: "irlab-udc/redsm5"
  revision: "main"

trainer:
  max_epochs: 30
  batch_size: 16
  learning_rate: 3e-5
  optimization_metric: "val_criteria_f1"
  mixed_precision: true

retention:
  keep_last_n: 2
  keep_best_k: 3

seeds:
  python: 42
  numpy: 42
  torch: 42
  torch_cuda: 42
```

Run with custom config:

```bash
poetry run python -m dataaug_multi_both.cli.train \
  --config configs/my_training.yaml
```

---

### Custom HPO Search Space

Create `configs/my_hpo.yaml`:

```yaml
defaults:
  - hpo
  - _self_

study:
  name: "custom-hpo"
  num_trials: 100
  optimization_metric: "val_evidence_char_f1"

search_space:
  model_id:
    type: categorical
    choices: [mental-bert, psychbert]
  
  learning_rate:
    type: loguniform
    low: 1e-5
    high: 5e-4
  
  batch_size:
    type: categorical
    choices: [8, 16]
  
  trainer.max_epochs:
    type: int
    low: 10
    high: 30
    step: 5
```

Run with custom HPO config:

```bash
poetry run python -m dataaug_multi_both.cli.hpo \
  --config configs/my_hpo.yaml
```

---

## Monitoring and Debugging

### View Training Logs

```bash
# Real-time JSON logs (structured)
tail -f experiments/trial_*/logs/train.jsonl | jq '.message'

# Filter by severity
tail -f experiments/trial_*/logs/train.jsonl | jq 'select(.severity == "ERROR")'

# Human-readable stdout
poetry run python -m dataaug_multi_both.cli.train 2>&1 | tee training.log
```

### Check MLflow Metrics

```bash
# Open MLflow UI (if running locally)
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

# Or access remote tracking server
open $MLFLOW_TRACKING_URI
```

### Inspect Checkpoints

```bash
# List checkpoints for a trial
poetry run python -m dataaug_multi_both.cli.list_checkpoints \
  --trial-dir experiments/trial_*/  \
  --show-metrics

# Manually load checkpoint in Python
python
>>> import torch
>>> checkpoint = torch.load('experiments/trial_*/checkpoints/best.pt')
>>> print(checkpoint.keys())
>>> print(checkpoint['epoch'], checkpoint['metrics'])
```

### Validate Checkpoint Integrity

```bash
# Resume will validate integrity automatically
poetry run python -m dataaug_multi_both.cli.train \
  --resume-from experiments/trial_*/checkpoints/epoch_010.pt
# If hash mismatch detected, will fall back to previous checkpoint
```

---

## Troubleshooting

### Issue: MLflow Tracking Unreachable

**Symptom**: Training continues but metrics buffered to disk, warning in logs.

**Solution**:
```bash
# Check buffer size
ls -lh experiments/trial_*/logs/metrics_buffer.jsonl

# Verify MLflow connectivity
curl -X GET $MLFLOW_TRACKING_URI/api/2.0/mlflow/experiments/list

# Metrics will auto-replay when connectivity restored
# Check logs for "Metrics replayed successfully"
```

---

### Issue: Disk Space Exhausted

**Symptom**: Training fails with error "Cannot save checkpoint: disk full".

**Solution**:
```bash
# Check current disk usage
df -h /workspace

# List largest artifacts
du -sh experiments/*/ | sort -h | tail -10

# Run aggressive cleanup
poetry run python -m dataaug_multi_both.cli.cleanup \
  --study-dir experiments/study_*/  \
  --aggressive

# Adjust retention policy for future runs
# Edit configs/retention/*.yaml: lower keep_best_k, max_total_size_gb
```

---

### Issue: Hugging Face Token Expired

**Symptom**: Long-running HPO paused with "Hugging Face token invalid/expired" warning.

**Solution**:
```bash
# Re-authenticate (in a separate terminal or SSH session)
huggingface-cli login

# Study will auto-resume within 5 minutes after token refresh
# No need to restart the HPO command
```

---

### Issue: Checkpoint Corruption After Crash

**Symptom**: Resume fails with "Checksum mismatch" error.

**Solution**:
```bash
# System automatically falls back to previous valid checkpoint
# Check logs for "Falling back to epoch_009.pt"

# If all checkpoints corrupted (unlikely), restart from scratch
poetry run python -m dataaug_multi_both.cli.train \
  --config configs/train.yaml
```

---

## Best Practices

### 1. **Reproducibility**

Always set seeds explicitly and log exact dependency versions:

```bash
# Record poetry.lock hash for reproducibility
md5sum poetry.lock > experiments/study_*/poetry.lock.md5

# Pin dataset revision in config
dataset:
  revision: "abc123def456"  # Git commit hash, not "main"
```

### 2. **Storage Management**

For long-running HPO, monitor disk usage proactively:

```bash
# Add cron job or systemd timer to check disk space
*/10 * * * * df -h /workspace | awk '/\/$/ {if ($5+0 > 90) print "WARN: Disk usage >90%"}' | logger
```

### 3. **Experiment Tracking**

Use descriptive study and run names:

```bash
poetry run python -m dataaug_multi_both.cli.hpo \
  --study-name "redsm5-bert-family-$(date +%Y%m%d)" \
  --num-trials 100
```

### 4. **Backup Critical Artifacts**

Backup best checkpoints and evaluation reports:

```bash
# After study completes
rsync -av experiments/study_*/evaluation_report.json /backup/
rsync -av experiments/trial_*/checkpoints/best_checkpoint.pt /backup/
```

### 5. **Resource Allocation**

For multi-machine training, use Docker volume mounts:

```bash
# Mount shared storage for experiments
docker run -it --gpus all \
  -v /shared/experiments:/workspace/experiments \
  -v /shared/cache:/workspace/.cache \
  dataaug-multi-both:latest
```

---

## Next Steps

After completing the quickstart:

1. **Read the full specification**: `specs/002-storage-optimized-training/spec.md`
2. **Review the data model**: `specs/002-storage-optimized-training/data-model.md`
3. **Explore CLI contracts**: `specs/002-storage-optimized-training/contracts/cli_contracts.md`
4. **Review the constitution**: `.specify/memory/constitution.md`
5. **Customize for your dataset**: Modify `configs/data/*.yaml`
6. **Scale to production**: Set up remote MLflow, Vault, and distributed storage

---

## Support

- **Documentation**: `README.md` in project root
- **Issues**: GitHub Issues (if applicable)
- **Logs**: Check `experiments/*/logs/*.jsonl` for debugging
- **MLflow UI**: Monitor experiments at `$MLFLOW_TRACKING_URI`

---

**Version**: 1.0  
**Tested with**: Docker 24.0, Python 3.10, PyTorch 2.2, CUDA 12.1  
**Estimated Setup Time**: 15 minutes (cold start, moderate network)
