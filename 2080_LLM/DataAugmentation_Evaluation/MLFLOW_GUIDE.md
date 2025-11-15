# MLflow Auto-Logging Guide

This project is configured for **automatic MLflow tracking** - every training run is automatically logged without any code changes required.

## 🎯 What Gets Auto-Logged?

### Standard Training (`make train`)

**Every run automatically logs:**
- ✅ All configuration parameters (batch_size, learning_rate, optimizer, etc.)
- ✅ Training loss (per epoch)
- ✅ Validation metrics (accuracy, precision, recall, F1, ROC-AUC) per epoch
- ✅ Test metrics (final evaluation)
- ✅ Best model checkpoint (`model.pt`)
- ✅ Configuration snapshot (`config.yaml`)
- ✅ Test metrics JSON

**Experiment name**: `redsm5-classification`

### Optuna Hyperparameter Search (`make train-optuna`)

**Only the BEST trial logs to MLflow:**
- ✅ All hyperparameters from best trial
- ✅ Best metric value
- ✅ Test metrics from best trial
- ✅ Best model checkpoint

**Why?** To avoid cluttering MLflow with 100s/1000s of trials. You get the winner without the noise.

**Experiment name**: `redsm5-optuna`

## 🚀 Quick Start

### Option 1: Dev Container (Recommended - Zero Setup!)

```bash
# 1. Open in VS Code Dev Container
# Ctrl+Shift+P → "Dev Containers: Reopen in Container"

# 2. Run training - MLflow auto-logging is ALREADY enabled!
make train

# 3. View results
# Open browser: http://localhost:5000
```

**That's it!** No configuration needed. MLflow server is running via docker-compose.

### Option 2: Mamba Environment (Local Machine)

```bash
# 1. Create/update mamba environment
make env-create  # or make env-update

# 2. Activate environment
mamba activate redsm5
# Environment variables are automatically set!

# 3. Start MLflow server (one-time, in separate terminal)
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

# 4. Run training - auto-logs to MLflow
make train

# 5. View results
# Open browser: http://localhost:5000
```

**Environment variable is set automatically** when you activate the `redsm5` conda environment.

### Option 3: Any Python Environment

```bash
# 1. Source the environment setup script
source scripts/setup_env.sh

# 2. Start MLflow server (if not running)
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

# 3. Run training - auto-logs to MLflow
python -m src.training.train

# 4. View results
# Open browser: http://localhost:5000
```

## 📊 Viewing Your Experiments

### MLflow UI

1. **Open browser**: http://localhost:5000

2. **Navigate to experiments**:
   - Left sidebar → Click experiment name
   - `redsm5-classification` - All training runs
   - `redsm5-optuna` - Best Optuna trials

3. **Explore a run**:
   - Click on any run to see details
   - **Parameters** tab: All config values
   - **Metrics** tab: Training/validation/test metrics with charts
   - **Artifacts** tab: Download model, config, etc.

4. **Compare runs**:
   - Select multiple runs (checkboxes)
   - Click "Compare" button
   - See side-by-side parameter and metric comparison

### Example: Finding Your Best Model

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Get best run by ROC-AUC
experiment = mlflow.get_experiment_by_name("redsm5-classification")
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_roc_auc DESC"],
    max_results=1
)

print(f"Best run ID: {runs.iloc[0]['run_id']}")
print(f"Best test ROC-AUC: {runs.iloc[0]['metrics.test_roc_auc']}")

# Download best model
best_run_id = runs.iloc[0]['run_id']
mlflow.artifacts.download_artifacts(
    run_id=best_run_id,
    artifact_path="model",
    dst_path="./best_model"
)
```

## 🔧 How It Works

### Environment Variables

MLflow auto-logging is enabled when `MLFLOW_TRACKING_URI` is set:

| Environment | How It's Set | Value |
|-------------|--------------|-------|
| **Dev Container** | `docker-compose.yml` | `http://mlflow:5000` |
| **Mamba/Conda** | `environment.yml` | `http://localhost:5000` |
| **Manual** | `scripts/setup_env.sh` | `http://localhost:5000` |

### Code Integration

The training code checks for MLflow availability:

```python
# src/utils/mlflow_utils.py
def is_mlflow_enabled() -> bool:
    """Check if MLflow tracking is available and configured."""
    if not MLFLOW_AVAILABLE:
        return False
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    return tracking_uri is not None and tracking_uri != ""
```

If enabled, all logging happens automatically in `src/training/engine.py`.

**If MLflow is NOT available:**
- ✅ Training runs normally
- ✅ Outputs saved to `outputs/train/` as usual
- ❌ No MLflow logging
- ❌ No errors or warnings

## 🎛️ Configuration

### Persistent Setup (Recommended)

**Method 1: Use mamba environment (automatic)**
```bash
mamba activate redsm5
# MLFLOW_TRACKING_URI is set automatically from environment.yml
```

**Method 2: Add to shell profile**
```bash
# Add to ~/.bashrc or ~/.zshrc
export MLFLOW_TRACKING_URI=http://localhost:5000

# Reload shell
source ~/.bashrc  # or source ~/.zshrc
```

**Method 3: Use .env file**
```bash
# Copy example file
cp .env.example .env

# Edit if needed (default values should work)
# nano .env

# Load in shell
set -a; source .env; set +a
```

### Temporary Setup

```bash
# Set for current session only
export MLFLOW_TRACKING_URI=http://localhost:5000

# Run training
make train
```

### Disable MLflow Temporarily

```bash
# Unset environment variable
unset MLFLOW_TRACKING_URI

# Run training - no MLflow logging
make train
```

## 🏃 Example Workflows

### Workflow 1: Quick Experiment

```bash
# In dev container
make train  # Auto-logs to MLflow

# Check MLflow UI
# http://localhost:5000
```

### Workflow 2: Compare Different Datasets

```bash
# Train with original data
mamba run -n redsm5 python -m src.training.train dataset=original

# Train with NLPAug augmentation
mamba run -n redsm5 python -m src.training.train dataset=original_nlpaug

# Train with hybrid augmentation
mamba run -n redsm5 python -m src.training.train dataset=original_hybrid

# Compare all 3 runs in MLflow UI
# Select runs → Compare
```

### Workflow 3: Hyperparameter Search

```bash
# Run Optuna search (e.g., 100 trials)
make train-optuna

# Or with custom trials
mamba run -n redsm5 python -m src.training.train_optuna n_trials=100

# Best trial auto-logged to MLflow
# Check "redsm5-optuna" experiment in UI
```

### Workflow 4: Reproduce Best Run

```bash
# 1. Find best run in MLflow UI
# 2. Download config.yaml from Artifacts tab
# 3. Copy parameter values

# 4. Re-run with same config
mamba run -n redsm5 python -m src.training.train \
    model.batch_size=32 \
    model.learning_rate=2e-5 \
    model.num_epochs=10 \
    # ... other params from best run
```

## 📈 Advanced: MLflow with Remote Server

### Using Remote MLflow Server

```bash
# Point to remote server
export MLFLOW_TRACKING_URI=http://your-mlflow-server.com:5000

# Run training - logs to remote server
make train
```

### Using MLflow with S3 Artifact Storage

```bash
# Configure AWS credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Start MLflow with S3 backend
mlflow server \
    --backend-store-uri postgresql://user:pass@localhost/mlflow \
    --default-artifact-root s3://your-bucket/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

## 🐛 Troubleshooting

### Issue: "MLflow not logging"

**Check 1: Environment variable set?**
```bash
echo $MLFLOW_TRACKING_URI
# Should output: http://localhost:5000 or http://mlflow:5000
```

**Check 2: MLflow server running?**
```bash
curl http://localhost:5000/health
# Should return: {"status": "OK"}

# Or check docker containers
docker ps | grep mlflow
```

**Check 3: Network connectivity**
```bash
# Test connection
python -c "import mlflow; mlflow.set_tracking_uri('http://localhost:5000'); print(mlflow.list_experiments())"
```

### Issue: "Cannot connect to MLflow server"

**In dev container:**
```bash
# Check all services running
docker ps

# Restart MLflow
docker restart redsm5-mlflow

# Check logs
docker logs redsm5-mlflow
```

**On host machine:**
```bash
# Check if port 5000 is in use
lsof -i :5000

# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000
```

### Issue: "Runs appear but no artifacts"

**Check artifact location:**
```bash
# In MLflow UI, check run details
# Artifacts should be in /mlflow/artifacts (dev container)
# Or ./mlruns/ (local machine)

# Verify path exists and has write permissions
ls -la /mlflow/artifacts  # or ./mlruns/
```

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)

## 🎯 Summary

✅ **Dev Container**: MLflow works automatically, zero setup
✅ **Mamba Environment**: MLflow enabled when environment activated
✅ **Manual Setup**: Source `scripts/setup_env.sh`
✅ **Always Logging**: Every `make train` logs to MLflow
✅ **Best Trials Only**: Optuna logs only the best trial
✅ **Graceful Fallback**: Works without MLflow, just doesn't log

**Access MLflow UI**: http://localhost:5000

**Start MLflow Server** (if not using dev container):
```bash
mlflow server --host 0.0.0.0 --port 5000
```
