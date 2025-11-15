# Setup and Validation Guide

This guide walks you through setting up the development environment and validating that all components work correctly.

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with drivers (for GPU support)
- VS Code with Dev Containers extension
- Git

## Step 1: Setup Dev Container

### 1.1 Install NVIDIA Container Toolkit (for GPU support)

```bash
# Run the setup script from the project root
.devcontainer/setup-nvidia-docker.sh

# Verify installation
docker info | grep -i runtime  # Should show "nvidia"
nvidia-smi  # Should show your GPU
```

### 1.2 Open Project in Dev Container

1. Open the project folder in VS Code
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
3. Select "Dev Containers: Reopen in Container"
4. Wait for all services to build and start (~5-10 minutes first time)

### 1.3 Verify Services

```bash
# Inside the dev container terminal:

# Check all services are running
docker ps
# Should see: redsm5-dev, redsm5-mlflow, redsm5-postgres

# Check GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Devices: {torch.cuda.device_count()}')"

# Check MLflow connection
python -c "import os; print(f'MLflow URI: {os.getenv(\"MLFLOW_TRACKING_URI\")}')"

# Check PostgreSQL
psql -h postgres -U mlflow -d mlflow -c "SELECT 1;"
```

## Step 2: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Or using the Makefile
make pip-sync
```

## Step 3: Verify All Commands

Run through all make commands to ensure everything works:

```bash
# Show all available commands
make help

# Code quality
make format    # Format code with black
make lint      # Check code with ruff
make test      # Run pytest (may have no tests initially)
make clean     # Clean cache files
```

## Step 4: Test Data Augmentation

### 4.1 Verify Data Exists

```bash
# Check for required data files
ls -lh Data/ReDSM5/
ls -lh Data/GroundTruth/
```

### 4.2 Run Augmentation Pipelines

```bash
# Generate augmented datasets
make augment-nlpaug        # Takes ~5-15 minutes
make augment-textattack    # Takes ~10-20 minutes
make augment-hybrid        # Takes ~15-30 minutes

# Or run all at once (WARNING: takes significant time)
# make augment-all

# Verify outputs
ls -lh Data/Augmentation/
```

Expected output files:
- `Data/Augmentation/nlpaug_dataset_YYYYMMDD_HHMMSS.csv`
- `Data/Augmentation/textattack_dataset_YYYYMMDD_HHMMSS.csv`
- `Data/Augmentation/hybrid_dataset_YYYYMMDD_HHMMSS.csv`

## Step 5: Test Training Pipeline

### 5.1 Quick Training Test (CPU/GPU)

```bash
# Run a quick training test with minimal epochs
mamba run -n redsm5 python -m src.training.train \
    model.num_epochs=2 \
    model.batch_size=8

# Expected output:
# - Training progress bars
# - Validation metrics after each epoch
# - Final test metrics
# - Artifacts saved to outputs/train/
```

### 5.2 Verify Training Artifacts

```bash
# Check output directory structure
tree outputs/train/

# Expected structure:
# outputs/train/
# ├── checkpoints/
# │   └── last.pt
# ├── best/
# │   ├── model.pt
# │   ├── config.yaml
# │   └── val_metrics.json
# └── test_metrics.json
```

### 5.3 Check MLflow Tracking

1. Open MLflow UI: http://localhost:5000
2. Navigate to "redsm5-classification" experiment
3. Verify you see your training run with:
   - Parameters (all config values)
   - Metrics (train_loss, val_*, test_*)
   - Artifacts (model.pt, config.yaml, test_metrics.json)

## Step 6: Test Optuna Hyperparameter Search

### 6.1 Quick Optuna Test

```bash
# Run Optuna with very few trials for testing
mamba run -n redsm5 python -m src.training.train_optuna \
    n_trials=3 \
    model.num_epochs=2

# Expected output:
# - Multiple trial runs
# - Best trial summary
# - Best trial artifacts saved
```

### 6.2 Check Optuna in MLflow

1. Open MLflow UI: http://localhost:5000
2. Navigate to "redsm5-optuna" experiment
3. Verify you see the best trial logged

### 6.3 Launch Optuna Dashboard (Optional)

```bash
# Start Optuna Dashboard
make optuna-dashboard

# Access at: http://localhost:8080
```

## Step 7: Test Monitoring Dashboards

### 7.1 TensorBoard

```bash
# Create some logs first (if training didn't create them)
mkdir -p logs

# Start TensorBoard
make tensorboard

# Access at: http://localhost:6006
```

### 7.2 MLflow UI

```bash
# MLflow should already be running
# Access at: http://localhost:5000

# Check experiments:
# - redsm5-classification (training runs)
# - redsm5-optuna (best trials)
```

## Step 8: Test DVC (Optional)

```bash
# Initialize DVC (already done in postCreateCommand)
make dvc-init

# Check DVC status
make dvc-status

# Configure remote storage (example with S3)
dvc remote add -d myremote s3://mybucket/path

# Add data to DVC tracking
dvc add Data/ReDSM5/redsm5_posts.csv

# Push to remote
make dvc-push
```

## Step 9: Verify Evaluation Pipeline

```bash
# Evaluate the best model
mamba run -n redsm5 python -m src.training.evaluate \
    evaluation.checkpoint=outputs/train/best/model.pt \
    evaluation.split=test

# Expected output:
# - Evaluation metrics printed
# - Metrics saved to outputs/train/test_metrics.json
```

## Troubleshooting

### Issue: MLflow not tracking

**Solution**:
```bash
# Check environment variable
echo $MLFLOW_TRACKING_URI
# Should output: http://mlflow:5000

# Test connection
python -c "import mlflow; mlflow.set_tracking_uri('http://mlflow:5000'); print(mlflow.get_tracking_uri())"

# Check MLflow server logs
docker logs redsm5-mlflow
```

### Issue: GPU not detected

**Solution**:
```bash
# Verify on host
nvidia-smi

# Verify Docker runtime
docker info | grep -i runtime

# Rebuild container
# Ctrl+Shift+P → "Dev Containers: Rebuild Container"
```

### Issue: PostgreSQL connection errors

**Solution**:
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# View logs
docker logs redsm5-postgres

# Restart PostgreSQL
docker restart redsm5-postgres
```

### Issue: Out of memory during training

**Solution**:
```bash
# Reduce batch size
mamba run -n redsm5 python -m src.training.train model.batch_size=8

# Increase gradient accumulation
mamba run -n redsm5 python -m src.training.train \
    model.batch_size=8 \
    model.gradient_accumulation_steps=4

# Reduce sequence length
mamba run -n redsm5 python -m src.training.train model.max_seq_length=128
```

## Success Checklist

- [ ] Dev container built and all services running
- [ ] GPU detected (if applicable)
- [ ] MLflow UI accessible at http://localhost:5000
- [ ] Python dependencies installed
- [ ] Data augmentation pipelines run successfully
- [ ] Training pipeline completes without errors
- [ ] Training artifacts saved to `outputs/train/`
- [ ] MLflow tracking captures all runs
- [ ] Optuna search runs successfully
- [ ] Evaluation pipeline works
- [ ] All dashboards accessible (MLflow, TensorBoard, Optuna)
- [ ] Code quality tools work (format, lint, test)

## Next Steps

Once everything is validated:

1. **Configure DVC remote** for data version control
2. **Run full augmentation** pipelines (if not already done)
3. **Start hyperparameter optimization** with more trials:
   ```bash
   make train-optuna  # Uses default 500 trials from config
   ```
4. **Monitor experiments** via MLflow UI
5. **Review best models** and select for deployment

## Additional Resources

- **CLAUDE.md**: Architecture and development guide
- **README.md**: Project overview and documentation
- **.devcontainer/README.md**: Dev container detailed documentation
- **Makefile**: Run `make help` for all available commands
