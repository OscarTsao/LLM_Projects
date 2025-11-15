# ✅ MLflow Auto-Logging is Always Enabled

## Summary

**Every time you run `make train` or `make train-optuna`, your experiments AUTOMATICALLY log to MLflow.**

No code changes needed. No manual configuration needed. It just works.

## How It Works

### 🐳 In Dev Container (Recommended)
**Status**: ✅ **ALWAYS AUTO-LOGS**

```bash
# 1. Open in dev container
# Ctrl+Shift+P → "Dev Containers: Reopen in Container"

# 2. Run training
make train

# 3. View in MLflow UI
# http://localhost:5000
```

**Why it works**:
- Environment variable `MLFLOW_TRACKING_URI=http://mlflow:5000` is set in `docker-compose.yml`
- MLflow server is running automatically
- PostgreSQL backend is running automatically

**No setup needed!**

### 🐍 With Mamba/Conda Environment
**Status**: ✅ **ALWAYS AUTO-LOGS** (after environment activation)

```bash
# 1. Activate environment (sets MLFLOW_TRACKING_URI automatically!)
mamba activate redsm5

# 2. Start MLflow server (one-time, keep running in separate terminal)
mlflow server --host 0.0.0.0 --port 5000

# 3. Run training
make train

# 4. View in MLflow UI
# http://localhost:5000
```

**Why it works**:
- `environment.yml` includes:
  ```yaml
  variables:
    MLFLOW_TRACKING_URI: http://localhost:5000
  ```
- Variable is set automatically when you activate the environment
- Training code detects it and enables MLflow logging

**Setup needed**: Start MLflow server once (keep it running)

### 🔧 Manual Setup (Any Environment)
**Status**: ✅ **AUTO-LOGS** (after sourcing setup script)

```bash
# 1. Source environment setup (sets MLFLOW_TRACKING_URI)
source scripts/setup_env.sh

# 2. Start MLflow server (one-time, keep running)
mlflow server --host 0.0.0.0 --port 5000

# 3. Run training
python -m src.training.train

# 4. View in MLflow UI
# http://localhost:5000
```

**Why it works**:
- `scripts/setup_env.sh` sets `MLFLOW_TRACKING_URI`
- Training code detects it and enables MLflow logging

**Setup needed**:
- Source script once per shell session
- Start MLflow server once (keep it running)

## What Gets Logged Automatically

### Every Training Run (`make train`)

| Category | What's Logged |
|----------|---------------|
| **Parameters** | All config values (batch_size, learning_rate, optimizer, scheduler, etc.) |
| **Metrics** | Training loss (per epoch), Validation metrics (per epoch), Test metrics (final) |
| **Artifacts** | Best model checkpoint, Config YAML, Test metrics JSON |
| **Tags** | model_type, framework |

### Optuna Runs (`make train-optuna`)

| What's Logged | Description |
|---------------|-------------|
| **Best Trial Only** | Only the best trial is logged (not all 100s of trials) |
| **Parameters** | All hyperparameters from best trial |
| **Metrics** | Best metric value, Test metrics |
| **Artifacts** | Best model checkpoint |
| **Tags** | trial_number, optimization_study |

## Verification Checklist

### ✓ Verify Environment Variable is Set

```bash
# Should output: http://localhost:5000 or http://mlflow:5000
echo $MLFLOW_TRACKING_URI
```

If empty:
- **Dev container**: Rebuild container
- **Mamba**: Deactivate and reactivate environment
- **Manual**: Source `scripts/setup_env.sh`

### ✓ Verify MLflow Server is Running

```bash
# Test health endpoint
curl http://localhost:5000/health
# Should return: {"status": "OK"}
```

If fails:
- **Dev container**: Check `docker ps | grep mlflow`
- **Local**: Start server with `mlflow server --host 0.0.0.0 --port 5000`

### ✓ Verify Auto-Logging Works

```bash
# Run a quick test (2 epochs)
make train model.num_epochs=2

# Open MLflow UI
# http://localhost:5000

# Check "redsm5-classification" experiment
# You should see your run!
```

## Quick Reference

| Environment | Command to Enable | MLflow Server Location |
|-------------|-------------------|------------------------|
| **Dev Container** | *Already enabled* | http://mlflow:5000 |
| **Mamba** | `mamba activate redsm5` | http://localhost:5000 |
| **Manual** | `source scripts/setup_env.sh` | http://localhost:5000 |

## Troubleshooting

### "No runs showing in MLflow UI"

**Check 1**: Is environment variable set?
```bash
echo $MLFLOW_TRACKING_URI
```

**Check 2**: Is MLflow server running?
```bash
curl http://localhost:5000/health
```

**Check 3**: Run training and watch for errors
```bash
make train
# Look for any MLflow-related errors in output
```

### "MLflow server not running"

**Dev container**:
```bash
docker ps | grep mlflow  # Check if running
docker restart redsm5-mlflow  # Restart if needed
```

**Local machine**:
```bash
# Start server in separate terminal
mlflow server --host 0.0.0.0 --port 5000
```

### "Environment variable not set"

**Mamba environment**:
```bash
# Recreate environment with updated config
mamba env remove -n redsm5
make env-create
mamba activate redsm5
echo $MLFLOW_TRACKING_URI  # Should now be set
```

**Manual**:
```bash
source scripts/setup_env.sh
echo $MLFLOW_TRACKING_URI  # Should now be set
```

## FAQ

### Q: Do I need to modify my training code?
**A**: No! Auto-logging is already integrated in the training code.

### Q: What if I don't want MLflow logging?
**A**: Unset the environment variable:
```bash
unset MLFLOW_TRACKING_URI
make train  # Runs normally, no MLflow logging
```

### Q: Can I use a remote MLflow server?
**A**: Yes! Just set the URI:
```bash
export MLFLOW_TRACKING_URI=http://your-server.com:5000
make train
```

### Q: Where are MLflow artifacts stored?
**A**:
- **Dev container**: `/mlflow/artifacts` (persistent volume)
- **Local**: `./mlruns/` directory

### Q: How do I find my best model?
**A**:
1. Open http://localhost:5000
2. Click "redsm5-classification" experiment
3. Sort by your metric (e.g., ROC-AUC)
4. Click best run → Artifacts tab → Download model.pt

### Q: Does Optuna log all trials?
**A**: No, only the best trial to avoid clutter. Individual trials run but don't log to MLflow.

## Next Steps

1. **Start training**: `make train`
2. **Open MLflow UI**: http://localhost:5000
3. **Explore your runs**: Click on experiments, view metrics, compare runs
4. **Read full guide**: [MLFLOW_GUIDE.md](MLFLOW_GUIDE.md)

---

**🎉 You're all set! MLflow will now automatically log every training run.**

For detailed documentation, see [MLFLOW_GUIDE.md](MLFLOW_GUIDE.md).
