# Quick Reference Card

## 🚀 Essential Commands

```bash
make help                  # Show all available commands
make train                 # Train model (auto-logs to MLflow)
make train-optuna          # Hyperparameter optimization
make evaluate              # Evaluate trained model
```

## 📊 Monitoring Dashboards

| Dashboard | URL | Command |
|-----------|-----|---------|
| **MLflow UI** | http://localhost:5000 | `make mlflow-ui` |
| **TensorBoard** | http://localhost:6006 | `make tensorboard` |
| **Optuna Dashboard** | http://localhost:8080 | `make optuna-dashboard` |

## ✅ MLflow Auto-Logging Status

### Dev Container
```bash
# ✅ Always enabled - no setup needed!
make train
# → Open http://localhost:5000
```

### Mamba Environment
```bash
# ✅ Enabled after activation
mamba activate redsm5
make train
# → Open http://localhost:5000
```

### Manual Setup
```bash
# Enable once per session
source scripts/setup_env.sh
make train
```

## 🔍 Quick Checks

```bash
# Check MLflow is configured
echo $MLFLOW_TRACKING_URI
# Should show: http://localhost:5000 or http://mlflow:5000

# Test MLflow server
curl http://localhost:5000/health
# Should return: {"status": "OK"}

# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🎯 Common Workflows

### Run Quick Test
```bash
make train model.num_epochs=2 model.batch_size=8
```

### Compare Datasets
```bash
# Train with different datasets
make train dataset=original
make train dataset=original_nlpaug
make train dataset=original_hybrid

# Compare in MLflow UI
```

### Hyperparameter Search
```bash
# Quick test (3 trials)
python -m src.training.train_optuna n_trials=3 model.num_epochs=2

# Full search (500 trials - default)
make train-optuna
```

## 📁 Output Locations

```
outputs/train/
├── checkpoints/last.pt      # Resume training
├── best/
│   ├── model.pt             # Best model
│   ├── config.yaml          # Config used
│   └── val_metrics.json     # Validation metrics
└── test_metrics.json        # Final test results
```

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| MLflow not logging | Check: `echo $MLFLOW_TRACKING_URI` |
| Server not running | Dev: `docker restart redsm5-mlflow`<br>Local: `mlflow server --host 0.0.0.0 --port 5000` |
| GPU not detected | `python -c "import torch; print(torch.cuda.is_available())"` |
| Out of memory | Reduce batch size: `model.batch_size=8` |

## 📚 Documentation

| File | Description |
|------|-------------|
| `AUTO_LOGGING_SETUP.md` | MLflow always-on guide |
| `MLFLOW_GUIDE.md` | Complete MLflow documentation |
| `SETUP_GUIDE.md` | Full setup and validation |
| `CLAUDE.md` | Architecture guide |
| `README.md` | Project overview |

## 💡 Pro Tips

1. **Use Dev Container** - Everything configured automatically
2. **Check MLflow UI after training** - View all metrics and artifacts
3. **Compare runs** - Select multiple runs in MLflow UI → Compare
4. **Download best model** - MLflow UI → Experiment → Best run → Artifacts
5. **Use `make help`** - See all available commands

## 🆘 Getting Help

1. Check troubleshooting in `SETUP_GUIDE.md`
2. Review `MLFLOW_GUIDE.md` for MLflow issues
3. Read `.devcontainer/README.md` for container problems
4. Run `make help` for command reference

---

**Keep this card handy for quick reference!**
