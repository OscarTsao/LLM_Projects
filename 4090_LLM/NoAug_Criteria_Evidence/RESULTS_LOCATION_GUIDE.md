# HPO Results Location Guide
**Generated:** 2025-10-25 02:16

## 📁 Where Your Results Are Stored

### **1. Optuna Database (Trial Metadata)**
**Location:** `_optuna/noaug.db`
**Contains:**
- All trial configurations
- Trial states (COMPLETE, FAILED, PRUNED, RUNNING)
- Objective values (F1 scores)
- Trial parameters (hyperparameters)

**Access:**
```bash
# View all trials
sqlite3 _optuna/noaug.db "SELECT * FROM trials;"

# Get best trial
sqlite3 _optuna/noaug.db "SELECT trial_id, value FROM trials WHERE state='COMPLETE' ORDER BY value DESC LIMIT 1;"

# Get trial parameters
sqlite3 _optuna/noaug.db "SELECT * FROM trial_params WHERE trial_id=<ID>;"
```

---

### **2. MLflow Tracking (Models & Metrics)**
**Location:** `_runs/mlruns/`
**Structure:**
```
_runs/mlruns/
├── 0/                          # Default experiment
├── 775384145209058606/         # Main HPO experiment
│   ├── <run_id_1>/            # Each trial gets a run
│   │   ├── artifacts/         # Model checkpoints (*.pt)
│   │   ├── metrics/           # Training metrics
│   │   ├── params/            # Hyperparameters
│   │   └── tags/              # Metadata
│   ├── <run_id_2>/
│   └── ...
└── models/                     # Registered models
```

**Access:**
```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:///_runs/mlruns

# Then open: http://localhost:5000
```

---

### **3. Model Checkpoints (When Saved)**
**Note:** Currently, models are NOT being saved to disk during HPO trials.

**Why?** HPO is exploring 5000+ configurations. Saving all models would require:
- ~500MB per model × 5000 = ~2.5TB storage
- Significant I/O overhead slowing down trials

**Solution:** Best model will be saved after HPO completes via:
```bash
# After HPO completes, refit best model
make refit HPO_TASK=criteria

# This will save to:
# outputs/checkpoints/best_checkpoint.pt
```

---

### **4. Current Best Results**

**Access Best Trial:**
```bash
# Get best trial ID
BEST_ID=$(sqlite3 _optuna/noaug.db "SELECT trial_id FROM trials WHERE state='COMPLETE' ORDER BY value DESC LIMIT 1;")

# Get best trial configuration
sqlite3 _optuna/noaug.db "SELECT * FROM trial_params WHERE trial_id=$BEST_ID;"

# Get best F1 score
sqlite3 _optuna/noaug.db "SELECT value FROM trials WHERE trial_id=$BEST_ID;"
```

**Current Best (as of latest check):**
- **Trial ID:** 159
- **F1 Score:** 0.701
- **Status:** COMPLETE

---

### **5. Logs & Monitoring**
**Locations:**
```
hpo_supermax_run.log          # Main HPO execution log
hpo_supervisor.log            # Auto-recovery supervisor
smart_monitor.log             # Detailed monitoring
hpo_resources.log             # Resource usage
hpo_progress.log              # Progress tracking
```

---

## 🔍 How to Access Results

### **Option 1: Query Optuna Database**
```bash
# Best trial
sqlite3 _optuna/noaug.db "SELECT trial_id, value FROM trials WHERE state='COMPLETE' ORDER BY value DESC LIMIT 1;"

# Top 10 trials
sqlite3 _optuna/noaug.db "SELECT trial_id, value FROM trials WHERE state='COMPLETE' ORDER BY value DESC LIMIT 10;"

# Trial breakdown
sqlite3 _optuna/noaug.db "SELECT state, COUNT(*) FROM trials GROUP BY state;"
```

### **Option 2: MLflow UI**
```bash
# Start UI
cd /media/user/SSD1/YuNing/NoAug_Criteria_Evidence
mlflow ui --backend-store-uri file:///_runs/mlruns

# Open browser to: http://localhost:5000
# Navigate to experiment "NoAug_Criteria_Evidence"
```

### **Option 3: Python API**
```python
import optuna

# Load study
study = optuna.load_study(
    study_name="noaug-criteria-supermax",
    storage="sqlite:///_optuna/noaug.db"
)

# Get best trial
best_trial = study.best_trial
print(f"Best F1: {best_trial.value}")
print(f"Best params: {best_trial.params}")
```

---

## 📊 What Gets Saved When

### **During HPO (Current):**
✅ Trial configurations → Optuna DB
✅ Trial states → Optuna DB
✅ Objective values (F1) → Optuna DB
✅ MLflow metrics → MLflow tracking
❌ Model checkpoints → NOT saved (too many trials)

### **After HPO Completes:**
✅ Best configuration → `outputs/hpo_stage2/best_config.yaml`
✅ Best model checkpoint → `outputs/checkpoints/best_checkpoint.pt`
✅ Evaluation metrics → `outputs/metrics/`
✅ MLflow experiment → Archived

---

## 🎯 Quick Reference

| What | Where | How to Access |
|------|-------|---------------|
| **Trial metadata** | `_optuna/noaug.db` | `sqlite3 _optuna/noaug.db` |
| **Best F1 score** | `_optuna/noaug.db` | Query: `SELECT MAX(value) FROM trials WHERE state='COMPLETE';` |
| **Training metrics** | `_runs/mlruns/775384145209058606/` | `mlflow ui` |
| **Best config** | Not yet (after refit) | `outputs/hpo_stage2/best_config.yaml` |
| **Best model** | Not yet (after refit) | `outputs/checkpoints/best_checkpoint.pt` |
| **Logs** | `*.log` files | `tail -f <logfile>` |

---

## 📝 Next Steps (After HPO Completes)

1. **Identify best configuration:**
   ```bash
   # Optuna will save this automatically
   cat outputs/hpo_stage2/best_config.yaml
   ```

2. **Refit on full dataset:**
   ```bash
   make refit HPO_TASK=criteria
   ```

3. **Get final model:**
   ```bash
   # Saved to:
   outputs/checkpoints/best_checkpoint.pt
   ```

4. **Evaluate:**
   ```bash
   make eval CHECKPOINT=outputs/checkpoints/best_checkpoint.pt
   ```

---

## 🔧 Troubleshooting

**Q: Where are model checkpoints?**
A: Not saved during HPO (too many trials). Use `make refit` after HPO to get best model.

**Q: How to view MLflow logs?**
A: `mlflow ui --backend-store-uri file:///_runs/mlruns` then open http://localhost:5000

**Q: Can I get intermediate checkpoints?**
A: No, but you can refit any trial's config by extracting params from Optuna DB.

**Q: What if I want to save models during HPO?**
A: Edit `scripts/tune_max.py` to add checkpoint saving in the training loop (not recommended - huge storage overhead).

---

## 📍 Summary

**Your results are stored in 2 main locations:**

1. **`_optuna/noaug.db`** - All trial data, configurations, F1 scores
2. **`_runs/mlruns/`** - MLflow tracking data (metrics, params)

**Best model will be saved AFTER HPO completes** via the refit stage.

**Current status:** HPO running, 14+ trials complete, best F1 = 0.701
