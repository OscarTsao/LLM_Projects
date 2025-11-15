# HPO Results Location and Storage

**Date**: 2025-10-27
**Status**: ✅ All results saved

---

## 📁 Storage Locations

### **1. Optuna Database** (Primary Results Storage)

**Location**:
```
_optuna/noaug.db
```

**Full Path**:
```
/media/cvrlab308/cvrlab308_4090/YuNing/DataAug_Criteria_Evidence/_optuna/noaug.db
```

**Details**:
- **Size**: 3.82 MB
- **Last Updated**: 2025-10-27 21:53:24 ✅
- **Format**: SQLite database
- **Total Studies**: 16

**Main Studies**:
1. `aug-evidence-production-2025-10-27`: **106 trials** ⭐
2. `aug-criteria-production-2025-10-27`: **21 trials**
3. `noaug-criteria-supermax`: 298 trials (older baseline)

**Access**:
```python
import optuna

# Load Evidence study
study = optuna.load_study(
    study_name='aug-evidence-production-2025-10-27',
    storage='sqlite:////media/cvrlab308/cvrlab308_4090/YuNing/DataAug_Criteria_Evidence/_optuna/noaug.db'
)

# Get best trial
best_trial = study.best_trial
print(f'Best value: {study.best_value}')
print(f'Best params: {best_trial.params}')

# Get all trials
trials = study.trials
```

---

### **2. HPO Log Files**

**Location**: Project root directory

**Evidence HPO Logs**:
- `evidence_hpo_final.log` (135 KB) - **Main production log** ⭐
  - Contains all 106 trials
  - Updated: 2025-10-27 21:53

- `evidence_hpo_prod.log` (9.5 KB)
- `evidence_hpo_prod_fixed.log` (6.3 KB)

**Criteria HPO Logs**:
- `criteria_hpo_prod_fixed.log` (19 KB) - **Main log**
  - Contains 21 trials
  - Updated: 2025-10-27 13:20

**View logs**:
```bash
# Evidence (main log)
tail -100 evidence_hpo_final.log

# Criteria
tail -100 criteria_hpo_prod_fixed.log

# Search for completed trials
grep "finished with value" evidence_hpo_final.log
```

---

### **3. Results JSON Files**

**Location**: `outputs/` directory

**Files**:
- `evidence_aug-evidence-production-2025-10-27_topk.json` ⭐
  - Top-K trials results
  - Best trial configuration
  - Performance metrics

- `criteria_aug-criteria-fast-test_topk.json`
- `criteria_aug-criteria-phase2-test_topk.json`

**Access**:
```python
import json

with open('outputs/evidence_aug-evidence-production-2025-10-27_topk.json') as f:
    results = json.load(f)

best_trial = results['best_trial']
top_k = results['top_k_trials']
```

---

### **4. MLflow Tracking**

**Location**:
- Database: `mlflow.db` (508 KB)
- Runs: `mlruns/` directory

**Note**: MLflow last updated 2025-10-27 00:14 (before production run)
- May not contain latest HPO results
- Optuna DB is the authoritative source

**Access**:
```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# View at http://localhost:5000
```

---

### **5. Outputs Directory**

**Location**: `outputs/`

**Contents**:
```
outputs/
├── criteria/                           # Criteria task results
├── evidence_aug-evidence-production-2025-10-27_topk.json  # Evidence results ⭐
├── criteria_aug-criteria-fast-test_topk.json
├── criteria_aug-criteria-phase2-test_topk.json
└── mlruns/                            # MLflow runs (symlink)
```

---

## 🔍 How to Access Results

### **Option 1: Query Optuna Database** (Recommended)

```python
import optuna

# Load study
study = optuna.load_study(
    study_name='aug-evidence-production-2025-10-27',
    storage='sqlite:////media/cvrlab308/cvrlab308_4090/YuNing/DataAug_Criteria_Evidence/_optuna/noaug.db'
)

# Best trial
print(f'Best value: {study.best_value:.4f}')
print(f'Best trial number: {study.best_trial.number}')
print(f'Best parameters: {study.best_trial.params}')

# All completed trials
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f'Completed trials: {len(completed)}')

# Top 5 trials
sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
for i, trial in enumerate(sorted_trials, 1):
    print(f'{i}. Trial #{trial.number}: {trial.value:.4f}')
```

### **Option 2: Read Results JSON**

```python
import json

with open('outputs/evidence_aug-evidence-production-2025-10-27_topk.json') as f:
    results = json.load(f)

# Best configuration
best = results['best_trial']
print(f"Best trial: {best['number']}")
print(f"Best value: {best['value']}")
print(f"Best params: {best['params']}")

# Top K trials
for trial in results['top_k_trials']:
    print(f"Trial {trial['number']}: {trial['value']}")
```

### **Option 3: Parse Log Files**

```bash
# Find all completed trials
grep "finished with value" evidence_hpo_final.log

# Find best trial
grep "Best is trial" evidence_hpo_final.log | tail -1

# Count trials
grep -c "Trial.*finished" evidence_hpo_final.log
```

---

## 📊 Database Verification

**Verify database is updated**:
```bash
python -c "
import optuna
import sqlite3
import os

db_path = '_optuna/noaug.db'
print(f'Database size: {os.path.getsize(db_path) / (1024*1024):.2f} MB')

from datetime import datetime
mtime = os.path.getmtime(db_path)
print(f'Last modified: {datetime.fromtimestamp(mtime)}')

study = optuna.load_study(
    study_name='aug-evidence-production-2025-10-27',
    storage=f'sqlite:///{db_path}'
)
print(f'Total trials in Evidence study: {len(study.trials)}')
print(f'Best value: {study.best_value}')
"
```

**Expected Output**:
```
Database size: 3.82 MB
Last modified: 2025-10-27 21:53:24
Total trials in Evidence study: 106
Best value: 0.678048780487805
```

✅ **All verified** - Database is up to date!

---

## 📋 Summary

### **Where Everything Is**:

| Resource | Location | Status |
|----------|----------|--------|
| **Optuna DB** | `_optuna/noaug.db` | ✅ Updated (21:53) |
| **Evidence Log** | `evidence_hpo_final.log` | ✅ 135 KB |
| **Criteria Log** | `criteria_hpo_prod_fixed.log` | ✅ 19 KB |
| **Results JSON** | `outputs/evidence_*_topk.json` | ✅ Exists |
| **MLflow DB** | `mlflow.db` | ⚠️ Old (00:14) |
| **MLflow Runs** | `mlruns/` | ⚠️ May be outdated |

### **Primary Source of Truth**:
✅ **`_optuna/noaug.db`** - Contains all 106 trials with complete parameters and results

### **Best Trial Saved**:
- Trial #69
- 67.80% exact match
- xlm-roberta-base + nlpaug augmentation
- All parameters stored in database

---

## 🚀 Next Steps

To use these results:

1. **Load best configuration** from Optuna DB
2. **Retrain model** with best hyperparameters
3. **Save final checkpoint** for production use
4. **Document best configuration** in model card

**Example**:
```bash
# Get best config and train final model
python scripts/train_best.py \
    --study aug-evidence-production-2025-10-27 \
    --storage sqlite:///_optuna/noaug.db \
    --task evidence
```
