# Epoch Configuration Explanation

## Overview
The training epochs are configured differently for different modes to balance speed and performance.

## Configuration Strategy (Updated)

### Automatic Mode Selection
The system automatically selects epoch configuration based on the study name:

- **Production Studies** (hpo, hpo-production): Fixed 100 epochs
- **Test Studies** (hpo-test, any study with "test" in name): Search epochs [5, 8, 10, 15]

**Implementation**: `src/dataaug_multi_both/cli/train.py` (around line 551)

```python
use_fixed_epochs = 100 if "test" not in study_name.lower() else None
```

## Configuration Levels

### 1. HPO Production Mode (Fixed 100 Epochs)
**Commands**: `make hpo`, `make hpo-production`
**Study Names**: `hpo_default`, `mental_health_hpo_production`

**Location**: `src/dataaug_multi_both/hpo/search_space.py`

```python
# Epochs: use fixed value if provided, otherwise search
if self.fixed_epochs is not None:
    params["epochs"] = self.fixed_epochs  # Fixed at 100
    logger.info(f"Using fixed epochs: {self.fixed_epochs}")
```

**Epochs**: Fixed at 100

**Rationale**: 
- Production HPO focuses on exploring model architectures and hyperparameters
- Fixed epochs ensures consistent training duration across all trials
- Fair comparison between different backbone models
- Reduces search space complexity (~97 hyperparameters instead of 100+)
- 100 epochs provides sufficient training for convergence
- Optuna can focus on architecture and learning dynamics without learning epoch count

**Benefits**:
- Consistent training duration
- Fair model comparison
- Faster HPO convergence
- Reduced computational variance

### 2. HPO Test Mode (Search Epochs)
**Command**: `make hpo-test`
**Study Name**: `hpo_quick_test`

**Location**: `src/dataaug_multi_both/hpo/search_space.py`

```python
else:
    params["epochs"] = trial.suggest_categorical("epochs", [5, 8, 10, 15])
    logger.info(f"Searching epochs from [5, 8, 10, 15]")
```

**Epochs**: 5, 8, 10, or 15 (searched)

**Rationale**: 
- Quick validation of HPO pipeline functionality
- Allows testing different epoch configurations
- Faster iteration during development
- Minimum 5 epochs ensures reasonable model convergence signal

### 3. Standard Training (Config File)
**Location**: `configs/train.yaml` line 10

```yaml
trainer:
  max_epochs: 100
```

**Epochs**: 100

**Rationale**:
- Full training with optimal hyperparameters
- Allows models to fully converge
- Used for final production models after HPO finds best config

## Summary by Command

| Command | Study Name | Epochs | Hyperparameters | Use Case |
|---------|-----------|--------|-----------------|----------|
| `make hpo-test` | hpo_quick_test | **Search**: [5, 8, 10, 15] | ~100 | Fast validation |
| `make hpo` | hpo_default | **Fixed**: 100 | ~97 | General exploration |
| `make hpo-production` | mental_health_hpo_production | **Fixed**: 100 | ~97 | Production run |

## How to Change

### To customize test mode epoch search:
Edit `src/dataaug_multi_both/hpo/search_space.py` (around line 321):
```python
params["epochs"] = trial.suggest_categorical("epochs", [3, 5, 8, 10])  # Your values
```

### To change production fixed epochs:
Edit `src/dataaug_multi_both/cli/train.py` (around line 551):
```python
use_fixed_epochs = 150 if "test" not in study_name.lower() else None  # Your value
```

### To override for specific study:
```bash
# This will use the study name logic
make hpo ARGS="--study-name my_custom_test"  # Will search epochs (has "test")
make hpo ARGS="--study-name my_production"   # Will use fixed 100 epochs
```

## Performance vs Speed Trade-off

| Epochs | HPO Time per Trial | HPO Total Time (500 trials) | Quality |
|--------|-------------------|----------------------------|---------|
| 5      | ~8-15 min         | ~67-125 hours              | Good signal |
| 10     | ~15-30 min        | ~125-250 hours             | Better |
| 15     | ~22-45 min        | ~183-375 hours             | Very good |
| 100    | ~2.5-5 hours      | ~1250-2500 hours           | Full convergence |

**Current Strategy**: 
- Production HPO uses fixed 100 epochs for ~1250-2500 hours total (52-104 days)
- Test HPO searches 5-15 epochs for fast validation (~0.5-2 hours for 3 trials)
- Balances thorough exploration with practical training times
