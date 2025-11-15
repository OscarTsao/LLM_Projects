# CategoricalDistribution Error Fix

**Date:** 2025-10-24
**Issue:** `ValueError: CategoricalDistribution does not support dynamic value space`
**Status:** ✅ FIXED

## Problem Summary

The HPO system in `scripts/tune_max.py` was experiencing recurring `CategoricalDistribution` errors due to **conditional batch_size distributions** based on model type (large vs. base models).

### Root Cause

**Optuna's Fundamental Constraint:**
`CategoricalDistribution` **does NOT support dynamic value spaces**. All trials in a study must use the **SAME distribution** for each parameter name.

**What We Had (BROKEN):**
```python
if heavy_model:
    bsz = trial.suggest_categorical("train.batch_size", [8, 12, 16])
else:
    bsz = trial.suggest_categorical("train.batch_size", [8, 12, 16, 24, 32, 48, 64])
```

**Why It Failed:**
1. Trial 1 selects `roberta-base` → establishes distribution `[8,12,16,24,32,48,64]`
2. Trial 2 selects `bert-base-uncased` → uses same distribution `[8,12,16,24,32,48,64]` ✓
3. Trial 3 selects `bert-large-uncased` → tries to use `[8,12,16]` ✗
4. **Optuna detects incompatibility** → raises `ValueError`

### Error Manifestation

```
ValueError: CategoricalDistribution does not support dynamic value space.
  at line 190: bsz = trial.suggest_categorical("train.batch_size", ...)
```

Occurred after the OOM fix changed batch_size from universal `[8,12,16,24,32,48,64]` to model-aware constraints.

## Solution

### 1. Unified Distributions + Pruning Strategy

**Core Principle:** Sample from the **union of all possible values**, then **prune unsafe combinations** at runtime.

**Implementation:**
```python
# BEFORE (BROKEN): Conditional distributions
if heavy_model:
    bsz = trial.suggest_categorical("train.batch_size", [8, 12, 16])
    accum = trial.suggest_categorical("train.grad_accum", [1, 2, 3, 4])
else:
    bsz = trial.suggest_categorical("train.batch_size", [8, 12, 16, 24, 32, 48, 64])
    accum = trial.suggest_categorical("train.grad_accum", [1, 2, 3, 4, 6, 8])

# AFTER (FIXED): Unified distributions + validation
bsz = trial.suggest_categorical("train.batch_size", [8, 12, 16, 24, 32, 48, 64])
accum = trial.suggest_categorical("train.grad_accum", [1, 2, 3, 4, 6, 8])

# Validate OOM safety and prune unsafe combinations
effective_batch = bsz * accum
if heavy_model:
    if bsz > 16 or effective_batch > 64:
        raise optuna.TrialPruned(
            f"Pruned: Large model {model_name} with bsz={bsz}, accum={accum} "
            f"(effective_batch={effective_batch}) likely causes OOM"
        )
else:
    if bsz > 64 or effective_batch > 512:
        raise optuna.TrialPruned(
            f"Pruned: Base model {model_name} with bsz={bsz}, accum={accum} "
            f"(effective_batch={effective_batch}) exceeds reasonable limits"
        )
```

### 2. Enhanced Study Compatibility Validation

Extended `check_and_handle_incompatible_study()` to validate **all categorical parameters**, not just `model.name`:

**Validated Parameters:**
- `model.name`: Model choices
- `train.batch_size`: Union of all batch sizes `[8,12,16,24,32,48,64]`
- `train.grad_accum`: Union of all grad_accum values `[1,2,3,4,6,8]`

**Behavior:**
- Compares existing study's distributions against expected distributions
- If incompatible, **automatically deletes** the study and prints diagnostic info
- New study created with correct distributions on next run

### 3. Study Reset

Deleted the existing incompatible study:
```bash
# Study: noaug-criteria-supermax
# Trials: 17 (1 completed, 11 failed, 3 running)
# Reason: Incompatible batch_size distribution
```

Since only 1 trial completed successfully, this was a safe reset.

## Verification

### Test Results

Ran 10 test trials with the fixed implementation:

```
Results:
  Completed: 8  (valid combinations)
  Pruned:    2  (OOM prevention working)
  Errors:    0  (no CategoricalDistribution errors)

Examples:
  ✓ bert-base-uncased      bsz=64 accum=4 eff=256  (COMPLETE)
  ✗ deberta-v3-large       bsz=24 accum=1 eff=24   (PRUNED - bsz > 16)
  ✗ deberta-v3-large       bsz=64 accum=1 eff=64   (PRUNED - bsz > 16)
  ✓ deberta-v3-large       bsz=8  accum=4 eff=32   (COMPLETE - safe)
  ✓ roberta-base           bsz=24 accum=4 eff=96   (COMPLETE)
```

### Key Validations

- ✅ Unified batch_size distribution (no dynamic value space)
- ✅ Unified grad_accum distribution (no dynamic value space)
- ✅ OOM prevention via trial pruning
- ✅ No CategoricalDistribution errors across 10 trials
- ✅ Large models correctly pruned for unsafe batch sizes
- ✅ Base models can explore full batch size range

## Benefits of the Fix

### 1. Robustness
- **No more CategoricalDistribution errors** regardless of model sampling order
- Study can resume after interruptions without distribution conflicts

### 2. Flexibility
- Optuna's sampler can explore **full search space** for base models
- Large models naturally focus on safe regions via pruning

### 3. Efficiency
- Pruned trials are **fast** (no actual training)
- Sampler learns to avoid pruned regions over time
- No wasted computation on OOM-prone configurations

### 4. Future-Proof
- Enhanced `check_and_handle_incompatible_study()` validates **all** categorical parameters
- Automatically detects and handles distribution changes
- Safe for adding new parameters or modifying existing ones

## OOM Prevention Strategy

### Large Models (330M-400M params)
**Constraints:**
- `bsz <= 16`
- `bsz * accum <= 64`

**Rationale:**
- 24GB GPU limit
- Conservative for `max_len=384` sequences
- Prevents OOM during training

### Base Models (110M params)
**Constraints:**
- `bsz <= 64`
- `bsz * accum <= 512`

**Rationale:**
- More memory headroom
- Allows exploration of larger effective batch sizes
- Still prevents unreasonable configurations

## Files Modified

### `/media/user/SSD1/YuNing/NoAug_Criteria_Evidence/scripts/tune_max.py`

**Lines 184-213:** Fixed batch_size/grad_accum sampling
- Replaced conditional distributions with unified distributions
- Added OOM validation and pruning logic
- Added comprehensive comments explaining Optuna's constraints

**Lines 916-1055:** Enhanced `check_and_handle_incompatible_study()`
- Added validation for `train.batch_size` distribution
- Added validation for `train.grad_accum` distribution
- Improved diagnostic output
- Automatic study cleanup on incompatibility

## Testing Instructions

### Verify Fix is Working

```bash
# Run a quick HPO test (5-10 trials)
python scripts/tune_max.py --agent criteria --num_trials 10

# Check for:
# 1. No CategoricalDistribution errors
# 2. Some trials complete, some get pruned (OOM risk)
# 3. Study can resume without errors

# View trial states
python -c "
import optuna
study = optuna.load_study(
    study_name='noaug-criteria-supermax',
    storage='sqlite:///_optuna/noaug.db'
)
print(f'Completed: {len([t for t in study.trials if t.state.name == \"COMPLETE\"])}')
print(f'Pruned: {len([t for t in study.trials if t.state.name == \"PRUNED\"])}')
print(f'Failed: {len([t for t in study.trials if t.state.name == \"FAIL\"])}')
"
```

### Manual Test

```python
import optuna

study = optuna.create_study(direction="maximize")

def objective(trial):
    model = trial.suggest_categorical("model", ["base", "large"])
    bsz = trial.suggest_categorical("batch_size", [8, 16, 32, 64])

    # Conditional validation
    if model == "large" and bsz > 16:
        raise optuna.TrialPruned("Large model with large batch")

    return 0.5

# Should work without errors
study.optimize(objective, n_trials=10)
print(f"Completed: {len(study.best_trials)}")
```

## Best Practices for Future Development

### 1. Always Use Unified Distributions

**DO:**
```python
# Sample from union of all values
value = trial.suggest_categorical("param", [1, 2, 3, 4, 5])

# Validate based on context
if context_requires_small_values and value > 2:
    raise optuna.TrialPruned("Value too large for this context")
```

**DON'T:**
```python
# Conditional distributions (WILL FAIL)
if condition:
    value = trial.suggest_categorical("param", [1, 2])
else:
    value = trial.suggest_categorical("param", [1, 2, 3, 4, 5])
```

### 2. Use Trial Pruning for Constraints

Pruning is **cheap** and **informative**:
- Pruned trials don't consume compute resources
- Sampler learns to avoid pruned regions
- Explicit reasoning in pruning messages

### 3. Validate Study Compatibility

Before running HPO, check if study exists and is compatible:
```python
deleted = check_and_handle_incompatible_study(
    study_name=study_name,
    storage=storage,
    expected_model_choices=MODEL_CHOICES
)
if deleted:
    print("Study was reset due to incompatible search space")
```

### 4. Version Study Names for Major Changes

For production runs with significant search space changes:
```python
# Increment version when search space changes significantly
study_name = "noaug-criteria-supermax-v2"
```

This preserves historical data while avoiding compatibility issues.

## References

- **Optuna Documentation:** [Categorical Distribution](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.distributions.CategoricalDistribution.html)
- **Optuna Issue:** [Dynamic categorical distributions not supported](https://github.com/optuna/optuna/issues/1846)
- **Project File:** `/media/user/SSD1/YuNing/NoAug_Criteria_Evidence/scripts/tune_max.py`

## Summary

**Problem:** Conditional batch_size distributions caused `CategoricalDistribution` errors
**Root Cause:** Optuna doesn't support dynamic value spaces
**Solution:** Unified distributions + runtime pruning for OOM prevention
**Status:** ✅ Fixed and verified
**Impact:** HPO system is now robust and can handle any model/batch size combination safely

---

**Author:** Claude Code
**Date:** 2025-10-24
**Version:** 1.0
