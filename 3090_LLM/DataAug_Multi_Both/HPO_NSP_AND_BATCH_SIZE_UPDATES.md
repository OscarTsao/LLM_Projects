# HPO and Batch Size Updates

## Updates Made

### 1. HPO Uses NSP Format ✓

**Confirmation:** Yes, HPO automatically uses the new NSP input format!

The HPO pipeline creates datasets using `create_pytorch_dataset()` which instantiates `RedSM5Dataset`. All the NSP format changes we made are automatically applied:

```python
# In trial_executor.py (lines 451-458)
train_dataset = create_pytorch_dataset(
    dataset_dict["train"],
    tokenizer=tokenizer,
    input_format=config.get("input_format", "multi_label"),
    max_length=max_length,
    augmentation_prob=config.get("augmentation_prob", 0.0),
    augmentation_methods=config.get("augmentation_methods", [])
)
```

This means:
- ✓ HPO trials use NSP format: `[CLS] post [SEP] criterion(s) [SEP]`
- ✓ Token type IDs are included in training
- ✓ Both binary_pairs and multi_label modes benefit from NSP

### 2. Batch Size Search Space Expanded ✓

**File:** `src/dataaug_multi_both/hpo/search_space.py`

#### Changes:

**Before:**
```python
params["batch_size"] = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
```

**After:**
```python
params["batch_size"] = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64, 128, 256])
```

#### Validation Updated:

**Line 510-511** (validation check):
```python
if params["batch_size"] not in [4, 8, 16, 32, 64, 128, 256]:
    logger.warning(f"Batch size {params['batch_size']} not in standard options")
```

### 3. New Batch Size Options

The HPO search space now includes:
- 4 (original)
- 8 (original)
- 16 (original)
- 32 (original)
- **64** (new)
- **128** (new)
- **256** (new - upper bound)

This allows Optuna to explore:
- Small batches (4-16) for fine-grained updates
- Medium batches (32-64) for balanced training
- Large batches (128-256) for faster throughput and potentially better generalization

## Impact

### Memory Considerations

With larger batch sizes:
- **Batch 64:** ~4x memory vs batch 16
- **Batch 128:** ~8x memory vs batch 16
- **Batch 256:** ~16x memory vs batch 16

**Recommendation:** Ensure GPU has sufficient memory or use gradient accumulation:
```yaml
accumulation_steps: 4  # Effective batch = batch_size * accumulation_steps
```

### Expected Benefits

1. **Faster Training:** Larger batches can improve GPU utilization
2. **Better Generalization:** Larger batches may provide more stable gradients
3. **HPO Exploration:** Optuna can now optimize batch size across wider range

## Verification

To verify NSP format is used in HPO:

```python
# During HPO trial
# Dataset automatically uses NSP format based on input_format config
# For multi_label: [CLS] post [SEP] criterion1 ... criterion9 [SEP]
# For binary_pairs: [CLS] post [SEP] criterion_i [SEP]
```

## Files Modified

1. `src/dataaug_multi_both/hpo/search_space.py`
   - Line 317: Expanded batch_size choices to [4, 8, 16, 32, 64, 128, 256]
   - Line 510: Updated validation to include new batch sizes

## Summary

- ✓ HPO now uses NSP format automatically (no changes needed)
- ✓ Batch size upper bound increased to 256
- ✓ Added batch sizes: 64, 128, 256
- ✓ Validation checks updated
- ✓ Total batch size options: 7 (was 4)
