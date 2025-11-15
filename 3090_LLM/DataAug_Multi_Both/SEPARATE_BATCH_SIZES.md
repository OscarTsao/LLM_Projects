# Separate Batch Sizes for Training and Evaluation

## Feature Overview

The system now supports **separate batch sizes for training and evaluation**. This is beneficial because:

1. **Memory Efficiency**: Evaluation doesn't need gradient computation, so can use larger batches
2. **Faster Inference**: Larger eval batches improve throughput during validation/testing
3. **Common Practice**: Many frameworks (PyTorch Lightning, HuggingFace) support this

## How It Works

### Automatic Fallback Logic

The batch size resolution follows this priority:

```python
train_batch_size = config.get("train_batch_size", config.get("batch_size", 16))
eval_batch_size = config.get("eval_batch_size", config.get("batch_size", 16))
```

**Priority Order:**
1. If `train_batch_size` specified → use it for training
2. Otherwise, if `batch_size` specified → use it for training
3. Otherwise → default to 16

Same logic for `eval_batch_size`.

### Configuration Options

#### Option 1: Same Batch Size (Default)
```yaml
batch_size: 16  # Used for both training and evaluation
```

#### Option 2: Separate Batch Sizes
```yaml
train_batch_size: 16
eval_batch_size: 32  # Can be larger since no gradients
```

#### Option 3: Mixed (backward compatible)
```yaml
batch_size: 16        # Fallback default
eval_batch_size: 32   # Override for evaluation only
```

## HPO Integration

### Basic Usage (Same Batch Size)

HPO will search the batch_size space and use it for both:
```python
# In HPO config - this is the default behavior
batch_size: [4, 8, 16, 32, 64, 128, 256]
```

### Advanced: Separate Batch Sizes in HPO

Enable separate batch size optimization:
```yaml
# In HPO config or search space
separate_batch_sizes: true
```

When enabled, HPO will:
1. Search for optimal `batch_size` (training batch size)
2. Search for `eval_batch_multiplier` [1, 2, 3, 4]
3. Set `eval_batch_size = batch_size * multiplier` (capped at 256)

**Example trial:**
- `batch_size = 16` (training)
- `eval_batch_multiplier = 2`
- → `eval_batch_size = 32` (evaluation)

## Memory Considerations

### Training vs Evaluation Memory

**Training memory includes:**
- Forward pass activations
- Gradients for all parameters
- Optimizer states (momentum, etc.)

**Evaluation memory includes:**
- Forward pass activations only

**Typical ratio:** Evaluation uses ~40-60% of training memory

### Safe Multipliers

| Training Batch | Safe Eval Multiplier | Eval Batch | Memory Increase |
|----------------|---------------------|------------|-----------------|
| 16             | 2x                  | 32         | Minimal         |
| 16             | 3x                  | 48         | Low             |
| 16             | 4x                  | 64         | Moderate        |
| 32             | 2x                  | 64         | Low             |
| 32             | 4x                  | 128        | Moderate        |
| 64             | 2x                  | 128        | Low             |
| 64             | 4x                  | 256        | Moderate        |

## Implementation Details

### Code Changes

**File: `src/dataaug_multi_both/hpo/trial_executor.py`**

```python
# Support separate batch sizes
train_batch_size = config.get("train_batch_size", config.get("batch_size", 16))
eval_batch_size = config.get("eval_batch_size", config.get("batch_size", 16))

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, ...)
val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, ...)
```

**File: `src/dataaug_multi_both/hpo/search_space.py`**

```python
if self.config.get("separate_batch_sizes", False):
    eval_batch_size_multiplier = trial.suggest_categorical("eval_batch_multiplier", [1, 2, 3, 4])
    params["eval_batch_size"] = min(params["batch_size"] * eval_batch_size_multiplier, 256)
    params["train_batch_size"] = params["batch_size"]
```

## Examples

### Example 1: Manual Configuration

```yaml
# config.yaml
train_batch_size: 16
eval_batch_size: 48  # 3x training batch
```

### Example 2: HPO with Same Batch Size (Default)

```yaml
# hpo_config.yaml
batch_size: [16, 32, 64]  # Search space
# Both train and eval will use selected batch_size
```

### Example 3: HPO with Separate Batch Sizes

```yaml
# hpo_config.yaml
separate_batch_sizes: true
batch_size: [16, 32, 64]  # Training batch size search space
# eval_batch_size will be automatically set to batch_size * [1,2,3,4]
```

## Benefits

### Performance
- **Faster Validation**: 2-4x larger eval batches → 2-4x faster validation
- **Better GPU Utilization**: Evaluation can max out GPU memory
- **Reduced Wall Time**: Faster validation between epochs

### Flexibility
- **Memory-Constrained Training**: Small training batch + larger eval batch
- **HPO Optimization**: Can optimize both separately
- **Backward Compatible**: Existing configs work unchanged

### Practical Impact

**Before (same batch size 16):**
- Training: 16 samples/batch
- Evaluation: 16 samples/batch
- Validation time: 100%

**After (train=16, eval=64):**
- Training: 16 samples/batch (unchanged)
- Evaluation: 64 samples/batch (4x larger)
- Validation time: ~25% (4x faster!)

## Recommendations

### General Guidelines

1. **Conservative Start**: `eval_batch_size = 2 × train_batch_size`
2. **Monitor Memory**: Watch GPU memory during validation
3. **Test Before HPO**: Verify eval batch size manually first
4. **Cap at 256**: Keep eval_batch_size ≤ 256 (diminishing returns)

### By Scenario

**Scenario 1: Memory Abundant**
```yaml
train_batch_size: 32
eval_batch_size: 128  # 4x training
```

**Scenario 2: Memory Limited**
```yaml
train_batch_size: 8
eval_batch_size: 32   # 4x training (still reasonable)
```

**Scenario 3: HPO Optimization**
```yaml
separate_batch_sizes: true  # Let Optuna find optimal ratio
```

## Files Modified

1. `src/dataaug_multi_both/hpo/trial_executor.py`
   - Added separate train_batch_size and eval_batch_size support
   
2. `src/dataaug_multi_both/hpo/search_space.py`
   - Added optional separate_batch_sizes HPO support
   - Added eval_batch_multiplier search parameter

3. `configs/data/redsm5.yaml`
   - Added documentation for separate batch size options

## Summary

✓ Evaluation can now use different (typically larger) batch size than training  
✓ Backward compatible - existing configs work unchanged  
✓ HPO support with `separate_batch_sizes: true`  
✓ Memory efficient - evaluation uses less memory per sample  
✓ Performance boost - faster validation with larger batches  
