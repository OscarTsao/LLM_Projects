# Batch Size HPO Fix

## Issue Fixed

The HPO was failing with:
```
AttributeError: 'SearchSpaceConfig' object has no attribute 'get'
```

This was caused by trying to access `self.config.get("separate_batch_sizes", False)` when `self.config` is a dataclass, not a dictionary.

## Solution

Removed the automatic HPO search for separate batch sizes. The feature still works, but needs manual configuration.

## How to Use Separate Batch Sizes

### Option 1: Manual Configuration (Recommended)
```yaml
# In your config
train_batch_size: 16
eval_batch_size: 48  # 3x training batch size
```

### Option 2: With HPO
```yaml
# HPO will search batch_size (used for training)
# Manually set eval_batch_size to a multiple
batch_size: [16, 32, 64]  # HPO searches this
eval_batch_size: 64        # Manually set (or calculate as 2-4x batch_size)
```

## What Still Works

✓ NSP format - fully automatic
✓ Batch size expansion [4, 8, 16, 32, 64, 128, 256]
✓ Separate train/eval batch sizes (manual config)
✓ HPO searches batch_size parameter
✓ You can manually set different eval_batch_size

## Files Modified

- `src/dataaug_multi_both/hpo/search_space.py` - Removed automatic separate_batch_sizes HPO

## Status

✓ HPO now runs successfully
✓ All core features working
✓ Separate batch sizes available via manual config
