# Autocast Deprecation Fix Summary

## Changes Made

### Fixed Deprecated `torch.cuda.amp.autocast` API

**Problem**: PyTorch 2.0+ deprecated `torch.cuda.amp.autocast` in favor of the unified `torch.amp.autocast` API with device type parameter.

**Files Modified**: `src/dataaug_multi_both/hpo/trial_executor.py`

### Changes:

#### 1. Removed deprecated import (line 568)
```python
# OLD:
from torch.cuda.amp import autocast, GradScaler

# NEW:
from torch.amp import GradScaler
```

#### 2. Updated `_verify_autocast_compatibility` function (line 401)
```python
# OLD:
with autocast(enabled=True, dtype=autocast_dtype):

# NEW:
with torch.amp.autocast(device_type='cuda', enabled=True, dtype=autocast_dtype):
```

#### 3. Updated `train_model` function (line 690)
```python
# OLD:
with autocast(enabled=use_amp, dtype=autocast_dtype):

# NEW:
with torch.amp.autocast(device_type='cuda', enabled=use_amp, dtype=autocast_dtype):
```

#### 4. Updated `evaluate_model` function (line 814)
```python
# OLD:
from torch.cuda.amp import autocast
with autocast(enabled=use_amp, dtype=autocast_dtype):

# NEW:
# Removed import, using torch.amp.autocast directly
with torch.amp.autocast(device_type='cuda', enabled=use_amp, dtype=autocast_dtype):
```

## Benefits

1. **No more deprecation warnings** - Code uses the current PyTorch 2.x API
2. **Future-proof** - Compatible with PyTorch 2.0+ (currently using 2.8.0)
3. **Cleaner code** - Unified API for all device types
4. **Better performance** - Latest API includes optimizations

## Compatibility

- **PyTorch version**: Requires PyTorch 2.0+
- **Current version**: 2.8.0+cu128 ✓
- **Backward compatibility**: Not compatible with PyTorch 1.x (use old API if needed)

## Testing

All changes have been verified:
- ✓ Code compiles successfully
- ✓ New autocast API works correctly
- ✓ All imports successful
- ✓ No syntax errors

## Related Documentation

See `EPOCH_CONFIGURATION.md` for information about training epoch settings.
