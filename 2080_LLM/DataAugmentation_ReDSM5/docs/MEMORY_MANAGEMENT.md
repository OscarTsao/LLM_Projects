# GPU Memory Management for Optuna Hyperparameter Optimization

This document describes the comprehensive memory management solution implemented to prevent CUDA out-of-memory (OOM) errors during Optuna hyperparameter optimization.

## Problem Description

The original issue occurred when training DeBERTa models with memory-intensive hyperparameter combinations:
- **Model**: DeBERTa-base with 3 classifier layers (512, 704, 448 hidden units)
- **Configuration**: batch_size=128, max_seq_length=384, gradient_accumulation=8, eval_batch_multiplier=2
- **Memory Usage**: 29.18 GiB allocated, causing OOM on 32GB GPU
- **Impact**: Entire Optuna optimization crashed instead of gracefully handling the error

## Solution Components

### 1. Memory Estimation and Safety Checks

**File**: `src/utils/memory_utils.py`

- **`estimate_model_memory()`**: Estimates GPU memory requirements based on model type, batch size, sequence length, and architecture
- **`is_configuration_memory_safe()`**: Checks if a configuration will fit in available GPU memory
- **Memory factors considered**:
  - Base model weights (DeBERTa: 0.6GB, BERT/RoBERTa: 0.5GB)
  - Activation memory (scales with batch size and sequence length)
  - Gradient memory (same size as model weights)
  - Optimizer states (2x model weights for AdamW)
  - 20% safety margin

### 2. Enhanced Optuna Objective Function

**File**: `src/training/train_optuna.py`

**Key improvements**:
- **Pre-training memory checks**: Configurations are validated before training starts
- **Model-specific constraints**: DeBERTa gets more conservative settings
- **Comprehensive error handling**: OOM errors are caught and logged instead of crashing
- **Memory monitoring**: GPU memory usage is logged at each stage

**Model-specific constraints**:
```python
if "deberta" in model_choice:
    # More conservative settings for DeBERTa
    batch_size_options = [8, 16, 32, 64]
    max_seq_options = [128, 256, 384]
    max_classifier_layers = 2
else:
    # Standard settings for BERT/RoBERTa
    batch_size_options = [8, 16, 32, 64, 128]
    max_seq_options = [128, 256, 384, 512]
    max_classifier_layers = 3
```

### 3. Memory Monitoring and Cleanup

**Features**:
- **`MemoryMonitor` context manager**: Tracks memory usage during training phases
- **Automatic cleanup**: GPU memory is cleared after each trial
- **Detailed logging**: Memory allocation, utilization, and changes are logged

### 4. Error Handling Strategy

**OOM Error Handling**:
```python
except torch.cuda.OutOfMemoryError as e:
    logger.error(f"Trial {trial.number}: CUDA OOM Error - {str(e)}")
    log_memory_usage("OOM Error", trial.number)
    clear_gpu_memory()
    trial.set_user_attr("oom_error", True)
    return float("-inf")  # Discourage similar configurations
```

**Benefits**:
- Optimization continues after OOM errors
- Failed configurations are marked and avoided
- Memory is cleaned up to prevent cascading failures

### 5. Memory-Safe Model Configurations

**File**: `conf/model/deberta_base_memory_safe.yaml`

Conservative DeBERTa configuration:
- Batch size: 16 (reduced from 32)
- Sequence length: 256 (conservative)
- No classifier layers initially
- Gradient checkpointing enabled
- Mixed precision disabled (DeBERTa compatibility)

## Usage

### Running Memory-Safe Optuna Optimization

```bash
make train-optuna
```

The enhanced script automatically:
1. Checks GPU memory availability
2. Applies model-specific constraints
3. Validates configurations before training
4. Handles OOM errors gracefully
5. Logs detailed memory usage

### Testing Memory Estimation

```bash
python scripts/test_memory_estimation.py
```

This script:
- Shows current GPU memory status
- Tests various configurations for safety
- Suggests maximum safe batch sizes for different sequence lengths

### Memory Monitoring During Training

The system automatically logs memory usage at key stages:
```
Trial 42 - Trial Start: Allocated: 0.12GB (0.4%), Reserved: 0.50GB, Free: 31.38GB, Total: 31.50GB
Trial 42 - Training Start: Allocated: 2.45GB (7.8%), Reserved: 2.50GB, Free: 29.00GB, Total: 31.50GB
Trial 42 - Training End: Memory change: +1.23GB, Current: 3.68GB
Trial 42 - Trial Success: Allocated: 0.15GB (0.5%), Reserved: 0.50GB, Free: 31.35GB, Total: 31.50GB
```

## Configuration Guidelines

### DeBERTa Models
- **Batch size**: 8-32 (avoid 64+ with long sequences)
- **Sequence length**: 128-256 (avoid 384+ with large batches)
- **Classifier layers**: 0-2 (avoid 3+ layers)
- **Gradient accumulation**: 2-4 (effective batch size consideration)

### BERT/RoBERTa Models
- **Batch size**: 8-64 (can go higher with shorter sequences)
- **Sequence length**: 128-512 (more flexible)
- **Classifier layers**: 0-3 (less memory overhead)
- **Gradient accumulation**: 1-8 (more flexible)

## Monitoring and Debugging

### Memory Usage Logs
- Check logs for memory allocation patterns
- Look for "OOM Error" entries to identify problematic configurations
- Monitor "Memory change" values to track memory leaks

### MLflow Tracking
The system logs study statistics to MLflow:
- `completed_trials`: Successfully completed trials
- `pruned_trials`: Trials pruned by Optuna
- `failed_trials`: Trials that failed due to errors
- `oom_trials`: Trials that failed due to OOM errors

### Failed Trial Analysis
Failed trials include detailed error information:
```python
trial.user_attrs = {
    "oom_error": True,
    "error_message": "CUDA out of memory. Tried to allocate 2.00 GiB...",
    "error_traceback": "..."
}
```

## Best Practices

1. **Start Conservative**: Begin with smaller batch sizes and sequence lengths
2. **Monitor Memory**: Watch memory logs during optimization
3. **Use Memory-Safe Configs**: Prefer the `*_memory_safe.yaml` configurations for DeBERTa
4. **Gradual Scaling**: Increase memory-intensive parameters gradually
5. **Regular Cleanup**: The system automatically cleans memory, but monitor for leaks

## Troubleshooting

### Still Getting OOM Errors?
1. Check if memory estimation is accurate for your specific setup
2. Reduce the safety buffer in `is_configuration_memory_safe()`
3. Add more aggressive constraints for problematic models
4. Consider using gradient checkpointing for all models

### Memory Leaks?
1. Check for models not being properly deleted
2. Ensure all tensors are moved to CPU before deletion
3. Monitor memory usage patterns across trials
4. Add explicit `del model` statements if needed

### Poor Optimization Performance?
1. The conservative constraints may limit exploration
2. Consider relaxing constraints after validating memory safety
3. Use the memory estimation script to find optimal boundaries
4. Balance memory safety with hyperparameter search space
