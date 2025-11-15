# Training Infrastructure Guide

This guide covers the complete training infrastructure for PSY Agents NO-AUG project, including reproducibility settings, hardware optimizations, and best practices.

## Table of Contents

1. [Overview](#overview)
2. [Architecture Support](#architecture-support)
3. [Reproducibility](#reproducibility)
4. [Hardware Optimization](#hardware-optimization)
5. [Training Scripts](#training-scripts)
6. [HPO Integration](#hpo-integration)
7. [Configuration](#configuration)
8. [Best Practices](#best-practices)

## Overview

The training infrastructure provides:
- **Full Reproducibility**: Deterministic training with seed control
- **Hardware Optimization**: Mixed precision (AMP), optimized DataLoader, GPU settings
- **Early Stopping**: Monitor validation metrics and stop when no improvement
- **MLflow Tracking**: Experiment tracking and model versioning
- **HPO Integration**: Load and retrain with best hyperparameters
- **Modular Design**: Separate scripts per architecture for maintainability

## Architecture Support

### Implemented
- ✅ **Criteria**: Binary classification (sentence-level status prediction)
  - `scripts/train_criteria.py` - Training script
  - `scripts/eval_criteria.py` - Evaluation script

### To Be Implemented
- ⏳ **Evidence**: Span extraction (start/end position prediction)
  - Template: Use `train_criteria.py` as reference
  - Key differences: Span prediction head, different loss function

- ⏳ **Joint**: Multi-task learning (criteria + evidence)
  - Template: Combine Criteria and Evidence approaches
  - Key differences: Two encoders, multi-task loss

## Reproducibility

### Seed Control

All scripts use the enhanced `set_seed()` function in `src/psy_agents_noaug/utils/reproducibility.py`:

```python
set_seed(
    seed=42,
    deterministic=True,      # Full reproducibility (slower)
    cudnn_benchmark=False    # Deterministic algorithm selection
)
```

**Settings:**
- `deterministic=True` + `cudnn_benchmark=False`: Full reproducibility (slower)
- `deterministic=False` + `cudnn_benchmark=True`: Maximum speed (non-deterministic)

**What it controls:**
- Python `random` module
- NumPy random state
- PyTorch CPU random state
- PyTorch CUDA random state (all devices)
- CuDNN deterministic algorithms
- `torch.use_deterministic_algorithms()` for full determinism
- `PYTHONHASHSEED` environment variable

### Trade-offs

**Reproducible Mode** (recommended for research):
```yaml
training:
  deterministic: true
  cudnn_benchmark: false
```
- ✅ Exact same results every run
- ❌ 10-30% slower training

**Performance Mode** (for production):
```yaml
training:
  deterministic: false
  cudnn_benchmark: true
```
- ✅ Maximum training speed
- ❌ Results vary slightly between runs

## Hardware Optimization

### Mixed Precision Training (AMP)

**Float16 vs BFloat16:**

```yaml
amp:
  enabled: true
  dtype: "float16"  # or "bfloat16"
```

**Choose based on GPU:**
- **Ampere+ (RTX 30xx, A100, H100)**: Use `bfloat16`
  - More stable than float16
  - No gradient scaling needed
  - Same range as float32
- **Older GPUs (V100, RTX 20xx)**: Use `float16`
  - Requires gradient scaling
  - Risk of underflow
  - Use GradScaler (automatically handled)

**Benefits:**
- 2-3x faster training
- ~2x less GPU memory
- Minimal accuracy loss

### DataLoader Optimization

Based on 2025 best practices:

```yaml
# Optimal settings for GPU training
num_workers: 8                # Start with 2x CPU cores per GPU
pin_memory: true              # Always true for GPU (faster transfer)
persistent_workers: true      # Keep workers alive between epochs
prefetch_factor: 2            # Batches prefetched per worker
```

**Tuning num_workers:**
1. Start with `2 × CPU cores per GPU`
2. Monitor GPU utilization (use `nvidia-smi`)
3. Increase if GPU utilization < 90%
4. Decrease if CPU is bottlenecked
5. Set to 0 if debugging (disables multiprocessing)

**Auto-detection:**
The `get_optimal_dataloader_kwargs()` function automatically configures optimal settings based on your hardware.

### GPU Settings

**Device Selection:**
```python
device = get_device(prefer_cuda=True, device_id=0)
```

This automatically:
- Detects GPU compute capability
- Reports TF32 and BFloat16 support
- Shows GPU memory
- Prints CUDA version

**TF32 Mode (Ampere+):**
TF32 is automatically enabled on Ampere and later GPUs:
- Near FP32 accuracy with FP16 speed
- No code changes needed
- Automatically used by PyTorch

## Training Scripts

### Train Criteria Architecture

**Basic training:**
```bash
python scripts/train_criteria.py
```

**With custom config:**
```bash
python scripts/train_criteria.py \
    training.epochs=20 \
    training.train_batch_size=32 \
    training.learning_rate=3e-5
```

**With best HPO config:**
```bash
python scripts/train_criteria.py \
    best_config=outputs/hpo_stage2/best_config.yaml
```

**With different model:**
```bash
python scripts/train_criteria.py \
    model.pretrained_model=roberta-base
```

### Evaluate Criteria Model

**Evaluate best checkpoint:**
```bash
python scripts/eval_criteria.py \
    checkpoint=outputs/checkpoints/best_checkpoint.pt
```

**Evaluate on custom dataset:**
```bash
python scripts/eval_criteria.py \
    checkpoint=path/to/checkpoint.pt \
    dataset.path=data/custom.csv
```

**Output:**
- Accuracy, F1, Precision, Recall, AUROC
- Confusion matrix
- Classification report
- Results saved to JSON

### Train with Best HPO Config

The `train_best.py` script routes to architecture-specific scripts:

```bash
# Train Criteria with best HPO config
python scripts/train_best.py \
    task=criteria \
    best_config=outputs/hpo_stage2/best_config.yaml

# Train Evidence with best HPO config
python scripts/train_best.py \
    task=evidence \
    best_config=outputs/hpo_stage2/best_config.yaml

# Train Joint with best HPO config
python scripts/train_best.py \
    task=joint \
    best_config=outputs/hpo_stage2/best_config.yaml
```

## HPO Integration

### 1. Run HPO Study

```bash
python scripts/run_hpo_stage.py \
    hpo=stage2_fine \
    task=criteria \
    model=roberta_base
```

This saves:
- `outputs/hpo_stage2/best_config.yaml` - Best hyperparameters
- `outputs/hpo_stage2/trials_history.json` - All trial results
- `outputs/hpo_stage2/study.pkl` - Optuna study object

### 2. Retrain with Best Config

```bash
python scripts/train_best.py \
    task=criteria \
    best_config=outputs/hpo_stage2/best_config.yaml
```

This:
1. Loads best hyperparameters from HPO
2. Merges with base config
3. Trains model with optimal settings
4. Evaluates on test set
5. Saves checkpoint and metrics

### 3. Evaluate Best Model

```bash
python scripts/eval_criteria.py \
    checkpoint=outputs/checkpoints/best_checkpoint.pt
```

## Configuration

### Training Configs

**Default Config** (`configs/training/default.yaml`):
- Balanced settings for most use cases
- Reproducibility enabled
- Moderate hardware optimization

**Optimized Config** (`configs/training/optimized.yaml`):
- Maximum performance settings
- Comprehensive documentation
- GPU-specific recommendations

### Key Parameters

**Training:**
```yaml
num_epochs: 10                    # Training epochs
batch_size: 16                    # Training batch size
eval_batch_size: 32               # Evaluation batch size
learning_rate: 2.0e-5             # Learning rate
weight_decay: 0.01                # L2 regularization
gradient_clip: 1.0                # Gradient clipping
gradient_accumulation_steps: 1    # Gradient accumulation
```

**Optimizer:**
```yaml
optimizer:
  name: "adamw"                   # AdamW optimizer
  betas: [0.9, 0.999]            # Adam betas
  eps: 1.0e-8                    # Epsilon for numerical stability
```

**Scheduler:**
```yaml
scheduler:
  type: "cosine"                  # cosine, linear, constant
  warmup_ratio: 0.06              # 6% warmup steps
```

**Early Stopping:**
```yaml
early_stopping:
  metric: "val_f1_macro"          # Metric to monitor
  mode: "max"                     # max or min
  patience: 3                     # Epochs without improvement
  min_delta: 0.0001              # Minimum improvement threshold
```

## Best Practices

### For Research (Reproducibility)

```yaml
training:
  seed: 42
  deterministic: true
  cudnn_benchmark: false

amp:
  enabled: true
  dtype: "float16"  # or bfloat16 for Ampere+

num_workers: 4  # Lower for reproducibility
```

### For Production (Speed)

```yaml
training:
  seed: 42
  deterministic: false
  cudnn_benchmark: true

amp:
  enabled: true
  dtype: "bfloat16"  # if available

num_workers: 8  # Higher for speed
persistent_workers: true
```

### GPU Selection Guide

**For Ampere+ GPUs (RTX 30xx, A100, H100):**
- Use `dtype: "bfloat16"`
- Enable `cudnn_benchmark: true`
- Use higher `num_workers`

**For Older GPUs (V100, RTX 20xx, GTX 1080):**
- Use `dtype: "float16"`
- Monitor for NaN/Inf with gradient scaling
- May need lower `num_workers`

**For CPU Training:**
- Set `num_workers: 4` (or less)
- Disable AMP: `amp.enabled: false`
- Expect 10-50x slower training

### Memory Optimization

If you encounter OOM (Out of Memory) errors:

1. **Reduce batch size:**
   ```yaml
   training:
     train_batch_size: 8  # Reduce from 16
     eval_batch_size: 16  # Reduce from 32
   ```

2. **Use gradient accumulation:**
   ```yaml
   training:
     train_batch_size: 8
     gradient_accumulation_steps: 2  # Effective batch size = 8 × 2 = 16
   ```

3. **Enable gradient checkpointing:**
   ```yaml
   training:
     gradient_checkpointing: true  # Trade compute for memory
   ```

4. **Use mixed precision:**
   ```yaml
   amp:
     enabled: true  # Reduces memory by ~2x
   ```

## Monitoring Training

### MLflow Tracking

All training runs are automatically tracked in MLflow:

```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# View at http://localhost:5000
```

**Logged Metrics:**
- Training loss (per step)
- Validation metrics (per epoch)
- Learning rate schedule
- System info (GPU, CUDA version, etc.)

### GPU Monitoring

During training, monitor GPU utilization:

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or use
nvtop  # More user-friendly
```

**Target:**
- GPU Utilization: > 90%
- GPU Memory: 70-90% (leave headroom)

**If GPU utilization < 90%:**
- Increase `num_workers`
- Increase `batch_size`
- Check if CPU is bottleneck

## Troubleshooting

### NaN/Inf in Loss

**Causes:**
- Learning rate too high
- Mixed precision instability
- Gradient explosion

**Solutions:**
1. Lower learning rate: `learning_rate: 1e-5`
2. Use BFloat16 instead of Float16
3. Increase gradient clipping: `gradient_clip: 0.5`
4. Check data for anomalies

### Slow Training

**Causes:**
- Inefficient DataLoader
- CPU bottleneck
- Synchronous operations

**Solutions:**
1. Increase `num_workers`
2. Enable `persistent_workers: true`
3. Use `pin_memory: true`
4. Profile with PyTorch Profiler

### Reproducibility Issues

**Causes:**
- Non-deterministic operations
- Random initialization
- Hardware differences

**Solutions:**
1. Enable full determinism: `deterministic: true`
2. Set `cudnn_benchmark: false`
3. Use `torch.use_deterministic_algorithms(True)`
4. Note: Some ops may still be non-deterministic

## Next Steps

### Implement Evidence and Joint Architectures

1. **Copy Criteria template:**
   ```bash
   cp scripts/train_criteria.py scripts/train_evidence.py
   cp scripts/eval_criteria.py scripts/eval_evidence.py
   ```

2. **Adapt for Evidence:**
   - Change dataset: `EvidenceDataset`
   - Change model: Span prediction head
   - Change loss: Span extraction loss (cross-entropy for start/end)
   - Change metrics: Exact match, F1 (span-level)

3. **Adapt for Joint:**
   - Change dataset: Joint dataset (combines both)
   - Change model: Two encoders + two heads
   - Change loss: Multi-task loss (weighted sum)
   - Change metrics: Both classification and span metrics

## References

- [PyTorch Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [PyTorch DataLoader Best Practices](https://pytorch.org/docs/stable/data.html)
- [PyTorch Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Optuna HPO Documentation](https://optuna.readthedocs.io/)

## Support

For issues or questions:
1. Check this guide first
2. Review `docs/TESTING.md` for validation
3. Check `docs/CLI_AND_MAKEFILE_GUIDE.md` for commands
4. Raise an issue with detailed error logs
