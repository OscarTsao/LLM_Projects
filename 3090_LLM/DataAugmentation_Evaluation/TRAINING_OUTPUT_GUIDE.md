# Training Output Directory Guide

This guide explains where training results are saved and how to find them.

## Directory Structure

Every training run is saved in a **unique timestamped directory** that will never be overridden:

```
outputs/
├── YYYY-MM-DD/                    # Date of training run
│   ├── HH-MM-SS-microseconds/     # Exact time of training run (with microseconds)
│   │   ├── best/                  # Best model checkpoint
│   │   │   ├── model.pt          # Model weights
│   │   │   └── config.yaml       # Training configuration
│   │   ├── .hydra/                # Hydra configuration snapshots
│   │   │   ├── config.yaml       # Full resolved config
│   │   │   ├── overrides.yaml    # Command-line overrides
│   │   │   └── hydra.yaml        # Hydra runtime config
│   │   └── train_*.log            # Training logs
```

## Examples

### Standard Training
```bash
make train
# Results saved to: outputs/2025-10-14/15-23-45-123456/
```

### Multi-Agent Training
```bash
make train-evidence
# Results saved to: outputs/2025-10-14/15-24-10-234567/

make train-criteria
# Results saved to: outputs/2025-10-14/15-24-35-345678/

make train-joint
# Results saved to: outputs/2025-10-14/15-25-00-456789/
```

### Best Config with Different Encoder
```bash
python -m src.training.train --config-name=best_config model.pretrained_model_name=FacebookAI/roberta-base
# Results saved to: outputs/2025-10-14/15-26-15-567890/
```

## Finding Your Results

### Method 1: Check the Training Logs
When training starts, the output directory is logged:
```
[INFO] Results will be saved to: /path/to/outputs/2025-10-14/15-23-45-123456
[INFO] Model checkpoint will be saved to: /path/to/outputs/2025-10-14/15-23-45-123456/best/model.pt
```

### Method 2: List Recent Runs
```bash
# Find the most recent training run
ls -lt outputs/$(date +%Y-%m-%d)/ | head -5

# Find all training runs today
find outputs/$(date +%Y-%m-%d) -name "model.pt"

# Find all training runs in the last week
find outputs/ -name "model.pt" -mtime -7
```

### Method 3: Search by Training Mode
```bash
# Find evidence agent runs
find outputs/ -name "train_evidence.log"

# Find criteria agent runs
find outputs/ -name "train_criteria.log"

# Find joint training runs
find outputs/ -name "train_joint.log"
```

## Key Features

1. **Never Overridden**: Each run gets a unique timestamp with microseconds, preventing collisions
2. **Organized by Date**: Easy to find runs from specific days
3. **Complete History**: All configs and checkpoints are preserved
4. **MLflow Integration**: All runs are also tracked in MLflow for comparison

## MLflow Tracking

In addition to file-based outputs, all training runs are tracked in MLflow:

```bash
# View training runs in MLflow UI
make mlflow-ui
# Then open http://localhost:5000
```

MLflow tracks:
- All hyperparameters
- Training/validation metrics over time
- Model artifacts
- Training mode and agent type tags

## Configuration Snapshots

Each run preserves its complete configuration in `.hydra/`:
- `config.yaml` - Full resolved configuration with all defaults
- `overrides.yaml` - Command-line overrides you specified
- `hydra.yaml` - Hydra runtime information

This allows you to exactly reproduce any training run.

## Best Practices

1. **Don't rely on `outputs/train/`**: This directory is from old configs and may not be used
2. **Use timestamps to identify runs**: The directory name tells you exactly when it ran
3. **Check MLflow for comparisons**: When comparing multiple runs, use MLflow UI
4. **Archive old runs**: Move old date directories to `outputs/archive/` to keep things organized

## Troubleshooting

### Model checkpoint not saved?
Check that:
1. Training completed successfully (didn't crash/timeout)
2. Early stopping was triggered or training reached the end
3. The `best/` subdirectory exists in your output directory

### Can't find recent run?
```bash
# Find the 5 most recently modified directories
find outputs/ -type d -name "*-*-*-*" | xargs ls -ltd | head -5
```

### Need to resume training?
Use the model checkpoint:
```bash
python -m src.training.train resume=true checkpoint=outputs/2025-10-14/15-23-45-123456/best/model.pt
```
