# Data Augmentation Training Guide

This guide explains how to train models with different augmentation methods using the best configuration.

## Quick Start

### Option 1: Train with Existing Augmentation Methods

Train with the 4 existing augmentation configurations:

```bash
# No augmentation (original data only)
python -m src.training.train --config-name=best_config dataset=original

# NLPAug only
python -m src.training.train --config-name=best_config dataset=original_nlpaug

# TextAttack only
python -m src.training.train --config-name=best_config dataset=original_textattack

# Hybrid (NLPAug + TextAttack) - This is what best_config uses
python -m src.training.train --config-name=best_config dataset=original_hybrid
```

### Option 2: Automated Training Script

Use the provided script to train all methods automatically:

```bash
# Train all augmentation methods with best encoder (DeBERTa)
./scripts/train_all_augmentations.sh

# Train all encoders with hybrid augmentation
./scripts/train_all_augmentations.sh --mode encoders

# Train all combinations (12 total: 4 augmentations × 3 encoders)
./scripts/train_all_augmentations.sh --mode full --encoder all

# Show help
./scripts/train_all_augmentations.sh --help
```

## TextAttack Methods

### Available TextAttack Augmenters

The current implementation uses **EmbeddingAugmenter** (word embedding-based replacement). To use other TextAttack methods:

1. **EmbeddingAugmenter** (current): Word replacement using word embeddings
2. **WordNetAugmenter**: Synonym replacement using WordNet
3. **EasyDataAugmenter (EDA)**: Synonym replacement, insertion, swap, deletion
4. **CharSwapAugmenter**: Character-level perturbations
5. **CheckListAugmenter**: Systematic contrast sets
6. **CLAREAugmenter**: Contextualized MLM-based augmentation

### Using Different TextAttack Methods

#### Method 1: Use the Python Script (Recommended)

```bash
# List available TextAttack methods
python scripts/train_textattack_methods.py --list-methods

# Train with specific TextAttack method
python scripts/train_textattack_methods.py --methods embedding --encoders deberta

# Train with multiple methods
python scripts/train_textattack_methods.py --methods embedding wordnet eda

# Train all TextAttack methods with all encoders
python scripts/train_textattack_methods.py --methods all --encoders all

# Only generate datasets without training
python scripts/train_textattack_methods.py --generation-only

# Skip generation and use existing datasets
python scripts/train_textattack_methods.py --skip-generation
```

#### Method 2: Manual Training

1. First, generate augmented dataset with your preferred method by modifying `src/augmentation/textattack_pipeline.py`

2. Then train:
```bash
python -m src.training.train \
  --config-name=best_config \
  dataset=original_textattack \
  notes="TextAttack with WordNetAugmenter"
```

## Training with Different Encoders

### Single Encoder Training

```bash
# DeBERTa (best performance from HPO)
python -m src.training.train --config-name=best_config \
  model.pretrained_model_name=microsoft/deberta-base

# RoBERTa
python -m src.training.train --config-name=best_config \
  model.pretrained_model_name=FacebookAI/roberta-base

# BERT
python -m src.training.train --config-name=best_config \
  model.pretrained_model_name=google-bert/bert-base-uncased
```

### All Encoders with Specific Augmentation

```bash
for encoder in microsoft/deberta-base FacebookAI/roberta-base google-bert/bert-base-uncased; do
  python -m src.training.train --config-name=best_config \
    dataset=original_hybrid \
    model.pretrained_model_name=$encoder \
    notes="Hybrid augmentation with $(basename $encoder)"
done
```

## Complete Comparison Matrix

To train all combinations for a complete comparison:

```bash
# 4 augmentation methods × 3 encoders = 12 training runs
for dataset in original original_nlpaug original_textattack original_hybrid; do
  for encoder in microsoft/deberta-base FacebookAI/roberta-base google-bert/bert-base-uncased; do
    python -m src.training.train --config-name=best_config \
      dataset=$dataset \
      model.pretrained_model_name=$encoder \
      notes="$dataset with $(basename $encoder)"
  done
done
```

Or use the automated script:
```bash
./scripts/train_all_augmentations.sh --mode full --encoder all
```

## Results Organization

Each training run saves to a unique timestamped directory:

```
outputs/
├── 2025-10-14/
│   ├── 15-30-45-123456/    # No augmentation + DeBERTa
│   ├── 15-35-10-234567/    # NLPAug + DeBERTa
│   ├── 15-40-25-345678/    # TextAttack + DeBERTa
│   └── 15-45-50-456789/    # Hybrid + DeBERTa
```

## Comparing Results

### Method 1: MLflow UI (Recommended)

```bash
make mlflow-ui
# Open http://localhost:5000
```

In MLflow:
1. Filter runs by tags or parameters
2. Compare metrics side-by-side
3. Visualize training curves
4. Export results to CSV

### Method 2: File System

```bash
# Find all runs from today
ls -lt outputs/$(date +%Y-%m-%d)/

# Find best models
find outputs/$(date +%Y-%m-%d)/ -name "model.pt" -exec ls -lh {} \;

# Compare test metrics
for dir in outputs/$(date +%Y-%m-%d)/*/; do
  echo "=== $(basename $dir) ==="
  cat "$dir/test_metrics.json" 2>/dev/null || echo "No test metrics"
done
```

### Method 3: Python Analysis

```python
import json
from pathlib import Path
import pandas as pd

# Collect all test metrics
results = []
for metrics_file in Path("outputs").rglob("test_metrics.json"):
    with open(metrics_file) as f:
        metrics = json.load(f)
        metrics['run_dir'] = str(metrics_file.parent)
        results.append(metrics)

# Create comparison DataFrame
df = pd.DataFrame(results)
print(df.sort_values('roc_auc', ascending=False))
```

## Configuration Details

### Best Config Parameters

The `best_config.yaml` uses these optimized hyperparameters from HPO:

- **Model**: DeBERTa-base
- **Dataset**: Hybrid augmentation (NLPAug + TextAttack)
- **Learning rate**: 2.78e-5
- **Batch size**: 8
- **Dropout**: 0.215
- **Max sequence length**: 384
- **Early stopping patience**: 20
- **Achieved ROC-AUC**: 0.943

### Overriding Parameters

You can override any parameter:

```bash
python -m src.training.train --config-name=best_config \
  dataset=original_textattack \
  model.pretrained_model_name=FacebookAI/roberta-base \
  model.batch_size=16 \
  model.num_epochs=50 \
  early_stopping_patience=10 \
  notes="Custom configuration test"
```

## Expected Training Times

Approximate times on RTX 5090:

- **No augmentation**: ~30-45 minutes
- **With augmentation**: ~60-90 minutes
- **Full comparison (12 runs)**: ~12-18 hours

## Tips and Best Practices

1. **Start with a subset**: Test with 1-2 methods first before running all combinations

2. **Monitor in real-time**: Open MLflow UI before starting to watch progress

3. **Use screen/tmux**: For long-running jobs, use screen or tmux to avoid interruption

4. **Check GPU memory**: Run `nvidia-smi` to ensure GPU isn't overloaded

5. **Disk space**: Ensure sufficient space (~10GB per run for models + logs)

6. **Resume after failure**: Each run is independent, so you can resume by running only the failed combinations

## Troubleshooting

### Issue: Out of memory
```bash
# Reduce batch size
python -m src.training.train --config-name=best_config \
  model.batch_size=4 \
  model.gradient_accumulation_steps=2
```

### Issue: Training too slow
```bash
# Reduce number of workers
python -m src.training.train --config-name=best_config \
  dataloader.num_workers=4
```

### Issue: Dataset not found
```bash
# Check dataset paths
ls -la Data/Augmentation/

# Regenerate if needed
make augment-all
```

## Next Steps

After training:
1. Compare results in MLflow
2. Select the best augmentation + encoder combination
3. Document your findings
4. Use the best model for downstream tasks
