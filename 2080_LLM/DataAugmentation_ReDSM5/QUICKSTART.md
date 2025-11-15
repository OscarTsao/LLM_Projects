# Quick Start Guide - REDSM5 Augmentation Pipeline

## Installation (5 minutes)

```bash
cd /experiment/YuNing/DataAugmentation_ReDSM5

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .

# Verify setup
python scripts/verify_setup.py
```

## Basic Workflow (Complete Pipeline)

### 1. Prepare Dataset (10-15 minutes)
```bash
# Load REDSM5 from HuggingFace and save as Parquet
python scripts/prepare_redsm5.py --source hub

# This creates:
# - data/redsm5/base/train.parquet
# - data/redsm5/base/val.parquet  
# - data/redsm5/base/test.parquet
```

### 2. Generate Valid Combinations (2-3 minutes)
```bash
# Enumerate all valid combinations up to k=3
python scripts/list_combos.py \
  --k-max 3 \
  --output data/redsm5/combos/valid_combos.json

# Expected output:
# - k=1: 28 combinations
# - k=2: ~300 combinations (capped)
# - k=3: ~1000 combinations (capped)
```

### 3. Generate Augmentation Cache (varies by resources)
```bash
# Pre-generate augmented data for all combinations
python scripts/generate_aug_cache.py \
  --config configs/run.yaml \
  --num-workers 8 \
  --splits train val

# Time estimate: 
# - 28 k=1 combos × 2 splits = 56 files
# - ~300 k=2 combos × 2 splits = ~600 files
# - ~1000 k=3 combos × 2 splits = ~2000 files
# Total: ~2656 parquet files
#
# With 8 workers: ~6-12 hours (depends on augmenter speed)
```

### 4. Run Stage 1 HPO: Augmenter Selection (1-2 days)
```bash
# Search for best augmentation combination
python scripts/run_hpo_stage1.py \
  --config configs/run.yaml \
  --combos data/redsm5/combos/valid_combos.json \
  --trials 100 \
  --study-name my_stage1_study

# Output:
# - Best combo (e.g., ["random_delete", "word_dropout", "mlm_infill_bert"])
# - Best validation F1 score
# - Study saved to results/stage1_my_stage1_study.pkl
```

### 5. Run Stage 2 HPO: Hyperparameter Tuning (12-24 hours)
```bash
# Fine-tune best combo from Stage 1
python scripts/run_hpo_stage2.py \
  --config configs/run.yaml \
  --combo random_delete word_dropout mlm_infill_bert \
  --trials 50 \
  --study-name my_stage2_study

# Output:
# - Best augmentation intensity
# - Best model hyperparameters (learning rate, batch size, etc.)
# - Study saved to results/stage2_my_stage2_study.pkl
```

## Quick Examples

### Example 1: Test a Single Augmenter
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from aug.registry import AugmenterRegistry

# Load registry
registry = AugmenterRegistry()

# List available augmenters
print(registry.list_augmenters(stage="char"))
# ['char_substitute', 'keyboard_error', 'ocr_noise', 'random_delete', 'random_insert', 'random_swap']

# Instantiate augmenter
augmenter = registry.instantiate_augmenter(
    "random_delete",
    params={"aug_char_p": 0.1},
    seed=13
)
```

### Example 2: Create an Augmentation Pipeline
```python
from aug.compose import AugmentationPipeline
import pandas as pd

# Create pipeline
pipeline = AugmentationPipeline(
    combo=["random_delete", "word_dropout"],
    seed=13
)

# Augment a DataFrame
df = pd.DataFrame({
    "evidence_sentence": [
        "This is a test sentence.",
        "Another example text.",
    ]
})

df_augmented = pipeline.augment_dataframe(df, text_field="evidence_sentence")
print(df_augmented)
```

### Example 3: Generate Combinations
```python
from aug.combos import ComboGenerator

# Create generator
generator = ComboGenerator(config_path="configs/run.yaml")

# Generate k=2 combinations
combos_k2 = generator.generate_k_combos(k=2, ordered=True)
print(f"Generated {len(combos_k2)} k=2 combinations")

# Check if combo is valid
combo = ["random_delete", "word_dropout"]
is_valid = generator.is_valid_combo(combo)
print(f"Combo {combo} is valid: {is_valid}")
```

### Example 4: Load Cached Augmented Data
```python
from dataio.parquet_io import ParquetIO
from utils.hashing import generate_cache_filename
from aug.compose import AugmentationPipeline

# Get combo hash
pipeline = AugmentationPipeline(
    combo=["random_delete", "word_dropout"],
    seed=13
)
combo_hash = pipeline.get_combo_hash()

# Load from cache
parquet_io = ParquetIO()
cache_path = f"data/redsm5/combos/{generate_cache_filename(combo_hash, 'train')}"
df = parquet_io.read_dataframe(cache_path)
print(f"Loaded {len(df)} augmented examples")
```

## Configuration Customization

### Adjust k_max
Edit `configs/run.yaml`:
```yaml
combinations:
  k_max: 2  # Change from 3 to 2 for faster experimentation
```

### Change HPO Settings
Edit `configs/run.yaml`:
```yaml
hpo:
  stage1:
    trials: 50  # Reduce from 100 for quick testing
    timeout_hours: 12  # Reduce from 24
```

### Modify Augmenter Parameters
Edit `configs/augmenters_28.yaml`:
```yaml
- name: "random_delete"
  defaults:
    aug_char_p: 0.15  # Change from 0.1
```

## Troubleshooting

### Issue: Augmenter not found
```bash
# Check available augmenters
python -c "from aug.registry import AugmenterRegistry; r = AugmenterRegistry(); print(r.list_augmenters())"
```

### Issue: Cache file not found
```bash
# List cached files
ls data/redsm5/combos/

# Regenerate specific combo
python scripts/generate_aug_cache.py --combos data/redsm5/combos/valid_combos.json
```

### Issue: Out of memory
```bash
# Reduce batch size in configs/run.yaml
training:
  batch_size: 8  # Reduce from 16
```

## Directory Structure Quick Reference

```
.
├── configs/
│   ├── run.yaml              # Main config
│   └── augmenters_28.yaml    # Augmenter specs
├── data/redsm5/
│   ├── base/                 # Original data
│   └── combos/               # Cached augmented data
├── scripts/
│   ├── prepare_redsm5.py     # Step 1
│   ├── list_combos.py        # Step 2
│   ├── generate_aug_cache.py # Step 3
│   ├── run_hpo_stage1.py     # Step 4
│   └── run_hpo_stage2.py     # Step 5
└── src/
    ├── dataio/               # Data loading
    ├── aug/                  # Augmentation
    ├── hpo/                  # HPO
    └── utils/                # Utilities
```

## Time Estimates

| Task | Time | Notes |
|------|------|-------|
| Installation | 5 min | Pip install |
| Prepare dataset | 10-15 min | Download + convert |
| List combos | 2-3 min | Enumeration |
| Generate cache | 6-12 hours | With 8 workers |
| Stage 1 HPO | 1-2 days | 100 trials |
| Stage 2 HPO | 12-24 hours | 50 trials |

## Resource Requirements

### Minimum
- **CPU**: 4 cores
- **RAM**: 16 GB
- **Disk**: 50 GB (for cache)
- **GPU**: Optional (speeds up training)

### Recommended
- **CPU**: 8+ cores
- **RAM**: 32 GB
- **Disk**: 100 GB
- **GPU**: 1x NVIDIA GPU (8GB+ VRAM)

## Getting Help

1. Check documentation:
   - `README.md` - Comprehensive guide
   - `PROJECT_STRUCTURE.md` - Detailed structure
   - `SETUP_SUMMARY.md` - Setup verification

2. Run verification:
   ```bash
   python scripts/verify_setup.py
   ```

3. Check logs:
   ```bash
   # Enable verbose logging
   python scripts/run_hpo_stage1.py --config configs/run.yaml --trials 1
   ```

## Next Steps

After completing the basic workflow:

1. **Analyze Results**: Load study pickle files and analyze trial results
2. **Test Best Model**: Evaluate best combo on test set
3. **Ablation Studies**: Test individual augmenters vs. combinations
4. **Hyperparameter Sensitivity**: Analyze parameter importance

Happy augmenting!
