# Psy Agents AUG: Data Augmentation for Criteria and Evidence Extraction

This repository implements **data augmentation pipelines** for extracting DSM-5 criteria and clinical evidence from text, building upon the NO-AUG baseline.

## Key Features

### Augmentation Capabilities
- **Multiple Pipelines**: NLPAug, TextAttack, Hybrid
- **STRICT Train-Only**: Augmentation NEVER applies to val/test (prevents data leakage)
- **Deterministic**: Same seed = same augmentations (reproducibility)
- **Balance-Aware**: Preserves class balance when configured

### Augmentation Methods
1. **NLPAug Pipeline** (`nlpaug_pipeline`)
   - Synonym replacement (WordNet)
   - Random word insertion
   - Random word swap

2. **TextAttack Pipeline** (`textattack_pipeline`)
   - WordNet-based synonym replacement
   - Embedding-based word replacement

3. **Hybrid Pipeline** (`hybrid_pipeline`)
   - Combines multiple methods with configurable proportions
   - Example: 50% NLPAug synonym + 50% TextAttack WordNet

4. **Back-translation** (optional, `backtranslation_pipeline`)
   - Translates to intermediate language (German/French) and back
   - Requires additional translation models

### STRICT Data Validation
Maintains the same strict rules as NO-AUG:
- **status field** → ONLY for criteria task
- **cases field** → ONLY for evidence task
- NO cross-contamination allowed

## Directory Structure

```
DataAug_Criteria_Evidence/
├── configs/
│   ├── config.yaml                    # Main config
│   ├── data/
│   │   ├── hf_redsm5_aug.yaml        # HF dataset with augmentation
│   │   └── local_csv_aug.yaml        # Local CSV with augmentation
│   ├── model/
│   │   └── mental_bert.yaml          # MentalBERT config
│   ├── training/
│   │   └── default_aug.yaml          # Training config (12 epochs)
│   └── augmentation/
│       ├── nlpaug_default.yaml       # NLPAug config
│       ├── textattack_default.yaml   # TextAttack config
│       ├── hybrid_default.yaml       # Hybrid config
│       └── disabled.yaml             # Disable augmentation
├── src/psy_agents_aug/
│   ├── augment/                       # Augmentation pipelines
│   │   ├── base_augmentor.py         # Base interface
│   │   ├── nlpaug_pipeline.py        # NLPAug implementation
│   │   ├── textattack_pipeline.py    # TextAttack implementation
│   │   ├── hybrid_pipeline.py        # Hybrid approach
│   │   └── backtranslation.py        # Back-translation (optional)
│   ├── data/                          # Data loading (augmentation-aware)
│   ├── models/                        # Model architectures
│   ├── training/                      # Training loops
│   ├── hpo/                           # Hyperparameter optimization
│   └── utils/                         # Utilities
├── scripts/
│   ├── make_groundtruth.py           # Generate ground truth
│   ├── test_augmentation.py          # Test augmentation
│   ├── run_hpo_stage.py              # Run HPO
│   ├── train_best.py                 # Train with best params
│   └── export_metrics.py             # Export results
├── tests/
│   ├── test_augment_contract.py      # Test determinism & train-only
│   ├── test_augment_pipelines.py     # Test each pipeline
│   └── test_augment_no_leak.py       # Verify no val/test leakage
└── pyproject.toml                     # Package config
```

## Installation

```bash
# Install with poetry
poetry install

# Or with pip
pip install -e .

# Install pre-commit hooks
pre-commit install
```

## Usage

### 1. Generate Ground Truth
```bash
python scripts/make_groundtruth.py \
    --raw-data data/raw/redsm5.csv \
    --dsm-criteria data/dsm_criteria.json \
    --output-dir data/groundtruth \
    --split train \
    --task both
```

### 2. Test Augmentation
```bash
# Test NLPAug pipeline
python scripts/test_augmentation.py --pipeline nlpaug

# Test TextAttack pipeline
python scripts/test_augmentation.py --pipeline textattack

# Test Hybrid pipeline
python scripts/test_augmentation.py --pipeline hybrid
```

### 3. Train with Augmentation
```bash
# Train with NLPAug (default)
psy-aug train task=criteria

# Train with TextAttack
psy-aug train task=criteria augmentation=textattack_default

# Train with Hybrid
psy-aug train task=criteria augmentation=hybrid_default

# Train without augmentation (baseline)
psy-aug train task=criteria augmentation=disabled
```

### 4. Hyperparameter Optimization
```bash
# Run HPO with augmentation
python scripts/run_hpo_stage.py \
    --task criteria \
    --n-trials 50 \
    --augmentation nlpaug_default
```

### 5. Evaluate
```bash
# Evaluate on test set
psy-aug evaluate \
    --checkpoint outputs/best_model.pt \
    --task criteria \
    --split test
```

## Configuration

### Augmentation Settings

All augmentation configs have these key parameters:

```yaml
enabled: true                    # Enable/disable augmentation
pipeline: nlpaug_pipeline        # Which pipeline to use
ratio: 0.5                       # Augment 50% of training data
max_aug_per_sample: 1           # Max augmented variants per sample
seed: 42                         # Random seed
preserve_balance: true           # Preserve class balance
train_only: true                 # CRITICAL: Only augment training data
```

### Training Settings

Training with augmentation uses **12 epochs** (vs 10 for NO-AUG) to account for increased data volume:

```yaml
epochs: 12                       # Increased from 10
batch_size: 16
learning_rate: 2e-5
log_augmentation_stats: true    # Log augmentation stats to MLflow
```

## Testing

```bash
# Run all tests
make test

# Run augmentation-specific tests
make test-aug

# Test augmentation contracts (determinism, train-only)
make test-contract

# Test individual pipelines
make test-pipelines

# Test no data leakage
make test-no-leak

# Verify complete augmentation setup
make verify-aug
```

## Critical Guarantees

### 1. Train-Only Augmentation
**CRITICAL**: Augmentation ONLY applies to training data, NEVER to validation or test sets.

This is enforced at multiple levels:
- `AugmentationConfig.train_only` defaults to `True` and raises warning if set to `False`
- `BaseAugmentor.augment_batch()` checks split name and skips if not "train"
- `ReDSM5Loader.load_csv()` only augments when `split == "train"`
- Tests verify this guarantee in `test_augment_no_leak.py`

### 2. Deterministic Augmentation
Same seed produces same augmentations:
- All augmentors use configurable random seed
- Verified by `test_augment_contract.py`

### 3. STRICT Data Validation
Same rules as NO-AUG:
- status field → criteria task
- cases field → evidence task
- NO cross-contamination

## Comparison with NO-AUG

| Feature | NO-AUG | AUG (this repo) |
|---------|--------|-----------------|
| Package name | `psy_agents_noaug` | `psy_agents_aug` |
| Augmentation | None | NLPAug, TextAttack, Hybrid |
| Training epochs | 10 | 12 |
| Data validation | STRICT | STRICT (same rules) |
| Val/Test guarantee | N/A | NEVER augmented |
| Dependencies | Basic ML | + nlpaug, textattack |

## MLflow Tracking

Augmentation stats are automatically logged to MLflow:
- Augmentation method used
- Ratio and max_aug_per_sample
- Original vs augmented sample counts
- Augmentation time

## Development

```bash
# Format code
make format

# Lint code
make lint

# Clean generated files
make clean
```

## License

[Specify license]

## Citation

If you use this code, please cite:

```bibtex
@software{psy_agents_aug,
  title={Psy Agents AUG: Data Augmentation for Clinical Text Extraction},
  author={[Authors]},
  year={2025},
}
```
