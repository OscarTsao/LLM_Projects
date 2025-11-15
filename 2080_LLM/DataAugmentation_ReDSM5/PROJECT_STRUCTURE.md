# REDSM5 Augmentation Cache and HPO Pipeline - Project Structure

## Overview
This document describes the complete project structure for the REDSM5 augmentation cache and hyperparameter optimization pipeline.

**Note**: This file replaces the legacy `FILE_MANIFEST.txt` which has been removed. For a comprehensive overview of the project structure, see this document and `README.md`.

**Last Updated**: 2025-10-24

## Directory Structure

```
/experiment/YuNing/DataAugmentation_ReDSM5/
├── configs/
│   ├── run.yaml                    # Main configuration (dataset, HPO, I/O settings)
│   └── augmenters_28.yaml          # Registry of 28 augmentation methods
│
├── data/
│   └── redsm5/
│       ├── base/                   # Original train/val/test parquet files
│       └── combos/                 # Cached augmented data
│           └── aug_{hash}_{split}.parquet
│
├── scripts/
│   ├── prepare_redsm5.py          # Load and prepare REDSM5 dataset
│   ├── list_combos.py             # Enumerate valid combinations
│   ├── generate_aug_cache.py      # Pre-generate augmented cache
│   ├── run_hpo_stage1.py          # Stage 1: Augmenter selection
│   └── run_hpo_stage2.py          # Stage 2: Hyperparameter tuning
│
├── src/
│   ├── __init__.py
│   │
│   ├── dataio/                    # Data loading and I/O
│   │   ├── __init__.py
│   │   ├── loader.py              # REDSM5Loader class
│   │   └── parquet_io.py          # ParquetIO utilities
│   │
│   ├── aug/                       # Augmentation management
│   │   ├── __init__.py
│   │   ├── registry.py            # AugmenterRegistry (28 augmenters)
│   │   ├── compose.py             # AugmentationPipeline
│   │   ├── combos.py              # ComboGenerator
│   │   └── seeds.py               # SeedManager (deterministic seeding)
│   │
│   ├── hpo/                       # Hyperparameter optimization
│   │   ├── __init__.py
│   │   ├── search.py              # HPOSearch (Optuna/Ray wrapper)
│   │   └── trainer.py             # AugmentationTrainer
│   │
│   └── utils/                     # Utilities
│       ├── __init__.py
│       ├── hashing.py             # Deterministic hashing (xxhash)
│       ├── logging.py             # Logging setup
│       ├── stats.py               # Dataset statistics
│       └── estimate.py            # Cache size estimation
│
├── tests/
│   ├── __init__.py
│   ├── test_augmenters.py         # Test AugmenterRegistry
│   ├── test_combos.py             # Test ComboGenerator
│   └── test_determinism.py        # Test deterministic behavior
│
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation
├── README.md                       # Project documentation
└── PROJECT_STRUCTURE.md            # This file

```

## Key Configuration Files

### configs/run.yaml
Main configuration containing:
- **Dataset settings**: Name, text field, label fields, paths
- **Combination settings**: k_max=3, exclusions, safety caps, min stage diversity
- **I/O settings**: Parquet compression (zstd), chunk sizes
- **HPO settings**: Engine (Optuna), trials, resources, search spaces
- **Global seed**: 13 (for reproducibility)

### configs/augmenters_28.yaml
Registry of 28 augmentation methods:
- **Character-level (6)**: random_delete, random_insert, random_swap, keyboard_error, ocr_noise, char_substitute
- **Word-level (5)**: word_dropout, word_swap, wordnet_synonym, embedding_substitute, spelling_noise
- **Contextual (4)**: mlm_infill_bert, mlm_infill_roberta, contextual_substitute, paraphrase_t5
- **Back-translation (5)**: en_de_en, en_fr_en, en_es_en, en_zh_en, en_ru_en
- **Formatting (8)**: whitespace_jitter, punctuation_noise, casing_jitter, contraction_expand, contraction_collapse, remove_punctuation, add_typos, normalize_unicode

Each augmenter includes:
- name, lib (nlpaug/textattack), stage
- defaults (default parameters for Stage 1)
- param_space (tunable ranges for Stage 2)

## Core Components

### 1. Data I/O (src/dataio/)
- **REDSM5Loader**: Load dataset from HuggingFace hub, CSV, or Parquet
- **ParquetIO**: Efficient reading/writing with compression and metadata

### 2. Augmentation (src/aug/)
- **AugmenterRegistry**: Central registry for 28 augmenters
- **AugmentationPipeline**: Sequential composition with deterministic seeding
- **ComboGenerator**: Enumerate valid combinations with constraints
- **SeedManager**: Hierarchical seeding for reproducibility

### 3. HPO (src/hpo/)
- **HPOSearch**: Unified interface for Optuna/Ray Tune
- **AugmentationTrainer**: Training loop with early stopping and metrics

### 4. Utilities (src/utils/)
- **hashing.py**: xxhash-based content hashing
- **logging.py**: Experiment logging
- **stats.py**: Dataset and augmentation statistics
- **estimate.py**: Cache size and time estimation

## Workflow

### 1. Prepare Dataset
```bash
python scripts/prepare_redsm5.py --source hub
```
Loads REDSM5 from HuggingFace and saves as Parquet.

### 2. List Valid Combinations
```bash
python scripts/list_combos.py --k-max 3 --output data/redsm5/combos/valid_combos.json
```
Generates all valid combinations respecting constraints.

### 3. Generate Augmentation Cache
```bash
python scripts/generate_aug_cache.py --num-workers 8
```
Pre-generates augmented data for all combinations.

### 4. Stage 1 HPO (Augmenter Selection)
```bash
python scripts/run_hpo_stage1.py --trials 100
```
Searches over augmentation combinations with fixed default parameters.

### 5. Stage 2 HPO (Hyperparameter Tuning)
```bash
python scripts/run_hpo_stage2.py --combo random_delete word_dropout --trials 50
```
Fine-tunes augmentation parameters and model hyperparameters.

## Key Features

### Deterministic Hashing
- Content-based hashing using xxhash
- Hash = f(combo, params, seed, text)
- Enables cache reuse across experiments

### Constraint System
- **Exclusions**: Mutually exclusive augmenters (e.g., only one back-translation)
- **Safety caps**: Limit combinatorial explosion
- **Stage diversity**: Require diverse augmentation stages

### Two-Stage HPO
- **Stage 1**: Find best augmentation combo (discrete search)
- **Stage 2**: Optimize continuous parameters (intensity, learning rate, etc.)

### Efficient I/O
- Parquet with zstd compression (compression_level=3)
- Chunked reading/writing for memory efficiency
- Metadata storage in Parquet files

## Testing

Run tests with:
```bash
pytest tests/ -v
```

Test coverage:
- **test_augmenters.py**: Registry initialization, listing, config retrieval
- **test_combos.py**: Combination generation, constraints, statistics
- **test_determinism.py**: Seeding, hashing consistency

## Installation

```bash
# Install in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

## Recent Updates

- **2025-10-24**: Fixed critical F1 score calculation bug in evidence binding agent
- **2025-10-24**: Code cleanup - removed unused imports from training modules

## Dependencies

Core dependencies:
- nlpaug>=1.1.11 (augmentation)
- textattack>=0.3.8 (augmentation)
- torch>=1.9.0 (deep learning)
- transformers>=4.20.0 (pretrained models)
- optuna>=3.0.0 (HPO)
- ray[tune]>=2.0.0 (HPO)
- pyarrow>=10.0.0 (Parquet I/O)
- xxhash>=3.0.0 (hashing)

## File Naming Convention

Cached augmented data:
- Format: `aug_{combo_hash}_{split}.parquet`
- Example: `aug_a1b2c3d4e5f6g7h8_train.parquet`

Combo hash is deterministic and includes:
- Ordered augmenter names
- Parameters for each augmenter
- Global seed

## Notes

- All paths in this document use absolute paths from project root
- Configuration files use relative paths (resolved at runtime)
- Default global seed is 13 for reproducibility
- Parquet compression uses zstd level 3 (good balance of speed/ratio)
- HPO uses Optuna by default (Ray Tune support planned)
