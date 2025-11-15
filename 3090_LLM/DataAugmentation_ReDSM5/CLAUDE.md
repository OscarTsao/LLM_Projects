# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

**Environment setup:**
```bash
make env-create              # Create mamba environment named 'redsm5'
make env-update              # Update existing environment
mamba activate redsm5        # Activate environment
make help                    # Show all available commands
```

**Dev Container (recommended):**
```bash
# Use VS Code Dev Containers to automatically start:
# - PostgreSQL database for MLflow backend
# - MLflow tracking server (http://localhost:5000)
# - Development environment with GPU support
```

**Data augmentation:**
```bash
make augment-nlpaug          # Generate NLPAug dataset
make augment-textattack      # Generate TextAttack dataset
make augment-hybrid          # Generate hybrid dataset
make augment-all             # Run all augmentation pipelines
```

**Training:**
```bash
make train                   # Standard training with current config
make train-optuna            # Hyperparameter optimization (Optuna)
make evaluate                # Evaluate trained model

# Override config via command line:
mamba run -n redsm5 python -m src.training.train \
    dataset=original_nlpaug \
    model=roberta_base \
    model.batch_size=32 \
    model.learning_rate=3e-5

# Available models: bert_base, roberta_base, deberta_base
# Available datasets: original, original_nlpaug, original_textattack, original_hybrid, original_nlpaug_textattack
```

**Monitoring and tracking:**
```bash
make mlflow-ui               # Access MLflow UI at http://localhost:5000
make tensorboard             # Start TensorBoard at http://localhost:6006
make optuna-dashboard        # Start Optuna Dashboard at http://localhost:8080
```

**Data version control (DVC):**
```bash
make dvc-init                # Initialize DVC (done automatically in dev container)
make dvc-status              # Check DVC status
make dvc-push                # Push data to remote storage
make dvc-pull                # Pull data from remote storage
```

**Testing and code quality:**
```bash
make test                    # Run pytest (--maxfail=1 --disable-warnings -v)
make lint                    # Run ruff check
make format                  # Run black formatter
make clean                   # Remove cache files

# Run specific test:
mamba run -n redsm5 pytest tests/training/test_dataset_builder.py
```

## Architecture Overview

### Core Data Flow

1. **Ground Truth** (`Data/GroundTruth/Final_Ground_Truth.json`): Canonical labels for post-criterion pairs
2. **Augmentation**: Generate variants by replacing evidence sentences in positive examples
3. **Dataset Assembly**: Combine original + augmented data using group-based splitting (by `post_id`)
4. **Training**: BERT pair classification with `[CLS] post [SEP] criterion_text [SEP]`

### Key Modules

**`src/augmentation/`** - Text augmentation pipelines:
- `base.py`: `BaseAugmenter` abstract class with evidence replacement logic
- `nlpaug_pipeline.py`, `textattack_pipeline.py`, `hybrid_pipeline.py`: Concrete implementations
- All augmenters inherit `generate()` → `save()` → timestamped CSV in `Data/Augmentation/`

**`src/data/`** - Data loading:
- `redsm5_loader.py`: Load posts, annotations, ground truth; extract positive evidence
- `criteria_descriptions.py`: DSM-5 criterion text dictionary (`CRITERIA`)

**`src/training/`** - Training pipeline:
- `modeling.py`: `BertPairClassifier` - BERT encoder + optional hidden layers + 2-class head
- `data_module.py`: `PairDataset` + `PairCollator` (tokenizes text_a/text_b pairs)
- `dataset_builder.py`: `assemble_dataset()` + `build_splits()` using `GroupShuffleSplit`
- `engine.py`: Training loop with AMP, gradient accumulation, scheduler, checkpointing
- `train.py`: Standard training entry point
- `train_optuna.py`: Hyperparameter search with Optuna
- `evaluate.py`: Model evaluation on specified split

### Configuration (Hydra)

**Main config** (`conf/config.yaml`):
- Defaults: `dataset: original`, `model: bert_base`
- Global params: seed, output_dir, resume, metric_for_best_model

**Model config** (`conf/model/*.yaml`):
- Available models: `bert_base`, `roberta_base`, `deberta_base` (and RTX 5090 optimized variants)
- Parameters: pretrained_model_name, batch_size, learning_rate, num_epochs, optimizer, scheduler, etc.

**Dataset config** (`conf/dataset/*.yaml`):
- `ground_truth_path`, `include_original`, `use_augmented` (list), `augmented_sources` (glob patterns)
- Example: `original_nlpaug.yaml` uses both original + NLPAug augmented data

### Training Outputs

```
outputs/train/
├── checkpoints/last.pt           # Latest checkpoint for resumption
├── best/
│   ├── model.pt                  # Best model based on metric_for_best_model
│   ├── val_metrics.json          # Validation metrics at best epoch
│   └── config.yaml               # Snapshot of config used
└── test_metrics.json             # Final test set metrics
```

### MLflow Tracking Integration

**Automatic tracking** (when `MLFLOW_TRACKING_URI` is set):
- **Parameters**: All config values (flattened) are logged as parameters
- **Metrics**: Training loss, validation metrics (per epoch), and test metrics
- **Artifacts**: Best model checkpoint, config YAML, test metrics JSON
- **Tags**: model_type, framework, trial_number (for Optuna)

**Experiments**:
- `redsm5-classification`: Standard training runs
- `redsm5-optuna`: Optuna hyperparameter optimization (best trial only)

**Access MLflow UI**: http://localhost:5000 (when using dev container)

**Note**: Optuna individual trials are NOT logged to MLflow to avoid clutter. Only the best trial from each study is logged.

### Important Data Constraints

- **Group-based splitting**: All examples from same `post_id` stay in same split (prevents data leakage)
- **Augmented data**: Only positive examples (label=1) are augmented; negatives come from ground truth
- **File selection**: When multiple timestamped augmentation files exist, `_latest_match()` selects most recent
- **Evidence replacement**: `BaseAugmenter._replace_evidence()` finds and replaces evidence sentences in full post text

### Optuna Hyperparameter Search Space

```python
{
    # Model & Dataset Selection
    "model": ["bert_base", "roberta_base", "deberta_base"],
    "dataset": ["original", "original_nlpaug", "original_textattack", "original_hybrid", "original_nlpaug_textattack"],

    # Optimizer & Scheduler
    "optimizer": ["adamw_torch", "adamw_hf", "sgd"],
    "scheduler": ["linear", "cosine", "polynomial"],

    # Learning Parameters (aligned with reference config)
    "learning_rate": [1e-6, 5e-5],  # log-uniform
    "weight_decay": [1e-5, 1e-1],   # log-uniform
    "warmup_ratio": [0.0, 0.2],
    "adam_eps": [1e-9, 1e-6],       # log-uniform

    # Batch & Gradient
    "batch_size": [8, 16, 32, 64, 128, 256],
    "eval_batch_multiplier": [1, 1.5, 2],
    "gradient_accumulation_steps": [1, 2, 4, 8],
    "max_grad_norm": [0.5, 5.0],

    # Model Architecture
    "max_seq_length": [128, 256, 384, 512],
    "classifier_dropout": [0.0, 0.5],
    "classifier_layers": [0, 3],        # number of hidden layers
    "classifier_hidden_{i}": [128, 768, step=64],  # per layer

    # GPU Optimizations
    "use_bfloat16": [True, False],
    "num_workers": [8, 20],
    "prefetch_factor": [4, 12],

    # Training Duration
    "num_epochs": 100,  # fixed per spec
    "seed": [1, 10_000]
}
```

**Key features:**
- **Model selection**: HPO automatically searches across BERT, RoBERTa, and DeBERTa architectures
- **Dataset selection**: Evaluates all augmentation strategies (NLPAug, TextAttack, Hybrid, combinations)
- **Architecture search**: Dynamic classifier head depth and width tuning
- **Expanded search ranges**: Aligned with reference Criteria_Evidence_Agent project
- **Pruning**: MedianPruner with 10 startup trials, 5 warmup steps

## Development Guidelines

### Code Style
- Python 3.10+ target
- Black formatting (88 columns): `make format`
- Ruff linting: `make lint`
- Type hints for all public functions; verify with `mypy src`

### Testing
- Mirror package structure in `tests/` (e.g., `tests/augmentation/test_*.py`)
- All executable logic requires pytest coverage
- Use lightweight fixtures from `Data/GroundTruth/Final_Ground_Truth.json`

### Commits
- Use Conventional Commits: `feat:`, `fix:`, `data:`, `test:`, `docs:`
- Separate code and data changes unless atomic
- Never overwrite timestamped augmentation files—generate new ones

### Data Handling
- Never commit Reddit user identifiers
- Augmented datasets go to `Data/Augmentation/` with UTC timestamp
- `Data/GroundTruth/` contains validated labels (treat as read-only)
- Large artifacts belong in object storage, not git
