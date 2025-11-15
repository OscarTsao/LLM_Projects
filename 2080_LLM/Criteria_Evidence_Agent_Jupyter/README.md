# Criteria Evidence Agent

Multi-label classification system for mental health symptom detection using transformer encoders (BERT, RoBERTa, DeBERTa) with evidence extraction capabilities. Built with PyTorch, Hydra for configuration management, MLflow for experiment tracking, and Optuna for hyperparameter optimization.

## Features

- **Multiple Encoder Support**: BERT, RoBERTa, and DeBERTa transformer encoders
- **Multi-Task Learning**: Symptom classification, token-level evidence extraction, and span-level evidence extraction
- **Advanced Training**: Mixed precision (FP16/BF16), gradient checkpointing, LoRA fine-tuning
- **Composite Losses**: Binary cross-entropy with adaptive focal loss for handling class imbalance
- **Hyperparameter Optimization**: Automated HPO with Optuna using TPE sampling and median pruning
- **Experiment Tracking**: Integrated MLflow tracking for experiments and model artifacts
- **Modular Configuration**: Hydra-based configuration with composable config groups
- **Production Ready**: Comprehensive logging, early stopping, model checkpointing

## Project Structure

```
.
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main configuration with defaults
│   ├── data/                  # Data configuration group
│   │   └── default.yaml
│   ├── model/                 # Model configuration group
│   │   ├── roberta_base.yaml
│   │   ├── bert_base.yaml
│   │   └── deberta_base.yaml
│   ├── training/              # Training configuration group
│   │   └── default.yaml
│   └── hpo/                   # HPO configuration group
│       └── default.yaml
├── src/
│   ├── data/                  # Data loading and preprocessing
│   ├── models/                # Model architectures and encoders
│   ├── utils/                 # Utility functions (metrics, schedulers, training)
│   ├── losses.py              # Loss functions
│   ├── train.py               # Training script
│   └── hpo.py                 # Hyperparameter optimization
├── Data/                      # Dataset directory
│   ├── redsm5/               # REDSM5 dataset
│   └── groundtruth/          # Ground truth annotations
├── Makefile                   # Simplified commands
└── pyproject.toml            # Project dependencies
```

## Installation

### Requirements
- Python >=3.9
- CUDA-compatible GPU (recommended)
- Conda or Miniforge (recommended for dependency management)

### Setup

```bash
# Clone the repository
cd /path/to/Criteria_Evidence_Agent

# Install the package and dependencies
make install

# Or install with development dependencies
make dev-install

# Alternatively, use pip directly
pip install -e .
```

## Quick Start

### Training

Train with default configuration (RoBERTa):
```bash
make train
```

Train with specific encoder:
```bash
make train-bert       # BERT encoder
make train-deberta    # DeBERTa encoder
make train-roberta    # RoBERTa encoder
```

Train with custom parameters:
```bash
# Using Makefile
make experiment ARGS='training.batch_size=16 training.learning_rate=1e-5'

# Using python directly
PYTHONPATH=. python src/train.py training.batch_size=16 training.learning_rate=1e-5

# Override model configuration
PYTHONPATH=. python src/train.py model=bert_base training.max_epochs=10
```

### Hyperparameter Optimization

Run HPO with Optuna:
```bash
make hpo
```

Configure HPO in `configs/hpo/default.yaml` to adjust:
- Search space (learning rate, batch size, model architecture, etc.)
- Number of trials
- Sampling strategy (TPE, Random, etc.)
- Pruning strategy (Median, Hyperband, etc.)

### Configuration Composition

Hydra allows flexible configuration composition and overriding:

```bash
# View current configuration
PYTHONPATH=. python src/train.py --help

# Override specific parameters
PYTHONPATH=. python src/train.py \
    model=bert_base \
    training.batch_size=32 \
    training.max_epochs=10 \
    training.learning_rate=3e-5

# Enable LoRA fine-tuning
PYTHONPATH=. python src/train.py \
    model.encoder.lora.enabled=true \
    model.encoder.lora.r=8 \
    model.encoder.lora.alpha=16

# Multiple overrides
PYTHONPATH=. python src/train.py \
    model=deberta_base \
    training.batch_size=16 \
    data.max_length=512 \
    model.encoder.gradient_checkpointing=true
```

## Makefile Commands

```bash
make help              # Show all available commands
make install           # Install production dependencies
make dev-install       # Install development dependencies
make clean             # Clean generated files and caches
make train             # Train with default config (RoBERTa)
make train-bert        # Train with BERT encoder
make train-deberta     # Train with DeBERTa encoder
make train-roberta     # Train with RoBERTa encoder
make hpo               # Run hyperparameter optimization
make format            # Format code with black
make lint              # Lint code with ruff
make check             # Run format and lint checks
make validate-config   # Validate Hydra configuration
make mlflow-server     # Start MLflow tracking server
```

## Configuration

### Modular Configuration System

The project uses Hydra's composition pattern with separate config groups:

#### Data Configuration (`configs/data/default.yaml`)
- Dataset paths and field names
- Multi-label symptom fields
- Train/val/test split ratios
- Maximum sequence length

#### Model Configuration (`configs/model/*.yaml`)
- **roberta_base.yaml**: RoBERTa encoder configuration
- **bert_base.yaml**: BERT encoder configuration
- **deberta_base.yaml**: DeBERTa encoder configuration

Each includes:
- Encoder type and pretrained model path
- Gradient checkpointing settings
- LoRA configuration for parameter-efficient fine-tuning
- Classification heads (symptom, token, span)
- Dropout rates and pooling strategy

#### Training Configuration (`configs/training/default.yaml`)
- Batch size and gradient accumulation
- Learning rate and optimizer settings (AdamW)
- Scheduler configuration (linear, cosine)
- Mixed precision training (AMP, BF16)
- Focal loss parameters for class imbalance
- Early stopping criteria
- Loss weights for multi-task learning

#### HPO Configuration (`configs/hpo/default.yaml`)
- Study name and storage backend
- Number of trials and parallel jobs
- Sampler configuration (TPE with multivariate sampling)
- Pruner configuration (Median pruner)
- Search space definitions for all hyperparameters

## Experiment Tracking

MLflow is used for comprehensive experiment tracking:

```bash
# Start MLflow UI
make mlflow-server

# Or manually
mlflow ui --port 5000
```

Access the UI at http://localhost:5000

### Environment Variables

```bash
export MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"
export MLFLOW_EXPERIMENT_NAME="custom_experiment"
export OPTUNA_STORAGE_URL="sqlite:///optuna.db"
```

### What's Tracked

- All configuration parameters (flattened)
- Training and validation metrics per epoch
- Best model checkpoint
- Test set evaluation metrics
- HPO trial results and best parameters

## Dataset

This project uses the REDSM5 dataset for mental health symptom classification. Ensure the data is placed in the `Data/` directory:

```
Data/
├── redsm5/
│   ├── redsm5_posts.csv
│   └── redsm5_annotations.csv
└── groundtruth/
    └── redsm5_ground_truth.json
```

### Multi-Label Symptoms

The system classifies text across 10 mental health symptom categories:
- ANHEDONIA
- APPETITE_CHANGE
- COGNITIVE_ISSUES
- DEPRESSED_MOOD
- FATIGUE
- PSYCHOMOTOR
- SLEEP_ISSUES
- SPECIAL_CASE
- SUICIDAL_THOUGHTS
- WORTHLESSNESS

## Advanced Features

### LoRA Fine-Tuning
Enable parameter-efficient fine-tuning with LoRA:
```bash
PYTHONPATH=. python src/train.py \
    model.encoder.lora.enabled=true \
    model.encoder.lora.r=16 \
    model.encoder.lora.alpha=32 \
    model.encoder.lora.dropout=0.05
```

### Mixed Precision Training
Automatically enabled for CUDA GPUs with AMP and BF16 support:
```yaml
training:
  amp: true      # Enable automatic mixed precision
  bf16: true     # Use bfloat16 if supported (RTX 3090/5090)
```

### Gradient Checkpointing
Reduce memory usage for larger models or longer sequences:
```yaml
model:
  encoder:
    gradient_checkpointing: true
```

### Custom Thresholds
Adjust classification thresholds per symptom:
```yaml
model:
  heads:
    symptom_labels:
      thresholds:
        ANHEDONIA: 0.6
        DEPRESSED_MOOD: 0.4
        SUICIDAL_THOUGHTS: 0.7
```

### Multi-Task Learning
Enable token-level or span-level evidence extraction:
```yaml
model:
  heads:
    evidence_token:
      enabled: true
    evidence_span:
      enabled: true
```

## Development

### Code Quality

```bash
# Format code with black
make format

# Lint code with ruff
make lint

# Run both
make check
```

### Clean Up

```bash
# Remove generated files, outputs, and caches
make clean
```

This removes:
- Python cache files (`__pycache__`, `*.pyc`)
- Hydra outputs (`outputs/`, `multirun/`)
- Model artifacts (`artifacts/`)
- MLflow runs (`mlruns/`)
- Optuna database (`optuna.db`)

## Dev Container

The project includes a `.devcontainer` setup with:
- CUDA-enabled PyTorch
- MLflow + Postgres services
- DVC for data versioning
- All project dependencies

Usage:
1. Open repository in VS Code
2. Run "Dev Containers: Reopen in Container"
3. Dependencies are automatically installed

## Reproducibility

- Deterministic seeds across Python, NumPy, and PyTorch
- Data splits are consistent with fixed seed
- MLflow tracks all parameters and artifacts
- Hydra outputs preserve full configuration

## Hardware Optimization

### GPU Configuration
```bash
# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0

# For multi-GPU
export CUDA_VISIBLE_DEVICES=0,1
```

### Memory Optimization
- Use gradient checkpointing for large models
- Enable mixed precision (BF16 on RTX 3090/5090)
- Adjust batch size and gradient accumulation
- Reduce `max_length` if memory constrained

### Performance Tuning
- Increase `num_workers` to match CPU cores
- Use larger batch sizes on high-end GPUs
- Enable gradient accumulation for effective large batches
- Adjust learning rate with batch size changes

## Troubleshooting

### Import Errors
Ensure PYTHONPATH is set:
```bash
export PYTHONPATH=.
```

Or use the Makefile commands which handle this automatically.

### Out of Memory
- Reduce `training.batch_size`
- Increase `training.gradient_accumulation_steps`
- Enable `model.encoder.gradient_checkpointing`
- Reduce `data.max_length`

### Slow Training
- Increase `training.num_workers` for faster data loading
- Enable mixed precision training
- Use gradient checkpointing only if memory-constrained

## Citation

If you use this code, please cite:
```bibtex
@software{criteria_evidence_agent,
  title={Criteria Evidence Agent},
  year={2025},
  url={https://github.com/yourusername/criteria-evidence-agent}
}
```

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please ensure:
1. Code is formatted with `make format`
2. Code passes linting with `make lint`
3. Configuration is validated with `make validate-config`
4. All tests pass (when implemented)
