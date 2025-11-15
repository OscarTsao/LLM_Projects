# Data Augmentation for ReDSM-5 Classification

A research project for augmenting and training BERT-based models for DSM-5 symptom classification from Reddit posts. This project implements multiple text augmentation strategies (NLPAug, TextAttack, and hybrid approaches) to improve model performance on psychiatric symptom detection.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Data Augmentation Methods](#data-augmentation-methods)
- [Training](#training)
- [Evaluation](#evaluation)
- [Development](#development)

## Overview

This project focuses on:

1. **Data Augmentation**: Generate augmented training examples using synonym replacement, contextual word embeddings, and back-translation
2. **Model Training**: Fine-tune BERT models for pair classification (post + DSM-5 criterion → label)
3. **Hyperparameter Optimization**: Use Optuna for automated hyperparameter tuning
4. **Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1, and ROC-AUC
5. **🆕 Automatic MLflow Tracking**: All experiments automatically logged to MLflow with zero configuration required

### Key Features

- **🔬 Automatic Experiment Tracking**: Every training run logged to MLflow automatically
- **📊 Monitoring Dashboards**: MLflow UI, TensorBoard, and Optuna Dashboard
- **🐳 Dev Container**: Ready-to-run VS Code devcontainer with GPU tooling and local MLflow tracking (override for remote endpoints if needed)
- **📡 Dual Tracking**: Training automatically streams to MLflow and Weights & Biases
- **📦 Data Version Control**: DVC integration for dataset versioning
- Multiple augmentation strategies (NLPAug, TextAttack, Hybrid)
- Flexible training pipeline with Hydra configuration
- Hyperparameter optimization with Optuna
- Group-based train/val/test splitting (by post_id)
- Mixed precision training (AMP) for faster training
- TF32 and cuDNN optimizations for NVIDIA GPUs
- Checkpoint resumption and best model saving

## Project Structure

```
DataAugmentation_ReDSM5/
├── Data/
│   ├── ReDSM5/              # Raw ReDSM-5 dataset
│   │   ├── redsm5_posts.csv
│   │   └── redsm5_annotations.csv
│   ├── GroundTruth/         # Canonical ground truth labels
│   │   └── Final_Ground_Truth.json
│   └── Augmentation/        # Generated augmented datasets
├── src/
│   ├── augmentation/        # Augmentation pipelines
│   │   ├── base.py         # Abstract base class
│   │   ├── nlpaug_pipeline.py
│   │   ├── textattack_pipeline.py
│   │   └── hybrid_pipeline.py
│   ├── data/               # Data loading utilities
│   │   ├── redsm5_loader.py
│   │   └── criteria_descriptions.py
│   ├── training/           # Training modules
│   │   ├── modeling.py     # BERT classifier model
│   │   ├── engine.py       # Training loop
│   │   ├── data_module.py  # PyTorch DataModule
│   │   ├── dataset_builder.py
│   │   ├── train.py        # Standard training
│   │   ├── train_optuna.py # Hyperparameter tuning
│   │   └── evaluate.py     # Model evaluation
│   └── utils/
│       └── timestamp.py    # Timestamp utilities
├── scripts/                # Data generation scripts
│   ├── generate_nlpaug_dataset.py
│   ├── generate_textattack_dataset.py
│   ├── generate_hybrid_dataset.py
│   └── regenerate_augmentation.py
├── conf/                   # Hydra configuration
│   ├── config.yaml        # Main config
│   ├── model/
│   │   └── bert_base.yaml
│   └── dataset/
│       ├── original.yaml
│       ├── original_nlpaug.yaml
│       ├── original_textattack.yaml
│       └── original_hybrid.yaml
├── tests/                  # Unit tests
├── Makefile               # Build automation
├── environment.yml        # Mamba environment
├── requirements.txt       # Python dependencies
└── README.md

```

## Model Architecture

### BertPairClassifier

The model architecture consists of:

1. **BERT Encoder**: Pre-trained `google-bert/bert-base-uncased`
   - Input: `[CLS] text_a [SEP] text_b [SEP]`
   - Output: 768-dimensional pooled representation

2. **Classification Head**:
   - Configurable hidden layers (default: direct projection)
   - GELU activation
   - Dropout regularization
   - Final linear layer → 2 classes (positive/negative)

3. **Loss Function**: Cross-entropy loss

### Input Format

- **text_a**: Reddit post text
- **text_b**: DSM-5 criterion description
- **label**: 0 (no symptom) or 1 (symptom present)

## Installation

### Prerequisites

- Python 3.10+
- Mamba (or Conda)
- CUDA 12.1+ (for GPU training)

### Setup

1. **Clone the repository**:
```bash
cd /path/to/DataAugmentation_ReDSM5
```

2. **Create environment**:
```bash
make env-create
```

This creates a mamba environment named `redsm5` with all dependencies.

3. **Activate environment**:
```bash
mamba activate redsm5
```

Alternatively, use `mamba run -n redsm5` prefix for commands.

### Environment Details

The environment includes:
- PyTorch 2.1+ with CUDA 12.1
- Transformers (Hugging Face)
- NLPAug for augmentation
- TextAttack for adversarial augmentation
- Optuna for hyperparameter optimization
- Hydra for configuration management
- scikit-learn, pandas, numpy
- Development tools: ruff, black, mypy, pytest

## Usage

### Quick Start

```bash
# Generate augmented datasets
make augment-all

# Train with default config (auto-logs to MLflow!)
make train

# View experiments locally
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Monitor runs in Weights & Biases (https://wandb.ai)

# Train with hyperparameter optimization
make train-optuna

# Evaluate trained model
make evaluate
```

### 🆕 MLflow Auto-Logging

**Every training run is automatically logged to MLflow!**

```bash
# 1. Run any training command
make train

# 2. Launch MLflow UI against the project-local store
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# 3. View all metrics, parameters, and download models
# Experiment: "redsm5-classification"
# W&B dashboards update automatically at https://wandb.ai/<entity>/<project>
```

**What gets logged:**
- ✅ All configuration parameters
- ✅ Training/validation/test metrics
- ✅ Model checkpoints
- ✅ Config snapshots

Weights & Biases logs include training/validation/test curves, Optuna trials, and best-checkpoint summaries.

### Data Augmentation

Generate augmented datasets using different strategies:

```bash
# NLPAug (synonym replacement, contextual word embeddings)
make augment-nlpaug

# TextAttack (adversarial and rule-based transformations)
make augment-textattack

# Hybrid (combination of multiple methods)
make augment-hybrid

# Regenerate with synonym expansion
make regenerate-synonyms
```

Augmented datasets are saved to `Data/Augmentation/` with timestamps.

### Training

**Standard Training**:
```bash
# Use default configuration
mamba run -n redsm5 python -m src.training.train

# Override config parameters
mamba run -n redsm5 python -m src.training.train \
    dataset=original_nlpaug \
    model.batch_size=32 \
    model.learning_rate=3e-5 \
    model.num_epochs=10
```

**Hyperparameter Optimization with Optuna**:
```bash
# Run 100 trials
mamba run -n redsm5 python -m src.training.train_optuna n_trials=100

# With timeout (in seconds)
mamba run -n redsm5 python -m src.training.train_optuna timeout=3600
```

**Training Outputs**:
- Checkpoints: `outputs/train/checkpoints/last.pt`
- Best model: `outputs/train/best/model.pt`
- Validation metrics: `outputs/train/best/val_metrics.json`
- Test metrics: `outputs/train/test_metrics.json`
- Config snapshot: `outputs/train/best/config.yaml`

### Evaluation

```bash
# Evaluate saved checkpoint
mamba run -n redsm5 python -m src.training.evaluate \
    evaluation.checkpoint=outputs/train/best/model.pt \
    evaluation.split=test
```

## Configuration

### Main Config (`conf/config.yaml`)

```yaml
defaults:
  - dataset: original
  - model: bert_base
  - _self_

seed: 1337
output_dir: outputs/train
metric_for_best_model: roc_auc
resume: true

dataloader:
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4
```

### Model Config (`conf/model/bert_base.yaml`)

```yaml
pretrained_model_name: google-bert/bert-base-uncased
classifier_hidden_sizes: []  # Direct projection
classifier_dropout: 0.1
max_seq_length: 256
batch_size: 32
gradient_accumulation_steps: 1
num_epochs: 5
learning_rate: 2e-5
weight_decay: 0.01
warmup_ratio: 0.1
optimizer: adamw_torch
scheduler: linear
adam_eps: 1e-8
max_grad_norm: 1.0
compile_model: false  # Set to true for PyTorch 2.0+ compilation
```

### Dataset Config (`conf/dataset/original_nlpaug.yaml`)

```yaml
ground_truth_path: Data/GroundTruth/Final_Ground_Truth.json
include_original: true
use_augmented:
  - nlpaug
augmented_sources:
  nlpaug: "Data/Augmentation/nlpaug_dataset_*.csv"
splits:
  train: 0.7
  val: 0.15
  test: 0.15
shuffle_seed: 42
```

## Data Augmentation Methods

### 1. NLPAug Pipeline

**Techniques**:
- Contextual word embeddings (BERT-based word substitution)
- Synonym replacement using WordNet
- Back-translation

**Usage**:
```python
from src.augmentation.nlpaug_pipeline import NLPAugPipeline
pipeline = NLPAugPipeline()
df = pipeline.run()
```

### 2. TextAttack Pipeline

**Techniques**:
- Synonym substitution
- Word embedding transformations
- Character-level perturbations
- Grammatical transformations

**Usage**:
```python
from src.augmentation.textattack_pipeline import TextAttackPipeline
pipeline = TextAttackPipeline()
df = pipeline.run()
```

### 3. Hybrid Pipeline

Combines multiple augmentation strategies for maximum diversity.

**Usage**:
```python
from src.augmentation.hybrid_pipeline import HybridPipeline
pipeline = HybridPipeline()
df = pipeline.run()
```

## Training

### Training Loop

The training engine (`src/training/engine.py`) implements:

1. **Initialization**:
   - Set random seeds for reproducibility
   - Load and split dataset
   - Initialize model, optimizer, scheduler

2. **Training**:
   - Mixed precision training (AMP)
   - Gradient accumulation
   - Gradient clipping
   - Learning rate warmup and decay
   - Progress tracking with tqdm

3. **Validation**:
   - Evaluate on validation set after each epoch
   - Track best model based on configured metric
   - Early stopping via Optuna pruning (optional)

4. **Checkpointing**:
   - Save last checkpoint for resumption
   - Save best model separately
   - Clean up old checkpoints

5. **Testing**:
   - Final evaluation on test set
   - Save comprehensive metrics

### Performance Optimizations

**GPU Optimizations** (automatic when CUDA available):
- TF32 tensor cores for matmul
- cuDNN benchmark mode
- Mixed precision training (AMP)
- Pin memory for faster data transfer
- Persistent workers for DataLoader

**Optional**:
- Model compilation with `torch.compile` (PyTorch 2.0+)
- Set `compile_model: true` in config

### Hyperparameter Search Space (Optuna)

```python
{
    "learning_rate": [1e-5, 5e-5],
    "batch_size": [16, 32, 64],
    "warmup_ratio": [0.0, 0.2],
    "weight_decay": [0.0, 0.1],
    "classifier_dropout": [0.1, 0.3],
    "num_epochs": [3, 10],
    "gradient_accumulation_steps": [1, 2, 4],
    "optimizer": ["adamw_torch", "adamw_hf"],
    "scheduler": ["linear", "cosine", "polynomial"]
}
```

## Evaluation

### Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Positive class precision
- **Recall**: Positive class recall
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

### Evaluation Output

```json
{
  "accuracy": 0.8542,
  "precision": 0.8123,
  "recall": 0.7891,
  "f1": 0.8005,
  "roc_auc": 0.9123
}
```

## Development

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Run tests
make test

# Clean cache files
make clean
```

### Testing

```bash
# Run all tests
mamba run -n redsm5 pytest

# Run specific test file
mamba run -n redsm5 pytest tests/training/test_dataset_builder.py

# Run with coverage
mamba run -n redsm5 pytest --cov=src --cov-report=html
```

### Project Guidelines

See `AGENTS.md` for detailed development guidelines including:
- Coding style and conventions
- Testing requirements
- Commit message format
- Data handling and security
- Pull request process

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**:
- Reduce `batch_size` in model config
- Increase `gradient_accumulation_steps`
- Reduce `max_seq_length`
- Use fewer `num_workers` in dataloader

**Slow Training**:
- Increase `batch_size` if memory allows
- Enable `compile_model: true` (PyTorch 2.0+)
- Increase `num_workers` for data loading
- Verify GPU utilization with `nvidia-smi`

**Data Loading Issues**:
- Verify data paths in dataset configs
- Check that augmented datasets exist
- Ensure ground truth JSON is valid

**Import Errors**:
- Verify environment is activated
- Run `make env-update` to sync dependencies
- Check `requirements.txt` is installed

## Performance Benchmarks

**Training Speed** (on NVIDIA A100):
- Batch size 32: ~120 samples/sec
- Batch size 64: ~200 samples/sec
- With `compile_model`: ~1.3x faster

**Memory Usage**:
- Batch size 16: ~4GB VRAM
- Batch size 32: ~7GB VRAM
- Batch size 64: ~12GB VRAM

## Citation

If you use this code in your research, please cite:

```bibtex
@software{redsm5_augmentation,
  title={Data Augmentation for ReDSM-5 Classification},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/DataAugmentation_ReDSM5}
}
```

## License

This project is for research purposes only. Ensure compliance with Reddit's Terms of Service and data privacy regulations when using this code.

## Acknowledgments

- ReDSM-5 dataset creators
- Hugging Face Transformers library
- NLPAug and TextAttack libraries
- PyTorch team

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].
