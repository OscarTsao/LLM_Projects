# ReDSM-5 Classification: Data Augmentation Evaluation

A machine learning system for psychiatric diagnosis classification using BERT-based models with advanced data augmentation strategies and multi-agent architecture.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Training](#training)
- [Data Augmentation](#data-augmentation)
- [Multi-Agent Architecture](#multi-agent-architecture)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Monitoring](#monitoring)
- [Contributing](#contributing)

## Overview

This project implements a sophisticated system for DSM-5 psychiatric criteria classification from social media posts. It features:

- **BERT-based Classification**: Uses transformer models (BERT, RoBERTa, DeBERTa) for text classification
- **Data Augmentation**: Three augmentation strategies (NLPAug, TextAttack, Hybrid) to improve model robustness
- **Multi-Agent System**: Separate agents for criteria matching and evidence extraction
- **Experiment Tracking**: Automatic MLflow logging for all experiments
- **GPU Optimization**: TF32, mixed precision, and torch.compile support

## Features

### Core Capabilities

- **Multiple Training Modes**:
  - Standard BERT pair classification
  - Criteria matching agent (DSM-5 criteria detection)
  - Evidence binding agent (evidence span extraction)
  - Joint multi-task training

- **Data Augmentation Pipelines**:
  - Synonym-based augmentation (NLPAug)
  - Adversarial perturbations (TextAttack)
  - Hybrid approach combining both strategies

- **Experiment Management**:
  - Automatic MLflow experiment tracking
  - Configuration management with Hydra
  - Early stopping with configurable patience
  - Checkpoint management and model resumption

- **Performance Optimizations**:
  - GPU acceleration with CUDA support
  - TF32 precision for Ampere GPUs (RTX 30xx/40xx)
  - Mixed precision training (fp16/bf16)
  - Gradient accumulation for memory efficiency
  - Optional torch.compile for 20-50% speedup

## Project Structure

```
.
├── conf/                          # Hydra configuration files
│   ├── config.yaml                # Default configuration
│   ├── best_config.yaml           # Best configuration (ROC-AUC: 0.943)
│   ├── dataset/                   # Dataset configurations
│   ├── model/                     # Model configurations
│   ├── training_mode/             # Training mode configurations
│   └── agent/                     # Agent-specific configurations
├── Data/
│   ├── ReDSM5/                    # Source data (posts and annotations)
│   ├── GroundTruth/               # Ground truth labels
│   └── Augmentation/              # Generated augmented datasets
├── src/
│   ├── agents/                    # Multi-agent system components
│   │   ├── base.py                # Base agent interface
│   │   ├── criteria_matching.py  # Criteria matching agent
│   │   ├── evidence_binding.py   # Evidence extraction agent
│   │   └── multi_agent_pipeline.py  # Combined pipeline
│   ├── augmentation/              # Data augmentation pipelines
│   │   ├── base.py                # Base augmenter class
│   │   ├── nlpaug_pipeline.py    # NLPAug implementation
│   │   ├── textattack_pipeline.py  # TextAttack implementation
│   │   └── hybrid_pipeline.py    # Hybrid approach
│   ├── data/                      # Data loading and processing
│   │   ├── redsm5_loader.py      # ReDSM-5 data loader
│   │   ├── evidence_loader.py    # Evidence span loader
│   │   ├── joint_dataset.py      # Multi-task dataset
│   │   └── criteria_descriptions.py  # DSM-5 criteria definitions
│   ├── training/                  # Training pipeline
│   │   ├── engine.py              # Core training engine
│   │   ├── train.py               # Standard training script
│   │   ├── train_criteria.py     # Criteria agent training
│   │   ├── train_evidence.py     # Evidence agent training
│   │   ├── train_joint.py        # Joint training
│   │   ├── evaluate.py            # Evaluation script
│   │   ├── modeling.py            # Model architecture
│   │   ├── dataset_builder.py    # Dataset construction
│   │   └── data_module.py        # Data module abstraction
│   └── utils/                     # Utility functions
│       ├── mlflow_utils.py        # MLflow integration
│       ├── wandb_utils.py         # Weights & Biases integration
│       ├── memory_utils.py        # GPU memory estimation
│       └── timestamp.py           # Timestamp utilities
├── scripts/                       # Utility scripts
│   ├── generate_nlpaug_dataset.py
│   ├── generate_textattack_dataset.py
│   └── generate_hybrid_dataset.py
├── tests/                         # Test suite
├── outputs/                       # Training outputs and checkpoints
├── Makefile                       # Build automation
├── environment.yml                # Conda/mamba environment
└── requirements.txt               # Python dependencies
```

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional but recommended)
- Mamba or Conda package manager

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd DataAugmentation_Evaluation
```

2. **Create environment**:
```bash
# Using mamba (recommended, 5-10x faster)
make env-create

# Or using conda
conda env create -f environment.yml
```

3. **Activate environment**:
```bash
mamba activate redsm5
# or
conda activate redsm5
```

4. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Basic Usage

```bash
# Train with default configuration
make train

# Train with best configuration (ROC-AUC: 0.943)
make train-best

# Evaluate trained model
make evaluate
```

## Documentation

Comprehensive documentation is available in the following files:

- **[CLAUDE.md](CLAUDE.md)**: Complete project guide for Claude Code users
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)**: Detailed setup and validation instructions
- **[MLFLOW_GUIDE.md](MLFLOW_GUIDE.md)**: MLflow tracking and experiment management
- **[MULTI_AGENT_ARCHITECTURE.md](MULTI_AGENT_ARCHITECTURE.md)**: Multi-agent system design
- **[TRAINING_CONFIGS_COMPARISON.md](TRAINING_CONFIGS_COMPARISON.md)**: Training configuration comparison
- **[GPU_SETUP.md](GPU_SETUP.md)**: GPU setup for Docker containers
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**: Quick reference card for common commands

## Training

### Standard Training

```bash
# Default configuration (BERT-base, no augmentation)
make train

# With custom parameters
python -m src.training.train model.batch_size=16 model.num_epochs=50

# GPU workstation configuration
make train-gpu
```

### Best Configuration Training

```bash
# Use pre-optimized configuration (DeBERTa + Hybrid augmentation)
make train-best
```

Configuration details:
- Model: DeBERTa-base
- Dataset: Original + Hybrid augmentation
- Batch size: 8
- Learning rate: 2.78e-5
- ROC-AUC: 0.943 (validation)

### Multi-Agent Training

```bash
# Train criteria matching agent only
make train-criteria

# Train evidence binding agent only
make train-evidence

# Joint training (multi-task learning)
make train-joint
```

### Training Output

All training runs produce:
- Model checkpoints: `outputs/train/<timestamp>/best/model.pt`
- Configuration snapshot: `outputs/train/<timestamp>/best/config.yaml`
- Validation metrics: `outputs/train/<timestamp>/best/val_metrics.json`
- Test metrics: `outputs/train/<timestamp>/test_metrics.json`

## Data Augmentation

Three augmentation strategies are available:

### 1. NLPAug (Synonym Replacement)

```bash
make augment-nlpaug
```

Replaces words with contextual synonyms using word embeddings.

### 2. TextAttack (Adversarial Perturbations)

```bash
make augment-textattack
```

Applies adversarial transformations to create challenging examples.

### 3. Hybrid (Combined Approach)

```bash
make augment-hybrid
```

Combines both strategies for maximum data diversity.

### Generate All

```bash
make augment-all
```

Runs all three augmentation pipelines sequentially.

## Multi-Agent Architecture

The system implements a two-stage pipeline:

### Criteria Matching Agent

Determines if DSM-5 criteria match a given post.

**Architecture**: BERT encoder + binary classifier
**Input**: `[CLS] post [SEP] criterion_text [SEP]`
**Output**: Binary classification + confidence score

### Evidence Binding Agent

Extracts evidence spans when criteria match.

**Architecture**: BERT encoder + token-level classifier
**Input**: `[CLS] post [SEP] criterion_text [SEP]`
**Output**: Start/end token positions for evidence

### Pipeline Flow

```
Input: Post + DSM-5 Criterion
         ↓
  Criteria Matching Agent
         ↓
  [If Match Detected]
         ↓
  Evidence Binding Agent
         ↓
Output: Match + Evidence Spans
```

## Configuration

The project uses Hydra for configuration management. Configurations are composed from multiple YAML files in `conf/`.

### Override Examples

```bash
# Change batch size
python -m src.training.train model.batch_size=32

# Use different model
python -m src.training.train model.pretrained_model_name=roberta-base

# Use different dataset
python -m src.training.train dataset=original_nlpaug

# Multiple overrides
python -m src.training.train \
    model.batch_size=16 \
    model.learning_rate=3e-5 \
    model.num_epochs=50
```

### Configuration Files

- `conf/config.yaml`: Base configuration
- `conf/best_config.yaml`: Best configuration from hyperparameter optimization
- `conf/model/*.yaml`: Model-specific configurations (BERT, RoBERTa, DeBERTa)
- `conf/dataset/*.yaml`: Dataset configurations (original, augmented variants)
- `conf/training_mode/*.yaml`: Training mode configurations
- `conf/agent/*.yaml`: Agent-specific configurations

## Evaluation

### Standard Evaluation

```bash
# Evaluate best model from training
make evaluate

# Evaluate specific checkpoint
python -m src.training.evaluate \
    evaluation.checkpoint=outputs/train/best/model.pt \
    evaluation.split=test
```

### Multi-Agent Evaluation

```bash
# Evaluate criteria matching agent
make evaluate-criteria

# Evaluate evidence binding agent
make evaluate-evidence

# Evaluate joint model
make evaluate-joint
```

### Metrics

The system tracks:
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: True positive rate
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## Monitoring

### MLflow UI

All experiments are automatically logged to MLflow.

```bash
# Access MLflow UI
# Open browser: http://localhost:5000
```

Features:
- Parameter tracking
- Metric visualization
- Model artifact storage
- Experiment comparison

### TensorBoard

```bash
# Launch TensorBoard
make tensorboard

# Access at: http://localhost:6006
```

### Weights & Biases

```bash
# Enable W&B tracking
export WANDB_MODE=online
make train
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
pytest

# Run specific test file
pytest tests/training/test_dataset_builder.py

# Run with coverage
pytest --cov=src --cov-report=html
```

## GPU Support

The project is optimized for NVIDIA GPUs with:

- **TF32 Precision**: Automatic on Ampere GPUs (RTX 30xx/40xx, A100)
- **Mixed Precision**: FP16/BF16 training support
- **CUDA Optimization**: cuDNN benchmark mode
- **Memory Efficiency**: Gradient accumulation and checkpointing

### GPU Memory Management

```bash
# Reduce batch size for limited memory
python -m src.training.train model.batch_size=8

# Use gradient accumulation
python -m src.training.train \
    model.batch_size=8 \
    model.gradient_accumulation_steps=4
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `make test`
5. Format code: `make format`
6. Commit changes: `git commit -am 'Add feature'`
7. Push to branch: `git push origin feature-name`
8. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Add docstrings to all public functions and classes
- Keep functions focused and modular
- Write unit tests for new functionality

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation information here]
```

## Acknowledgments

- HuggingFace Transformers for BERT models
- NLPAug for augmentation utilities
- TextAttack for adversarial augmentation
- MLflow for experiment tracking

## Support

For issues, questions, or contributions:
- Check existing documentation in the `docs/` folder
- Review [CLAUDE.md](CLAUDE.md) for architecture details
- Open an issue on GitHub

---

**Built with Claude Code** - AI-assisted development for modern machine learning projects.
