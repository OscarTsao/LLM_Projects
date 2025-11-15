# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ReDSM-5 Classification: A machine learning project for psychiatric diagnosis using BERT-based models. The system implements data augmentation evaluation and a multi-agent architecture for DSM-5 criteria matching and evidence extraction from social media posts.

## Build and Development Commands

### Environment Setup
```bash
# Using mamba (recommended)
make env-create          # Create conda environment
make env-update          # Update environment from environment.yml

# Direct installation
make pip-install         # Install dependencies directly with pip
```

### Training Commands
```bash
# Standard training (all with early stopping)
make train               # Train with default config (patience=20)
make train-best          # Train with best config from HPO (ROC-AUC: 0.943, patience=20)
make train-gpu           # Train with GPU workstation config (patience=20)
make evaluate            # Evaluate trained model

# GPU utilities
make gpu-test            # Test GPU availability
make memory-test         # Test GPU memory estimation
```

### Multi-Agent Training
```bash
# Three training modes (with early stopping, patience=10):
make train-criteria      # Mode 1: Criteria matching agent only
make train-evidence      # Mode 2: Evidence binding agent only
make train-joint         # Mode 3: Joint training of both agents

# Evaluation per mode:
make evaluate-criteria
make evaluate-evidence
make evaluate-joint
```

### Data Augmentation
```bash
make augment-nlpaug      # Generate NLPAug augmented dataset
make augment-textattack  # Generate TextAttack augmented dataset
make augment-hybrid      # Generate hybrid augmented dataset
make augment-all         # Run all augmentation pipelines
make regenerate-synonyms # Regenerate synonym-based augmentation
```

### Docker Workflow (Recommended for Training)
```bash
make docker-clean        # Clean up containers (run if docker-up fails)
make docker-up           # Start development container
make docker-down         # Stop containers
make docker-exec         # Enter container shell
make docker-train        # Run training from host
make docker-test         # Run tests from host
```

### Testing and Code Quality
```bash
make test                # Run pytest with maxfail=1
make lint                # Check code with ruff
make format              # Format code with black
make clean               # Remove cache files
```

### Monitoring Dashboards
```bash
make mlflow-ui           # Info about MLflow UI (http://localhost:5000)
make tensorboard         # Start TensorBoard (http://localhost:6006)
```

## Architecture Overview

### Multi-Agent System
The project implements a two-agent pipeline for psychiatric diagnosis:

1. **Criteria Matching Agent** (`src/agents/criteria_matching.py`)
   - Determines if DSM-5 criteria match a given post
   - Architecture: BERT + binary classifier
   - Input: `[CLS] post [SEP] criterion_text [SEP]`
   - Output: Binary classification + confidence score

2. **Evidence Binding Agent** (`src/agents/evidence_binding.py`)
   - Predicts start/end tokens of evidence sentences when criteria match
   - Architecture: BERT + token-level classification head
   - Input: Same as criteria agent
   - Output: Token span positions for evidence

3. **Multi-Agent Pipeline** (`src/agents/multi_agent_pipeline.py`)
   - Sequential pipeline: criteria matching → evidence binding
   - Evidence agent only invoked when criteria agent detects a match
   - Combined confidence scoring

### Training Engine
The core training logic is centralized in `src/training/engine.py` with a reusable `train_model()` function. All training scripts (standard, Optuna, multi-agent modes) use this shared engine.

### Configuration System (Hydra)
Configuration uses Hydra with composition from `conf/`:
- `conf/config.yaml` - Base configuration
- `conf/dataset/*.yaml` - Dataset configs (original, nlpaug, textattack, hybrid)
- `conf/model/*.yaml` - Model configs (bert_base, roberta_base, deberta_base)
- `conf/training_mode/*.yaml` - Training mode configs (criteria, evidence, joint, gpu_workstation)
- `conf/agent/*.yaml` - Agent-specific configs

Override any config parameter via command line:
```bash
python -m src.training.train model.batch_size=32 model.learning_rate=2e-5
```

### Data Pipeline

**Data Structure:**
```
Data/
├── ReDSM5/
│   ├── redsm5_posts.csv         # Source posts
│   └── redsm5_annotations.csv   # Sentence-level evidence annotations
├── GroundTruth/
│   └── Final_Ground_Truth.json  # Criteria matching labels
└── Augmentation/
    └── augmented_*.csv          # Generated augmented datasets
```

**Data Loaders:**
- `src/data/redsm5_loader.py` - Loads posts and ground truth labels
- `src/data/evidence_loader.py` - Loads evidence span annotations
- `src/data/joint_dataset.py` - Multi-task dataset for joint training
- `src/training/dataset_builder.py` - Builds train/val/test splits

### Model Architecture
- `src/training/modeling.py` contains `BertPairClassifier`
- Supports any HuggingFace transformer model (BERT, RoBERTa, DeBERTa)
- Architecture: BERT encoder → pooled CLS token → MLP classifier
- Handles models without pooler_output (like DeBERTa) by using CLS token

### Data Augmentation Pipelines
Three augmentation approaches implemented:
1. **NLPAug** (`src/augmentation/nlpaug_pipeline.py`) - Synonym replacement
2. **TextAttack** (`src/augmentation/textattack_pipeline.py`) - Adversarial perturbations
3. **Hybrid** (`src/augmentation/hybrid_pipeline.py`) - Combined approach

All inherit from `src/augmentation/base.py` with shared utilities.

## Key Technical Details

### GPU Optimization
- TF32 enabled for RTX 3090/5090 GPUs
- Mixed precision training (fp16/bf16)
- Gradient accumulation for large effective batch sizes
- cuDNN benchmark mode enabled
- Memory estimation utilities in `src/utils/memory_utils.py`

### MLflow Integration
- Auto-logging enabled in all training scripts
- Tracking URI: `http://localhost:5000` (or `http://mlflow:5000` in containers)
- Experiments organized by training mode:
  - `redsm5-classification` - Standard training
- All configs, metrics, and artifacts logged automatically

### Best Configuration
- Pre-optimized configuration available in `conf/best_config.yaml`
- Achieved ROC-AUC: 0.943 on validation set
- Uses DeBERTa-base model with hybrid data augmentation
- Train with: `make train-best` or `python -m src.training.train --config-name=best_config`

### Environment Variables
Set in `environment.yml` and automatically loaded:
```bash
MLFLOW_TRACKING_URI=http://host.docker.internal:5500  # MLflow server
PYTHONDONTWRITEBYTECODE=1                              # No .pyc files
WANDB_MODE=disabled                                    # W&B disabled by default
```

## Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/training/test_dataset_builder.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

Test structure mirrors `src/` structure in `tests/`.

## Important Notes

### Training Workflow
1. Training outputs go to `outputs/train/` with timestamp subdirectories
2. Best model saved to `outputs/train/<timestamp>/best/model.pt`
3. Resume training with `resume=true` in config (default)
4. All metrics automatically logged to MLflow
5. Early stopping enabled by default (patience=20 for standard, patience=10 for multi-agent)

### Configuration Override
Hydra allows flexible config overrides:
```bash
# Use best configuration
make train-best

# Override dataset
python -m src.training.train dataset=original_nlpaug

# Override multiple params
python -m src.training.train model.batch_size=16 model.num_epochs=5 seed=42

# Use different model
python -m src.training.train model=roberta_base
```

### Docker vs Local Development
- **Docker (recommended)**: All dependencies pre-configured, MLflow in separate container
- **Local**: Requires mamba environment setup, must run MLflow server separately
- Container paths: `/workspaces/DataAugmentation_ReDSM5`
- Host paths: Current directory

### Multi-Agent Training Modes
Three distinct training modes with separate configs:
- **Mode 1 (criteria)**: Binary classification only
- **Mode 2 (evidence)**: Token span prediction only
- **Mode 3 (joint)**: Multi-task learning with shared encoder
- Each mode has its own training script and evaluation script

### Memory-Safe Training
- Use `model=deberta_base_memory_safe` for constrained GPU memory
- Adjust `model.batch_size` to fit GPU capacity
- Enable gradient accumulation: `model.gradient_accumulation_steps=2`
- Monitor with `make memory-test` before long training runs

### DSM-5 Criteria
Criteria descriptions are centralized in `src/data/criteria_descriptions.py` as the `CRITERIA` dictionary, mapping criterion IDs to their full text descriptions.
