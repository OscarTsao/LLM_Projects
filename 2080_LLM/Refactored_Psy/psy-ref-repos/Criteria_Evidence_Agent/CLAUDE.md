# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-label mental health symptom classification system using transformer encoders (BERT, RoBERTa, DeBERTa) with evidence extraction capabilities. Built with PyTorch, Hydra for configuration, MLflow for tracking, and Optuna for hyperparameter optimization.

**Multi-Task Architecture**: The system performs three tasks simultaneously:
1. Symptom classification (multi-label, 10 mental health symptoms)
2. Token-level evidence extraction
3. Span-level evidence extraction (start/end positions)

## Common Commands

### Training
```bash
# Use Makefile (PYTHONPATH is automatically set)
make train              # Train with default RoBERTa config
make train-bert         # Train with BERT encoder
make train-deberta      # Train with DeBERTa encoder
make hpo                # Run hyperparameter optimization

# Direct Python (requires PYTHONPATH=.)
PYTHONPATH=. python src/train.py
PYTHONPATH=. python src/train.py model=bert_base training.batch_size=16
PYTHONPATH=. python src/hpo.py
```

### Development
```bash
make install           # Install production dependencies
make dev-install       # Install dev dependencies (black, ruff)
make format            # Format with black (line-length 100)
make lint              # Lint with ruff
make check             # Format + lint
make clean             # Clean outputs, caches, artifacts
make validate-config   # Validate Hydra configs
make mlflow-server     # Start MLflow UI on port 5000
```

### Running Single Tests
```bash
# Tests not yet implemented - pytest would go here
# pytest tests/test_specific.py::test_function -v
```

## Code Architecture

### Configuration System (Hydra-based)

**Composition Pattern**: Uses Hydra's config groups for modular composition
- `configs/config.yaml`: Main config with defaults
- `configs/data/`: Dataset configuration
- `configs/model/`: Encoder configs (bert_base, roberta_base, deberta_base)
- `configs/training/`: Training hyperparameters
- `configs/hpo/`: Optuna optimization settings

**Key Override Pattern**:
```bash
python src/train.py model=bert_base training.batch_size=32 model.encoder.lora.enabled=true
```

**Environment Resolver**: Configs use `${env:VAR, default}` syntax for environment variables (MLflow URI, experiment name, etc.)

### Model Architecture (`src/models/`)

**Encoder Factory Pattern** (`encoder_factory.py`):
- Central dispatcher: `build_encoder(cfg)` → returns (model, hidden_size)
- Encoder-specific loaders in `bert_encoder.py`, `roberta_encoder.py`, `deberta_encoder.py`
- Each encoder supports: gradient checkpointing, LoRA fine-tuning, custom pooling

**EvidenceModel** (`model.py`):
- Main model class wrapping encoder + multiple classification heads
- **Classification heads** (all inherit from `BaseClassificationHead`):
  - `symptom_labels`: Multi-label symptom classification (sigmoid activation)
  - `evidence_token`: Token-level evidence (BIO tagging)
  - `evidence_span`: Span extraction (start/end positions)
- **Pooling strategies**: `[CLS]`, mean, max, attention-based
- **LoRA support**: Uses PEFT library for parameter-efficient fine-tuning

### Loss Functions (`src/losses.py`)

**Composite Loss Strategy**:
- `adaptive_focal_loss`: Focal loss with dynamic gamma adjustment based on positive rate
- `multi_label_loss`: BCE with optional label smoothing and pos_weight
- Loss combination in `src/utils/training.py:compute_loss()` using configurable weights

### Data Pipeline (`src/data/`)

**DataModule Pattern**:
- `DataModule`: Orchestrates data loading, creates train/val/test splits
- `PostDataset`: Custom dataset for multi-label symptoms + evidence annotations
- `TokenizedDataCollator`: Dynamic padding and batch collation
- **Symptom fields**: ANHEDONIA, APPETITE_CHANGE, COGNITIVE_ISSUES, DEPRESSED_MOOD, FATIGUE, PSYCHOMOTOR, SLEEP_ISSUES, SPECIAL_CASE, SUICIDAL_THOUGHTS, WORTHLESSNESS

### Training Utilities (`src/utils/`)

**Key modules**:
- `optimizers.py`: AdamW with layer-wise learning rate decay
- `schedulers.py`: Linear, cosine, one-cycle, polynomial, plateau schedulers
- `training.py`: `compute_loss()` and `evaluate()` core functions
- `metrics.py`: Multi-label metrics (macro/micro F1, precision, recall), span metrics, token metrics
- `ema.py`: Exponential moving average for model weights
- `common.py`: `set_seed()` for reproducibility, `flatten_dict()` for MLflow logging

### Experiment Tracking Flow

1. **Configuration**: Hydra creates output dir at `outputs/${model.encoder.type}/${now:%Y-%m-%d}/${now:%H-%M-%S}`
2. **MLflow Setup**: Auto-logging enabled, tracks all config params (flattened), metrics, and artifacts
3. **Training Loop**: Metrics logged per epoch, best model checkpointed
4. **HPO**: Optuna trials logged as nested MLflow runs

## Important Patterns

### PYTHONPATH Requirement
All Python scripts require `PYTHONPATH=.` or use Makefile commands. Imports use absolute paths: `from src.data import DataModule`

### Mixed Precision Training
- Automatically uses AMP (Automatic Mixed Precision) if `training.amp=true`
- BF16 preferred on compatible GPUs (RTX 3090/5090+): `training.bf16=true`
- GradScaler handles loss scaling

### Multi-Task Loss Weighting
Configure relative importance in `configs/training/default.yaml`:
```yaml
loss_weights:
  symptom_labels: 1.0
  evidence_token: 1.0
  evidence_span_start: 1.0
  evidence_span_end: 1.0
```

### Hyperparameter Search Space
Defined in `configs/hpo/default.yaml` with Optuna syntax:
- Categorical: `{"type": "categorical", "choices": [...]}`
- Float: `{"type": "float", "low": ..., "high": ..., "log": true}`
- Int: `{"type": "int", "low": ..., "high": ...}`

## Development Container

`.devcontainer/docker-compose.yml` provides:
- CUDA-enabled dev environment
- MLflow server (port 5001) with Postgres backend
- Shared volume mounting
- Environment variables pre-configured

## Key Files to Modify

- **Add new encoder**: Create `src/models/your_encoder.py`, register in `encoder_factory.py`, add config to `configs/model/`
- **Modify loss function**: Edit `src/losses.py` and update `compute_loss()` in `src/utils/training.py`
- **Add metrics**: Extend `src/utils/metrics.py` and update `evaluate()` function
- **Change data loading**: Modify `src/data/dataset.py`
- **Adjust training loop**: Edit `src/train.py` main training loop
- **HPO search space**: Update `configs/hpo/default.yaml`

## Environment Variables

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"  # MLflow server
export MLFLOW_EXPERIMENT_NAME="redsm5_classification"  # Experiment name
export OPTUNA_STORAGE_URL="sqlite:///optuna.db"  # Optuna backend
export CUDA_VISIBLE_DEVICES=0  # GPU selection
```
