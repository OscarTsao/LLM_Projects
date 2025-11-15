# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-label mental health symptom classification system using transformer encoders (BERT, RoBERTa, DeBERTa) with evidence extraction capabilities. Built with PyTorch, Hydra configuration management, MLflow experiment tracking, and Optuna hyperparameter optimization. The project exists in two forms:
1. **Python scripts** (`src/train.py`, `src/hpo.py`) - Production training pipeline
2. **Jupyter notebooks** - Interactive development and analysis environment with auto-resume capabilities

## Core Commands

### Training
```bash
# Default training (RoBERTa)
PYTHONPATH=. python src/train.py
# OR
make train

# Train with specific encoder
make train-bert      # BERT
make train-deberta   # DeBERTa
make train-roberta   # RoBERTa

# Custom parameters
PYTHONPATH=. python src/train.py training.batch_size=16 training.learning_rate=1e-5

# Override model configuration
PYTHONPATH=. python src/train.py model=bert_base training.max_epochs=10
```

### Hyperparameter Optimization
```bash
PYTHONPATH=. python src/hpo.py
# OR
make hpo
```

### MLflow Tracking
```bash
make mlflow-server
# Access UI at http://localhost:5000
```

### Configuration Validation
```bash
make validate-config
```

### Code Quality
```bash
make format    # Black formatting
make lint      # Ruff linting
make check     # Both
```

## Architecture Overview

### Multi-Task Model Structure
The `EvidenceModel` (src/models/model.py) is a multi-head architecture:
- **Encoder**: Shared transformer base (BERT/RoBERTa/DeBERTa) with optional LoRA
- **Symptom Classification Head**: Multi-label classification (10 mental health symptoms)
- **Token Classification Head**: Token-level evidence extraction (optional)
- **Span Classification Head**: Span-level evidence extraction (start/end positions, optional)

All heads share the same encoder output but have independent classification layers.

### Hydra Configuration Composition
Configuration uses composition with separate groups (`configs/`):
- `config.yaml`: Main config with defaults
- `data/default.yaml`: Dataset paths, splits, max_length
- `model/{bert_base,roberta_base,deberta_base}.yaml`: Model architecture per encoder
- `training/default.yaml`: Training hyperparameters, optimizer, scheduler
- `hpo/default.yaml`: Optuna search space and study settings

Override any parameter: `python src/train.py model=bert_base training.batch_size=32`

### Data Pipeline Flow
1. **Loading** (src/data/dataset.py): Posts CSV + groundtruth JSON → merged DataFrame
2. **Splitting**: Train/val/test split with configurable ratios (default 0.1/0.1)
3. **Dataset**: `PostDataset` returns raw text + multi-labels + optional evidence annotations
4. **Collation**: `TokenizedDataCollator` tokenizes on-the-fly, converts char spans to token labels
5. **DataModule**: Encapsulates entire pipeline, returns train/val/test loaders

The collator handles evidence mapping from character-level spans to token-level labels automatically.

### Loss Computation Architecture
Multi-task loss (src/utils/training.py `compute_loss`):
- **Symptom labels**: Binary cross-entropy with adaptive focal loss (handles class imbalance)
- **Token evidence**: Cross-entropy (if enabled)
- **Span evidence**: Two cross-entropy losses for start/end positions (if enabled)

All losses are weighted (`training.loss_weights`) and summed. Focal loss gamma adapts based on positive prediction rate to combat label imbalance.

### Encoder Factory Pattern
`encoder_factory.py` + individual encoder modules (bert_encoder.py, etc.) abstract encoder creation:
- Returns (model, hidden_size) tuple
- Handles gradient checkpointing configuration
- Each encoder type has custom handling (e.g., DeBERTa pooler vs. BERT/RoBERTa)

### LoRA Fine-Tuning Integration
When `model.encoder.lora.enabled=true`, the encoder is wrapped with PEFT LoRA:
- Applied in `EvidenceModel._maybe_apply_lora()`
- Targets query/key/value by default
- Can freeze base encoder and only train LoRA adapters
- Configured per model in model/*.yaml

## Key Implementation Details

### Mixed Precision Training
Automatically enabled for CUDA GPUs:
- Uses AMP (Automatic Mixed Precision)
- Prefers BF16 if GPU supports it (RTX 3090/5090+), otherwise FP16
- Scaler only used for FP16 (not needed for BF16)

### Gradient Accumulation
Training loop accumulates gradients over `gradient_accumulation_steps` before optimizer step. Effective batch size = `batch_size * gradient_accumulation_steps`.

### Early Stopping Logic
Monitors `val_symptom_labels_macro_f1` by default:
- Saves best checkpoint when metric improves by > `min_delta`
- Stops if no improvement for `patience` epochs
- Best model is loaded for final test evaluation

### EMA (Exponential Moving Average)
When `training.ema_decay > 0`:
- Shadow weights maintained alongside model
- Updated after each optimizer step
- Applied during validation/test evaluation
- Restored after evaluation

### Scheduler Types
Multiple schedulers supported (src/utils/schedulers.py):
- `linear`: Linear warmup + linear decay
- `cosine`: Linear warmup + cosine annealing
- `onecycle`: OneCycleLR scheduler
- `plateau`: ReduceLROnPlateau (steps on validation metric, not per batch)

Note: `ReduceLROnPlateau` is stepped differently (after validation, not after optimizer step).

### HPO Search Space
Optuna trials suggest hyperparameters from `configs/hpo/default.yaml`:
- Encoder type (pretrained model)
- Learning rate, batch size, weight decay
- Dropout rates, number of classification layers
- LoRA parameters (r, alpha, dropout)
- Focal loss parameters

The objective function in `src/hpo.py` creates trial configs, runs `train_loop`, and returns best validation metric.

### Multi-Label Thresholds
Per-symptom classification thresholds configurable in model configs:
```yaml
model.heads.symptom_labels.thresholds:
  ANHEDONIA: 0.5
  DEPRESSED_MOOD: 0.5
  # ... other symptoms
```
If not specified, defaults to 0.5 for all labels.

## Jupyter Notebook System

The project includes 9 interactive notebooks for development:
1. **01_Configuration_Management.ipynb**: Widget-based config builder
2. **02_Enhanced_Checkpoint_System.ipynb**: Auto-resume checkpoint management
3. **03_Main_Training.ipynb**: Training with real-time monitoring
4. **04_HPO_Optimization.ipynb**: Optuna optimization with auto-resume
5. **05_Code_Verification.ipynb**: System integrity testing
6. **06_Data_Processing_and_Exploration.ipynb**: Data analysis
7. **07_Model_Evaluation.ipynb**: Model evaluation and comparison
8. **08_Model_Architecture_Explorer.ipynb**: Architecture exploration
9. **09_Utilities_and_Helpers.ipynb**: Utility testing

Notebooks have auto-resume capabilities that detect interrupted sessions and restore complete training state.

## Mental Health Symptom Labels

The system classifies 10 DSM-5 mental health symptoms:
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

These are configured in `data.multi_label_fields` and must match the column names in the dataset.

## Common Development Patterns

### Adding a New Encoder
1. Create encoder module in `src/models/` (follow bert_encoder.py pattern)
2. Register in `encoder_factory.py`
3. Add config file in `configs/model/`
4. Return (model, hidden_size) tuple from build function

### Modifying Loss Function
Edit `src/utils/training.py` `compute_loss()`:
- Handles all head outputs and targets
- Applies loss weights and focal loss
- Returns scalar loss for backprop

### Changing Metrics
Edit `src/utils/metrics.py` and `src/utils/training.py` `evaluate()`:
- Metrics computed per head
- Macro-averaged F1 is primary metric for early stopping
- Results logged to MLflow

### Environment Variables
```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT_NAME="custom_experiment"
export OPTUNA_STORAGE_URL="sqlite:///optuna.db"
export CUDA_VISIBLE_DEVICES=0
```

## Troubleshooting

### PYTHONPATH Issues
Always set PYTHONPATH when running scripts:
```bash
PYTHONPATH=. python src/train.py
```
Or use Makefile commands which handle this automatically.

### OOM (Out of Memory)
Reduce memory usage:
- Decrease `training.batch_size`
- Increase `training.gradient_accumulation_steps`
- Enable `model.encoder.gradient_checkpointing=true`
- Reduce `data.max_length`
- Use smaller model (BERT instead of DeBERTa)

### Hydra Output Directories
Hydra creates output dirs at `outputs/{encoder_type}/{date}/{time}/`:
- `.hydra/` contains resolved config
- `metrics.json` contains final results
- Use `hydra.run.dir` to customize

### ReduceLROnPlateau Not Stepping
This scheduler steps on validation metric, not per batch. The training loop handles this differently - check training.py lines 196-200.
