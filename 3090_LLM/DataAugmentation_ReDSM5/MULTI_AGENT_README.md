# Multi-Agent Psychiatric Diagnosis System

## Overview

This project implements a unified multi-agent system for psychiatric diagnosis using DSM-5 criteria. The system combines two specialized agents:

1. **Criteria Matching Agent**: Determines if DSM-5 criteria match a given post
2. **Evidence Binding Agent**: Predicts start and end tokens of evidence sentences when criteria matching is positive

## Architecture

### Multi-Agent Pipeline
```
Input Post + DSM-5 Criteria
           ↓
    Criteria Matching Agent
           ↓
    [If Match Detected]
           ↓
    Evidence Binding Agent
           ↓
    Output: {
        criteria_match: bool,
        confidence: float,
        evidence_spans: List[Tuple[int, int]]
    }
```

### Training Modes

The system supports three distinct training modes:

- **Mode 1**: Train criteria matching agent independently
- **Mode 2**: Train evidence binding agent independently  
- **Mode 3**: Train both agents jointly (end-to-end)

Each mode supports both standard training and Optuna hyperparameter optimization.

## Quick Start

### 1. Environment Setup

```bash
# Create and activate environment
make env-create
mamba activate redsm5

# Install dependencies
make pip-sync
```

### 2. Validate Integration

```bash
# Run comprehensive integration tests
python validate_integration.py
```

### 3. Training

```bash
# Mode 1: Criteria matching only
make train-criteria

# Mode 2: Evidence binding only
make train-evidence

# Mode 3: Joint training
make train-joint
```

### 4. Hyperparameter Optimization

```bash
# HPO for each mode (500 trials by default)
make train-criteria-optuna
make train-evidence-optuna
make train-joint-optuna
```

### 5. Monitoring

```bash
# MLflow dashboard
make mlflow-ui

# Optuna dashboard
make optuna-dashboard

# TensorBoard
make tensorboard
```

## Configuration

### Agent Configurations

- `conf/agent/criteria.yaml`: Criteria matching agent settings
- `conf/agent/evidence.yaml`: Evidence binding agent settings
- `conf/agent/joint.yaml`: Joint training settings

### Training Mode Configurations

- `conf/training_mode/criteria.yaml`: Mode 1 configuration
- `conf/training_mode/evidence.yaml`: Mode 2 configuration
- `conf/training_mode/joint.yaml`: Mode 3 configuration

### Key Parameters

#### Criteria Matching Agent
```yaml
model_name: google-bert/bert-base-uncased
max_seq_length: 512
classifier_hidden_sizes: [256]
loss_type: adaptive_focal  # bce, focal, adaptive_focal
learning_rate: 2e-5
batch_size: 32
num_epochs: 100
```

#### Evidence Binding Agent
```yaml
model_name: google-bert/bert-base-uncased
max_seq_length: 512
span_threshold: 0.5
max_span_length: 50
label_smoothing: 0.0
learning_rate: 2e-5
batch_size: 32
num_epochs: 100
```

#### Joint Training
```yaml
shared_encoder: true
criteria_loss_weight: 0.5
evidence_loss_weight: 0.5
freeze_encoder_epochs: 0
```

## Performance Optimizations

### GPU Optimizations (RTX 3090/5090)
- TF32 enabled for faster matrix operations
- cuDNN benchmark mode for consistent input sizes
- Mixed precision training (fp16/bf16)
- Gradient checkpointing for memory efficiency
- torch.compile for faster inference (PyTorch 2.0+)

### Training Optimizations
- Gradient accumulation for large effective batch sizes
- Early stopping with configurable patience
- Learning rate scheduling (linear, cosine, polynomial)
- Adaptive loss functions (focal, adaptive focal)

## Hyperparameter Search Space

### Criteria Agent
- Learning rate: [1e-6, 5e-5] (log scale)
- Batch size: [8, 16, 32, 64, 128]
- Dropout: [0.0, 0.4]
- Loss function: [BCE, Focal, Adaptive Focal]
- Architecture: [Direct, Small, Medium, Large]

### Evidence Agent
- Learning rate: [1e-6, 5e-5] (log scale)
- Batch size: [8, 16, 32, 64]
- Span threshold: [0.3, 0.7]
- Label smoothing: [0.0, 0.2]
- Max span length: [20, 100]

### Joint Training
- Task loss weights: [0.1, 0.9] for criteria vs evidence
- Shared vs separate encoders
- Encoder freezing epochs: [0, 5]

## Data Pipeline

### Criteria Matching Data
- Source: `Data/GroundTruth/Final_Ground_Truth.json`
- Format: (post, criterion) → binary label
- Uses existing augmentation pipeline

### Evidence Binding Data
- Source: `Data/ReDSM5/redsm5_annotations.csv`
- Format: (post, criterion, sentence_text) → (start_token, end_token)
- Extracts sentence spans from posts automatically

### Joint Training Data
- Combines both data sources
- Aligns criteria labels with evidence spans
- Multi-task format with shared inputs

## MLflow Integration

### Experiment Organization
- `redsm5-criteria`: Mode 1 experiments
- `redsm5-evidence`: Mode 2 experiments  
- `redsm5-joint`: Mode 3 experiments
- `redsm5-optuna-*`: HPO experiments

### Automatic Logging
- All training runs automatically logged
- Hyperparameters, metrics, and artifacts tracked
- Model checkpoints and configurations saved
- Optuna study results exported

## Available Commands

### Training
```bash
make train-criteria          # Mode 1: Criteria only
make train-evidence          # Mode 2: Evidence only  
make train-joint            # Mode 3: Joint training
```

### Hyperparameter Optimization
```bash
make train-criteria-optuna   # Mode 1 HPO
make train-evidence-optuna   # Mode 2 HPO
make train-joint-optuna     # Mode 3 HPO
```

### Evaluation
```bash
make evaluate-criteria       # Evaluate criteria agent
make evaluate-evidence       # Evaluate evidence agent
make evaluate-joint         # Evaluate joint model
```

### Monitoring
```bash
make mlflow-ui              # MLflow dashboard (http://localhost:5000)
make optuna-dashboard       # Optuna dashboard (http://localhost:8080)
make tensorboard           # TensorBoard (http://localhost:6006)
```

### Development
```bash
make lint                   # Code linting
make format                 # Code formatting
make test                   # Run tests
make clean                  # Clean cache files
```

## Success Metrics

### Criteria Matching
- F1 score > 0.85
- AUC > 0.90
- Balanced precision/recall

### Evidence Binding
- Span-level F1 > 0.75
- Token-level accuracy > 0.90
- Exact match > 0.60

### Joint Training
- Maintains criteria performance
- Achieves evidence binding targets
- Faster inference than sequential pipeline

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or enable gradient checkpointing
2. **Slow training**: Enable mixed precision and torch.compile
3. **Poor convergence**: Adjust learning rate or try different schedulers
4. **Data loading errors**: Check file paths in configuration

### Performance Tips

1. Use larger batch sizes with gradient accumulation
2. Enable all hardware optimizations for RTX 3090/5090
3. Use early stopping to prevent overfitting
4. Monitor MLflow for experiment tracking

## Development Guidelines

### Code Quality
- Follow Black formatting (88 columns)
- Use Ruff for linting
- Type hints for all public functions
- Comprehensive test coverage

### Testing
```bash
# Run all tests
make test

# Run specific test
pytest tests/agents/test_criteria_matching.py

# Run integration validation
python validate_integration.py
```

### Contributing
1. Create feature branch
2. Implement changes with tests
3. Run validation script
4. Submit pull request with metrics

## Next Steps

1. **Validation**: Run `python validate_integration.py`
2. **Training**: Test all three modes with `make train-*`
3. **Optimization**: Run HPO studies with `make train-*-optuna`
4. **Monitoring**: Use `make mlflow-ui` to track experiments
5. **Evaluation**: Assess model performance on test sets
