# Multi-Agent Architecture Design

## Overview

This document outlines the design for integrating two psychiatric diagnosis agents into a unified multi-agent system within the ReDSM-5 Classification project.

## Agent Components

### 1. Criteria Matching Agent
- **Purpose**: Determine if DSM-5 criteria match a given post
- **Architecture**: BERT + binary classifier
- **Input**: `[CLS] post [SEP] criterion_text [SEP]`
- **Output**: Binary classification (match/no match) + confidence score

### 2. Evidence Binding Agent
- **Purpose**: Predict start and end tokens of evidence sentences when criteria matching is positive
- **Architecture**: BERT + token-level classification head
- **Input**: `[CLS] post [SEP] criterion_text [SEP]`
- **Output**: Start/end token positions for evidence spans

## Multi-Agent Pipeline Architecture

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
        evidence_spans: List[Tuple[int, int]]  # (start, end) positions
    }
```

## Training Modes

### Mode 1: Independent Criteria Training
- Train only the criteria matching agent
- Use existing ground truth labels from `Final_Ground_Truth.json`
- Standard binary classification loss

### Mode 2: Independent Evidence Training
- Train only the evidence binding agent
- Use sentence-level annotations from `redsm5_annotations.csv`
- Token-level classification loss for span prediction

### Mode 3: Joint End-to-End Training
- Train both agents simultaneously
- Multi-task loss: criteria classification + evidence span prediction
- Shared BERT encoder with separate task-specific heads

## Implementation Structure

```
src/
├── agents/
│   ├── __init__.py
│   ├── base.py                    # Base agent interface
│   ├── criteria_matching.py      # Criteria matching agent
│   ├── evidence_binding.py       # Evidence binding agent
│   └── multi_agent_pipeline.py   # Unified pipeline
├── training/
│   ├── train_criteria.py         # Mode 1 training
│   ├── train_evidence.py         # Mode 2 training
│   ├── train_joint.py            # Mode 3 training
│   ├── train_criteria_optuna.py  # Mode 1 HPO
│   ├── train_evidence_optuna.py  # Mode 2 HPO
│   └── train_joint_optuna.py     # Mode 3 HPO
├── data/
│   ├── evidence_loader.py        # Evidence span data loading
│   └── joint_dataset.py          # Multi-task dataset
└── models/
    ├── criteria_model.py         # Criteria matching model
    ├── evidence_model.py         # Evidence binding model
    └── joint_model.py            # Multi-task model
```

## Configuration Structure

```
conf/
├── agent/
│   ├── criteria.yaml            # Criteria agent config
│   ├── evidence.yaml            # Evidence agent config
│   └── joint.yaml               # Joint training config
├── training_mode/
│   ├── criteria.yaml            # Mode 1 config
│   ├── evidence.yaml            # Mode 2 config
│   └── joint.yaml               # Mode 3 config
└── optuna/
    ├── criteria_hpo.yaml        # Mode 1 HPO config
    ├── evidence_hpo.yaml        # Mode 2 HPO config
    └── joint_hpo.yaml           # Mode 3 HPO config
```

## Data Pipeline

### Criteria Matching Data
- Source: `Data/GroundTruth/Final_Ground_Truth.json`
- Format: (post, criterion) → binary label
- Existing pipeline can be reused

### Evidence Binding Data
- Source: `Data/ReDSM5/redsm5_annotations.csv`
- Process: Extract sentence spans from posts
- Format: (post, criterion, sentence_text) → (start_token, end_token)
- New pipeline needed

### Joint Training Data
- Combine both data sources
- Align criteria labels with evidence spans
- Multi-task format: (post, criterion) → (binary_label, span_positions)

## Performance Optimizations

### GPU Optimizations
- TF32 enabled for RTX 3090/5090
- cuDNN benchmark mode
- Mixed precision training (fp16/bf16)
- Gradient checkpointing for memory efficiency
- torch.compile for faster inference

### Training Optimizations
- Gradient accumulation for large effective batch sizes
- Early stopping with patience
- Learning rate scheduling (linear, cosine, polynomial)
- Adaptive loss functions (focal, adaptive focal)

## Hyperparameter Search Space

### Criteria Agent
- Learning rate: [1e-6, 5e-5]
- Batch size: [8, 16, 32, 64, 128]
- Dropout: [0.0, 0.4]
- Loss function: [BCE, Focal, Adaptive Focal]
- Optimizer: [AdamW, Adam]
- Scheduler: [Linear, Cosine, Polynomial]

### Evidence Agent
- Learning rate: [1e-6, 5e-5]
- Batch size: [8, 16, 32, 64]
- Dropout: [0.0, 0.4]
- Loss function: [CrossEntropy, Focal]
- Label smoothing: [0.0, 0.2]

### Joint Training
- Task loss weights: [0.1, 0.9] for criteria vs evidence
- Shared encoder learning rate: [1e-6, 3e-5]
- Task-specific head learning rates: [1e-5, 1e-4]

## MLflow Integration

### Experiment Organization
- `redsm5-criteria`: Mode 1 experiments
- `redsm5-evidence`: Mode 2 experiments  
- `redsm5-joint`: Mode 3 experiments
- `redsm5-optuna-*`: HPO experiments

### Metrics Tracking
- Criteria: accuracy, precision, recall, F1, AUC
- Evidence: token-level accuracy, span-level F1, exact match
- Joint: combined metrics + task-specific metrics

### Artifact Logging
- Model checkpoints
- Configuration files
- Test predictions
- Optuna study results

## Makefile Commands

```bash
# Training modes
make train-criteria          # Mode 1: Criteria only
make train-evidence          # Mode 2: Evidence only  
make train-joint            # Mode 3: Joint training

# HPO modes
make train-criteria-optuna   # Mode 1 HPO
make train-evidence-optuna   # Mode 2 HPO
make train-joint-optuna     # Mode 3 HPO

# Evaluation
make evaluate-criteria       # Evaluate criteria agent
make evaluate-evidence       # Evaluate evidence agent
make evaluate-joint         # Evaluate joint model

# Infrastructure
make mlflow-ui              # MLflow dashboard
make optuna-dashboard       # Optuna dashboard
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
