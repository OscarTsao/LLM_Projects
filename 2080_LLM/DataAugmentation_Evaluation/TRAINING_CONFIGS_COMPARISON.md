# Training Configurations Comparison

## Is trial_0043 the best?

**Yes**, trial_0043 achieved the best performance with:
- **ROC-AUC: 0.943** (validation set)
- Accuracy: 0.882
- Precision: 0.840
- Recall: 0.842
- F1: 0.841

This was the result of hyperparameter optimization across 500 trials.

---

## Configuration Comparison by Command

### 1. `make train` (Default Configuration)

**Config File**: `conf/config.yaml` (composed from defaults)

**Key Settings**:
- **Model**: BERT-base-uncased (`google-bert/bert-base-uncased`)
- **Dataset**: Original only (no augmentation)
- **Batch Size**: 32
- **Learning Rate**: 2e-5
- **Epochs**: 100 (with early stopping, patience=20)
- **Max Seq Length**: 256
- **Optimizer**: AdamW (PyTorch)
- **Classifier**: No hidden layers (direct from CLS token)
- **Dropout**: 0.1
- **Seed**: 1337
- **Early Stopping**: Yes, patience=20

**Best For**: Standard training, baseline comparisons

---

### 2. `make train-best` (Optimized Configuration)

**Config File**: `conf/best_config.yaml` (from trial_0043)

**Key Settings**:
- **Model**: DeBERTa-base (`microsoft/deberta-base`)
- **Dataset**: Original + Hybrid augmentation
- **Batch Size**: 8
- **Learning Rate**: 2.78e-5
- **Epochs**: 100 (with early stopping, patience=20)
- **Max Seq Length**: 384
- **Optimizer**: AdamW (HuggingFace)
- **Classifier**: [448, 256] hidden layers
- **Dropout**: 0.215
- **Seed**: 4029
- **Early Stopping**: Yes, patience=20
- **Special**: Uses gradient checkpointing

**Best For**: Production training, achieving best performance

---

### 3. `make train-criteria` (Criteria Matching Agent)

**Config File**: Multi-agent system uses `conf/agent/criteria.yaml`

**Key Settings**:
- **Model**: BERT-base-uncased
- **Dataset**: ReDSM-5 posts with criteria matching labels
- **Batch Size**: 32
- **Learning Rate**: 2e-5
- **Epochs**: 100
- **Max Seq Length**: 512
- **Loss**: Adaptive Focal Loss
- **Classifier**: [256] hidden layer
- **Task**: Binary classification (criterion matches post or not)
- **Metric**: F1 score

**Best For**: Training only the criteria matching component

---

### 4. `make train-evidence` (Evidence Binding Agent)

**Config File**: Multi-agent system uses `conf/agent/evidence.yaml`

**Key Settings**:
- **Model**: BERT-base-uncased
- **Dataset**: ReDSM-5 posts with evidence span annotations
- **Batch Size**: 32
- **Learning Rate**: 2e-5
- **Epochs**: 100
- **Max Seq Length**: 512
- **Task**: Token-level classification (start/end of evidence spans)
- **Max Span Length**: 50 tokens
- **Metric**: Span F1 score

**Best For**: Training only the evidence extraction component

---

### 5. `make train-joint` (Joint Training)

**Config File**: Multi-agent system uses `conf/agent/joint.yaml`

**Key Settings**:
- **Model**: BERT-base-uncased (shared encoder)
- **Dataset**: ReDSM-5 posts with both criteria and evidence labels
- **Batch Size**: 32
- **Learning Rate**: 2e-5
- **Epochs**: 100
- **Max Seq Length**: 512
- **Tasks**:
  - Criteria matching (with Adaptive Focal Loss)
  - Evidence span detection
- **Loss Weights**: 0.5 criteria + 0.5 evidence
- **Metric**: Combined F1 score

**Best For**: Training both agents jointly with a shared encoder

---

## Quick Comparison Table

| Command | Model | Dataset | Epochs | Early Stop | Max Seq Len | Purpose |
|---------|-------|---------|--------|------------|-------------|---------|
| `make train` | BERT-base | Original only | 100 | patience=20 | 256 | Standard baseline |
| `make train-best` | **DeBERTa-base** | **Original + Hybrid** | **100** | **patience=20** | **384** | **Best performance** |
| `make train-criteria` | BERT-base | Criteria matching | 100 | patience=10 | 512 | Criteria agent only |
| `make train-evidence` | BERT-base | Evidence spans | 100 | patience=10 | 512 | Evidence agent only |
| `make train-joint` | BERT-base | Both tasks | 100 | patience=10 | 512 | Multi-task learning |

---

## Key Differences: `make train` vs `make train-best`

### Why `train-best` is better:

1. **Better Model**: DeBERTa-base vs BERT-base
   - DeBERTa has enhanced attention mechanism
   - Better pre-training on larger corpus

2. **Data Augmentation**: Includes hybrid augmented data vs original only
   - More training examples
   - Better generalization from data diversity

3. **Optimized Hyperparameters**: Found through 500 HPO trials
   - Learning rate: 2.78e-5 (vs 2e-5)
   - Dropout: 0.215 (vs 0.1)
   - Hidden layers: [448, 256] (vs none)
   - Adam epsilon: 9.69e-9 (vs 1e-8)

4. **Larger Context**: 384 tokens vs 256 tokens
   - Can process longer posts

5. **Gradient Checkpointing**: Enabled
   - Allows training with limited GPU memory

### When to use each:

- **`make train`**: Standard training with BERT baseline, original data only
- **`make train-best`**: Best model training with optimized hyperparameters and augmented data

---

## Override Examples

You can customize any training command:

```bash
# Use best config but with different epochs
python -m src.training.train --config-name=best_config model.num_epochs=50

# Use default config but with DeBERTa model
python -m src.training.train model.pretrained_model_name=microsoft/deberta-base

# Use best config but with original data only
python -m src.training.train --config-name=best_config dataset=original
```
