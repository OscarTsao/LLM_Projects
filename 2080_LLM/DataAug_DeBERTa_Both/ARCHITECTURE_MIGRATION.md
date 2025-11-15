# Architecture Migration Summary

## Overview

This document summarizes the major architectural changes made to transition from a multi-task learning model with separate criteria matching and evidence span extraction to a unified dual-agent model focused on criteria classification with integrated evidence embeddings.

---

## Key Changes

### 1. Model Architecture Refactoring

#### Before (Multi-Task Model):
```
Input Text
    ↓
  Encoder (Single)
    ├─→ Criteria Head → Classification Logits
    └─→ Evidence Head → Span Start/End Logits
```

#### After (Dual-Agent Model):
```
Input Text
    ├─→ Criteria Encoder → Criteria Embeddings ──┐
    │                                             ↓
    └─→ Evidence Encoder → Evidence Embeddings → COMBINE → Classification
```

**Two Encoder Modes:**
- **Shared Mode**: Single DeBERTa-v3-base encoder for both agents
- **Separate Mode**: Two independent DeBERTa-v3-base encoders

**Two Combination Methods:**
- **Concat**: Concatenate embeddings (input_size = 2 × hidden_size)
- **Add**: Element-wise addition (input_size = hidden_size)

---

### 2. Model Selection

**Changed From:**
- Configurable models (mental-bert, psychbert, clinical-bert, bert-base, roberta-base)

**Changed To:**
- **DeBERTa-v3-base only** (`microsoft/deberta-v3-base`)

---

### 3. Evidence Head Transformation

#### Before (EvidenceBindingHead):
- **Purpose**: Span extraction for evidence localization
- **Architecture**: `hidden_states → Linear(start), Linear(end) → start_logits, end_logits`
- **Output**: Start/end positions for evidence spans
- **Loss**: Cross-entropy on start/end positions

#### After (EvidenceEmbeddingHead):
- **Purpose**: Generate evidence embeddings
- **Architecture**: `hidden_states → Pool → Dropout → embeddings`
- **Output**: Fixed-size evidence embeddings [batch_size, hidden_size]
- **Loss**: No direct loss (embeddings used to improve criteria classification)

**Code Changes:**
- File: `src/dataaug_multi_both/models/heads/evidence_binding.py`
- Renamed class: `EvidenceBindingHead` → `EvidenceEmbeddingHead`
- Removed: `start_classifier`, `end_classifier`, span extraction methods
- Added: `pool_encoder_outputs()` with support for cls/mean/max pooling

---

### 4. Criteria Head Enhancement

#### Before (CriteriaMatchingHead):
- **Input**: Encoder outputs only
- **Architecture**: `pooled_embedding → Dropout → Linear → logits`
- **Classifier Input Size**: `hidden_size`

#### After (Enhanced CriteriaMatchingHead):
- **Input**: Encoder outputs + Evidence embeddings
- **Architecture**: `criteria_embed + evidence_embed → Combine → Dropout → Linear → logits`
- **Classifier Input Size**:
  - `concat` mode: `2 × hidden_size`
  - `add` mode: `hidden_size`

**Code Changes:**
- File: `src/dataaug_multi_both/models/heads/criteria_matching.py`
- Added: `combination_method` parameter
- Added: `combine_embeddings()` method
- Modified: `forward()` to accept `evidence_embedding`
- Updated: Classifier layer size based on combination method

---

### 5. Loss Function Simplification

#### Before (MultiTaskLoss):
```python
total_loss = criteria_weight × criteria_loss + evidence_weight × evidence_loss
```
- Criteria loss: BCE/Focal/Weighted BCE
- Evidence loss: CrossEntropy(start_logits) + CrossEntropy(end_logits)

#### After (DualAgentLoss):
```python
loss = criteria_loss_fn(criteria_logits, criteria_labels)
```
- Only criteria loss (evidence embeddings contribute implicitly)
- Evidence loss removed (no longer a separate task)

**Code Changes:**
- File: `src/dataaug_multi_both/training/losses.py`
- New class: `DualAgentLoss` (simplified loss)
- Backward compatibility: `MultiTaskLoss` now inherits from `DualAgentLoss`
- Removed: Evidence binding loss computation
- Simplified: Forward method now only takes criteria logits/labels

---

### 6. HPO Search Space Updates

#### New Hyperparameters:

| Hyperparameter | Type | Options | Description |
|---|---|---|---|
| `encoder_mode` | categorical | shared, separate | Encoder sharing mode |
| `combination_method` | categorical | concat, add | Embedding combination strategy |
| `model_name` | categorical | deberta-v3-base | Fixed to DeBERTa-v3-base |

#### Removed Hyperparameters:
- Multiple model options (mental-bert, psychbert, etc.) - consolidated to DeBERTa-v3-base

**Code Changes:**
- File: `src/dataaug_multi_both/hpo/search_space.py`
- Added: `encoder_mode` suggestion
- Added: `combination_method` suggestion
- Modified: `model_name` to only suggest `deberta-v3-base`

---

### 7. Configuration Files

#### New Configuration: `configs/model/deberta_v3_base.yaml`
```yaml
name: "microsoft/deberta-v3-base"
encoder_mode: "shared"  # or "separate"
combination_method: "concat"  # or "add"

encoder:
  model_id: "microsoft/deberta-v3-base"
  cache_dir: "~/.cache/huggingface"
  gradient_checkpointing: false

heads:
  criteria_matching:
    num_labels: 9
    dropout: 0.1
    pooling_strategy: "cls"
    combination_method: "concat"

  evidence_embedding:
    dropout: 0.1
    pooling_strategy: "cls"
```

---

### 8. Model Output Changes

#### Before (MultiTaskModelOutput):
```python
@dataclass
class MultiTaskModelOutput:
    criteria_logits: Tensor  # [batch, 9]
    start_logits: Tensor     # [batch, seq_len]
    end_logits: Tensor       # [batch, seq_len]
    span_predictions: Tuple[Tensor, Tensor]  # Optional
```

#### After (DualAgentModelOutput):
```python
@dataclass
class DualAgentModelOutput:
    criteria_logits: Tensor           # [batch, 9]
    evidence_embedding: Tensor        # [batch, hidden_size]
    criteria_encoder_outputs: Dict    # Optional
    evidence_encoder_outputs: Dict    # Optional
```

**Changes:**
- Removed: `start_logits`, `end_logits`, `span_predictions`
- Added: `evidence_embedding`
- Added: Separate encoder outputs for debugging

---

### 9. HFEncoder Wrapper Enhancement

**Code Changes:**
- File: `src/dataaug_multi_both/models/encoders/hf_encoder.py`
- Added: `__call__()` method for forward pass
- Added: `parameters()` method for optimization
- Enhancement: Can now be used directly in model forward pass

---

## Migration Impact

### Training Pipeline
- **Compatible**: Existing training code mostly compatible
- **Loss Computation**: Simplified (no evidence loss)
- **Data Loading**: No changes needed
- **Checkpointing**: Backward compatible

### Evaluation
- **Metrics**: Only criteria metrics (evidence metrics removed)
- **Predictions**: Only criteria predictions returned

### HPO
- **New Parameters**: `encoder_mode`, `combination_method`
- **Search Space**: Enlarged to explore encoder/combination strategies
- **Trials**: More combinations to explore

---

## Backward Compatibility

### Aliases Provided:
```python
MultiTaskModel = DualAgentModel
MultiTaskModelOutput = DualAgentModelOutput
create_multi_task_model = create_dual_agent_model
MultiTaskLoss = DualAgentLoss  # With deprecation warning
```

### Breaking Changes:
1. Evidence-related outputs removed from model output
2. Loss function signature changed (no more start/end positions)
3. Model factory function parameters changed

---

## File Summary

### Modified Files:
1. `src/dataaug_multi_both/models/multi_task_model.py` - Core model architecture
2. `src/dataaug_multi_both/models/heads/evidence_binding.py` - Evidence head refactored
3. `src/dataaug_multi_both/models/heads/criteria_matching.py` - Enhanced with combination
4. `src/dataaug_multi_both/models/encoders/hf_encoder.py` - Added forward capability
5. `src/dataaug_multi_both/training/losses.py` - Simplified loss function
6. `src/dataaug_multi_both/hpo/search_space.py` - New hyperparameters

### New Files:
1. `configs/model/deberta_v3_base.yaml` - DeBERTa-v3-base configuration

---

## Next Steps

### Remaining Tasks:
1. ✅ Core architecture refactored
2. ✅ Loss functions updated
3. ✅ HPO search space updated
4. ✅ Configuration files created
5. ⏳ Update training code/CLI integration
6. ⏳ Update tests
7. ⏳ Update documentation

### Testing Recommendations:
1. Test shared vs separate encoder modes
2. Test concat vs add combination methods
3. Validate model outputs match expected shapes
4. Run small HPO sweep to validate search space
5. Check backward compatibility with old configs

---

## Benefits of New Architecture

1. **Focused Objective**: Single clear goal (criteria classification)
2. **Evidence Integration**: Evidence informs criteria (not separate task)
3. **Flexibility**: Can switch between shared/separate encoders
4. **Combination Strategies**: Explore concat vs add embeddings
5. **Simplified Loss**: Easier to optimize (single objective)
6. **DeBERTa-v3**: State-of-the-art base model
7. **Cleaner Code**: Reduced complexity in loss computation

---

Generated: 2025-10-17
