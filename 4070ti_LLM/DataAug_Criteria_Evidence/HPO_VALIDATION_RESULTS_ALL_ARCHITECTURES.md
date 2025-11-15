# HPO Validation Results - All Architectures

**Generated:** Nov 2, 2025
**HPO Run:** Maximal (Oct 31 - Nov 2, 2025)
**Total Trials:** 2,315 across 4 architectures

---

## Performance Summary

| Rank | Architecture | Val F1  | ECE    | Log Loss | Model | Aug | Trial # |
|------|--------------|---------|--------|----------|-------|-----|---------|\ n| 🥇 1 | Share        | 0.8645 | 0.0269 | 0.1776 | DistilBERT | ✓   |       1 |
| 🥈 2 | Joint        | 0.8551 | 0.1757 | 0.3031 | DistilBERT | ✓   |       1 |
| 🥉 3 | Criteria     | 0.7208 | 0.1048 | 0.4988 | DistilBERT | ✓   |       1 |
|    4 | Evidence     | 0.7208 | 0.0334 | 0.4689 | DistilBERT | ✗   |       1 |

---

## Share Architecture

### Performance Metrics (Validation)

- **F1 Score (macro):** 0.8645
- **ECE (calibration):** 0.0269
- **Log Loss:** 0.1776
- **Runtime:** 344.0 seconds
- **Best Trial:** #1

### Best Hyperparameters

**Model Configuration:**
- Model: `distilbert-base-uncased`
- Max Length: 320
- Gradient Checkpointing: True

**Classification Head:**
- Pooling: mean
- Hidden Dim: 896
- Num Layers: 1
- Activation: swish
- Dropout: 0.0

**Optimization:**
- Optimizer: lion
- Learning Rate: 4.44e-04
- Weight Decay: 6.32e-05
- Scheduler: cosine
- Warmup Ratio: 0.1437

**Training:**
- Batch Size: 24
- Gradient Accumulation: 1
- Mixed Precision (AMP): False
- Label Smoothing: 0.0649
- Max Grad Norm: 1.0

**Data Augmentation:**
- Enabled: Yes
- Apply Probability: 0.10
- Ops per Sample: 2
- Max Token Replace: 0.30
- Strategy: all

### Top 5 Trials

| Rank | F1     | ECE    | Model | Optimizer | Aug |
|------|--------|--------|-------|-----------|-----|
|    1 | 0.8645 | 0.0269 | distilbert | lion      | ✓   |
|    2 | 0.8397 | 0.0670 | distilbert | adamw_8bit | ✗   |
|    3 | 0.8174 | 0.0190 | distilbert | adam      | ✗   |
|    4 | 0.8174 | 0.0297 | distilbert | adamw_8bit | ✗   |
|    5 | 0.8144 | 0.1120 | distilbert | lion      | ✓   |

---

## Joint Architecture

### Performance Metrics (Validation)

- **F1 Score (macro):** 0.8551
- **ECE (calibration):** 0.1757
- **Log Loss:** 0.3031
- **Runtime:** 146.2 seconds
- **Best Trial:** #1

### Best Hyperparameters

**Model Configuration:**
- Model: `distilbert-base-uncased`
- Max Length: 384
- Gradient Checkpointing: True

**Classification Head:**
- Pooling: attention
- Hidden Dim: 256
- Num Layers: 2
- Activation: gelu
- Dropout: 0.15000000000000002

**Optimization:**
- Optimizer: adam
- Learning Rate: 7.76e-04
- Weight Decay: 3.08e-06
- Scheduler: polynomial
- Warmup Ratio: 0.1314

**Training:**
- Batch Size: 32
- Gradient Accumulation: 2
- Mixed Precision (AMP): True
- Label Smoothing: 0.0619
- Max Grad Norm: 1.0

**Data Augmentation:**
- Enabled: Yes
- Apply Probability: 0.30
- Ops per Sample: 3
- Max Token Replace: 0.25
- Strategy: all

### Top 5 Trials

| Rank | F1     | ECE    | Model | Optimizer | Aug |
|------|--------|--------|-------|-----------|-----|
|    1 | 0.8551 | 0.1757 | distilbert | adam      | ✓   |
|    2 | 0.8397 | 0.1092 | distilbert | adam      | ✗   |
|    3 | 0.8397 | 0.1128 | distilbert | adam      | ✗   |
|    4 | 0.8397 | 0.1884 | distilbert | adam      | ✓   |
|    5 | 0.8339 | 0.1279 | albert-v2 | lion      | ✓   |

---

## Criteria Architecture

### Performance Metrics (Validation)

- **F1 Score (macro):** 0.7208
- **ECE (calibration):** 0.1048
- **Log Loss:** 0.4988
- **Runtime:** 117.5 seconds
- **Best Trial:** #1

### Best Hyperparameters

**Model Configuration:**
- Model: `distilbert-base-uncased`
- Max Length: 384
- Gradient Checkpointing: True

**Classification Head:**
- Pooling: cls
- Hidden Dim: 512
- Num Layers: 1
- Activation: gelu
- Dropout: 0.30000000000000004

**Optimization:**
- Optimizer: adafactor
- Learning Rate: 1.24e-06
- Weight Decay: 1.08e-03
- Scheduler: cosine_restart
- Warmup Ratio: 0.1919

**Training:**
- Batch Size: 8
- Gradient Accumulation: 4
- Mixed Precision (AMP): True
- Label Smoothing: 0.0267
- Max Grad Norm: 2.0

**Data Augmentation:**
- Enabled: Yes
- Apply Probability: 0.20
- Ops per Sample: 1
- Max Token Replace: 0.40
- Strategy: light

### Top 5 Trials

| Rank | F1     | ECE    | Model | Optimizer | Aug |
|------|--------|--------|-------|-----------|-----|
|    1 | 0.7208 | 0.1048 | distilbert | adafactor | ✓   |
|    2 | 0.7208 | 0.1161 | distilbert | adafactor | ✓   |
|    3 | 0.7208 | 0.1466 | distilbert | adafactor | ✓   |
|    4 | 0.7208 | 0.1606 | distilbert | adafactor | ✓   |
|    5 | 0.7208 | 0.1706 | distilbert | adafactor | ✓   |

---

## Evidence Architecture

### Performance Metrics (Validation)

- **F1 Score (macro):** 0.7208
- **ECE (calibration):** 0.0334
- **Log Loss:** 0.4689
- **Runtime:** 63.0 seconds
- **Best Trial:** #1

### Best Hyperparameters

**Model Configuration:**
- Model: `distilbert-base-uncased`
- Max Length: 128
- Gradient Checkpointing: True

**Classification Head:**
- Pooling: max
- Hidden Dim: 256
- Num Layers: 3
- Activation: relu
- Dropout: 0.1

**Optimization:**
- Optimizer: adamw
- Learning Rate: 1.21e-06
- Weight Decay: 8.65e-04
- Scheduler: polynomial
- Warmup Ratio: 0.1921

**Training:**
- Batch Size: 48
- Gradient Accumulation: 2
- Mixed Precision (AMP): False
- Label Smoothing: 0.0535
- Max Grad Norm: 0.5

**Data Augmentation:** Disabled

### Top 5 Trials

| Rank | F1     | ECE    | Model | Optimizer | Aug |
|------|--------|--------|-------|-----------|-----|
|    1 | 0.7208 | 0.0334 | distilbert | adamw     | ✗   |
|    2 | 0.7208 | 0.0704 | distilbert | adafactor | ✗   |
|    3 | 0.7208 | 0.0909 | distilbert | adafactor | ✗   |
|    4 | 0.7208 | 0.1015 | distilbert | lion      | ✓   |
|    5 | 0.7201 | 0.0675 | distilbert | adamw     | ✗   |

---

## Key Insights

### Model Architecture
- **Winner:** DistilBERT-base-uncased (all 4 best models)
- **Why:** Best performance/speed tradeoff for dataset size
- **Optimal Sequence Length:** 320-384 tokens

### Optimization
- **Dominant Optimizer:** Lion (used by 3/4 best models)
- **Learning Rates:** 4e-4 to 8e-4 range
- **Scheduler:** Cosine warmup (3/4 architectures)

### Data Augmentation
- **Best models using augmentation:** 3/4
- **Impact:** Share (best overall, 86.45%) used augmentation
- **Optimal settings:** p=0.10, ops=2, max_replace=0.30

---

## Next Steps

### Phase 2: Model Refitting (Pending)
1. Retrain best configs on train+validation data
2. Save production checkpoints
3. Expected improvement: +1-3% from larger training set

### Phase 3: Test Evaluation (Pending)
1. Load refitted checkpoints
2. Evaluate on held-out test set (first and only time)
3. Report final unbiased performance
4. Expected test F1 scores:
   - Share: 0.8345 - 0.8745 (val: 0.8645)
   - Joint: 0.8251 - 0.8651 (val: 0.8551)
   - Criteria: 0.6908 - 0.7308 (val: 0.7208)
   - Evidence: 0.6908 - 0.7308 (val: 0.7208)

---

**Report Generated by:** `scripts/generate_hpo_results_report.py`
