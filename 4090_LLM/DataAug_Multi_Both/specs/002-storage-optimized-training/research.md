# Research & Design Decisions

**Feature**: Storage-Optimized Training & HPO Pipeline
**Date**: 2025-10-10
**Phase**: 0 (Outline & Research)

## Overview

This document consolidates research findings and design decisions for the storage-optimized multi-task NLP training pipeline with extensive hyperparameter optimization capabilities.

---

## 1. HPO Framework Selection

### Decision: Optuna

**Rationale**:
- Native Python integration with PyTorch
- Supports sequential trial execution (our requirement from FR-021)
- Built-in pruning algorithms for early stopping of unpromising trials
- Lightweight compared to Ray Tune (simpler deployment, no distributed coordination overhead)
- Excellent support for categorical/conditional search spaces (needed for our complex architecture choices)
- Integration with MLflow via `optuna.integration.MLflowCallback`

**Alternatives Considered**:
- **Ray Tune**: More powerful for distributed/parallel trials, but adds complexity for sequential execution; our constraint of sequential trials makes Ray's distribution features unnecessary
- **Weights & Biases Sweeps**: Cloud-dependent, doesn't fit "local MLflow database" requirement (A-003)
- **Manual grid search**: Infeasible with 30+ models × architecture variations × loss functions × augmentation strategies

**Implementation Notes**:
- Use Optuna `Study` with SQLite storage for persistence and resume capability
- Configure `n_jobs=1` to enforce sequential execution
- Implement custom pruner based on disk space monitoring (10% threshold from FR-018)

---

## 2. Multi-Task Architecture Design

### Decision: Flexible Task Coupling with Optional Evidence-Guided Matching

**Rationale**:
- User requirements specify searching whether to connect tasks ("whether to connect two agents together")
- Evidence-first then criteria matching reflects clinical reasoning (find supporting text, then make diagnosis)
- Must support both independent and coupled modes to maximize HPO search space

**Architecture Components**:

#### Encoder (Shared or Separate)
- Single shared transformer encoder (memory-efficient, transfer learning across tasks)
- HPO searches across 30 transformer architectures

#### Task-Specific Heads:

1. **Criteria Matching Head** (9 binary classifiers or 1 multi-label classifier):
   - Input: Encoder hidden states
   - Pooling strategies (HPO):
     - CLS token
     - Mean pooling over sequence
     - Attention-weighted pooling
     - Last-2-layer scalar mix (weighted combination of last 2 encoder layers)
   - Architecture choices (HPO):
     - Linear (1 layer)
     - MLP (1 hidden layer, configurable dim)
     - MLP with residual connection (2 layers)
     - Gated head (learnable gating mechanism)
     - Multi-sample dropout (inference-time ensemble)

2. **Evidence Binding Head** (span extraction):
   - Input: Encoder hidden states (+ optional criteria matching embeddings if coupled)
   - Architecture choices (HPO):
     - Start/end linear (BERT QA-style)
     - Start/end MLP
     - Biaffine span scorer (scores all span combinations)
     - BIO tagging + CRF (sequence labeling)
     - Sentence-level reranker (rank evidence sentences)
   - Must support null span prediction (no evidence case)

#### Task Coupling Module (Optional, HPO-controlled):
- **When coupled**: Evidence head predictions inform criteria matching
- **Coupling method (HPO)**:
  - Concatenate [criteria_embed, evidence_embed]
  - Element-wise sum
  - Weighted sum (learnable weights)
  - Element-wise mean
  - Weighted mean
- **Pooling before combination (HPO)**:
  - Mean pooling
  - Max pooling
  - Attention-weighted pooling

**Alternatives Considered**:
- **Separate models per task**: Higher memory cost, no shared representations; rejected given 1-10GB model size constraint
- **Hard-coded coupling**: Reduces search space flexibility; violates user requirement to search coupling strategies

---

## 3. Input Formatting Strategy

### Decision: HPO-Searchable Format (Binary Post-Criterion Pairs vs Multi-Label)

**Rationale**:
- User explicitly requested searching "input format should also be searched in hpo"
- Binary format: `[CLS] post [SEP] criterion [SEP]` → single binary label per forward pass (9 forward passes per post for 9 criteria)
- Multi-label format: `[CLS] post [SEP] criterion1 [SEP] criterion2 [SEP] ...` → single forward pass with 9-dimensional output

**Implementation**:
- `BinaryPairDataset`: Explodes each post into 9 examples (post, criterion_i) with binary label
- `MultiLabelDataset`: One example per post with 9-dimensional label vector
- HPO selects format; classification head adjusts output dimension accordingly (1 vs 9)

**Tradeoffs**:
- Binary pairs: More training examples (9x), slower training, but cleaner separation per criterion
- Multi-label: Faster training, shared context across criteria, but assumes independence

**Alternatives Considered**:
- Fixed format: Limits exploration of task formulation impact on performance

---

## 4. Loss Function Design

### Decision: Modular Loss Library with Hybrid Support

**Loss Functions Implemented**:

1. **Binary Cross-Entropy (BCE)**: Standard sigmoid + BCE loss
2. **Weighted BCE**: Class-weighted BCE to handle imbalance
3. **Focal Loss**: Focuses on hard examples, reduces easy example gradient contribution
4. **Adaptive Focal Loss**: Learnable focusing parameter
5. **Hybrid Loss**: Weighted combination of one BCE variant + one Focal variant
   - Formula: `loss = α * bce_variant + (1-α) * focal_variant`
   - α is HPO-tunable weight

**Rationale**:
- Mental health data likely has class imbalance (not all posts match all criteria)
- Focal loss helps with hard negatives (posts that superficially look like matches)
- Hybrid allows balancing stability (BCE) with hard-example focus (Focal)

**Implementation Notes**:
- Use `torch.nn.BCEWithLogitsLoss` for numerical stability
- Focal loss: `FL = -α(1-p_t)^γ log(p_t)` where γ is HPO-tunable focusing parameter
- For evidence binding: Support both token-level cross-entropy (span endpoints) and sequence tagging loss (CRF)

---

## 5. Data Augmentation Strategy

### Decision: TextAttack with HPO-Controlled Method Selection

**Rationale**:
- User requirement: "use textattach library to perform augmentation"
- Augmentation only on evidence sentences (from `redsm5_annotations.csv` sentence column), not full posts
- HPO searches: which methods, how many methods (0 to all)

**TextAttack Augmentation Methods**:
- Word substitution (synonym replacement via WordNet, embedding-based)
- Word insertion
- Word swap
- Sentence paraphrasing (back-translation, T5-based)
- Character-level perturbations (typos)

**Implementation**:
- Parse `redsm5_annotations.csv` to extract evidence sentences per post-criterion pair
- Apply selected augmentation(s) to evidence sentences
- Reconstruct post with augmented evidence, keep non-evidence text unchanged
- HPO parameters:
  - `augmentation_methods`: Subset of available methods (empty set = no augmentation)
  - `num_methods`: Integer 0 to len(all_methods)
  - `augmentation_prob`: Per-example probability of applying augmentation

**Alternatives Considered**:
- Augment full posts: Rejected, violates requirement to preserve non-evidence text
- Fixed augmentation: Doesn't explore impact on different model architectures

---

## 6. Regularization Techniques

### Decision: Comprehensive Regularization Suite with HPO

**Techniques Implemented**:

1. **Dropout**:
   - In classification heads: 1 dropout before linear (1-layer), 2 dropouts before each linear (MLP)
   - Dropout rate: HPO-tunable (e.g., 0.1-0.5)

2. **Label Smoothing**:
   - Softens hard labels: `y_smooth = (1-ε)y + ε/K`
   - HPO-tunable ε (e.g., 0.0-0.2)

3. **Layer-Wise Learning Rate Decay**:
   - Lower layers get smaller LR (reduce catastrophic forgetting of pretrained weights)
   - Decay factor: HPO-tunable (e.g., 0.8-0.99)

4. **Differential Learning Rates**:
   - Encoder vs head different LRs
   - Ratio: HPO-tunable

5. **Warmup Ratio**:
   - Linear warmup for first N% of training
   - Ratio: HPO-tunable (e.g., 0.0-0.2)

6. **Adversarial Training**:
   - Add small adversarial perturbations to embeddings (e.g., FGM, PGD)
   - Perturbation magnitude: HPO-tunable

7. **Class Imbalance Handling**:
   - Compute class weights from training set
   - Apply to weighted BCE

**Rationale**:
- Large pretrained models prone to overfitting on small datasets
- Mental health data likely has class imbalance requiring explicit weighting
- Layer-wise LR decay preserves pretrained knowledge while adapting to task

---

## 7. Activation Functions

### Decision: HPO-Searchable Activations Across Model Components

**Supported Activations**:
- GELU (default for modern transformers)
- SiLU (Swish)
- Swish
- ReLU
- LeakyReLU
- Mish
- Tanh

**Application**:
- In MLP classification heads between linear layers
- In coupling modules

**Rationale**:
- Different tasks/datasets benefit from different non-linearities
- GELU common in transformers, but task-specific heads may benefit from alternatives

---

## 8. Checkpoint Management & Storage Optimization

### Decision: Epoch-Based Retention with Proactive Pruning

**Retention Policy**:
- Keep last N checkpoints (for resume)
- Keep best K checkpoints (by optimization metric)
- Keep all co-best checkpoints (tied validation scores)
- Minimum interval: 1 epoch (from FR-022)
- Max total size: User-configurable
- Disk space threshold: 10% available capacity triggers pruning (FR-018)

**Pruning Strategy**:
1. After each checkpoint save, check:
   - Total checkpoint directory size vs `max_total_size`
   - Available disk space vs 10% threshold
2. If violated, prune oldest non-best checkpoints first
3. Preserve: last N, best K, all co-best
4. Atomic checkpoint writes (temp file → rename) to avoid corruption (edge case from spec)

**Metadata Tracking**:
- Each checkpoint stores:
  - Trial ID, epoch, step
  - Metrics snapshot (validation scores)
  - Timestamp
  - Retained flag, co_best flag

**Rationale**:
- Sequential trials + pruning per trial prevents 1000-trial workload from exhausting storage
- Epoch-based checkpointing balances resume granularity vs I/O overhead
- 10% disk threshold provides safety margin before failure

**Implementation**:
- `CheckpointManager` class tracks all checkpoints in SQLite DB
- Async background thread monitors disk space (logs WARNING at 10%, triggers pruning)
- Pruning respects retention policy invariants (never delete last or best)

---

## 9. MLflow Integration & Metrics Buffering

### Decision: Local MLflow with Disk-Based Buffering

**Setup**:
- MLflow tracking server: Local SQLite database in `experiments/mlflow_db/`
- No network authentication required (A-003)

**Metrics Buffering**:
- If MLflow tracking backend temporarily unreachable (file lock, corruption):
  - Buffer metrics to disk (JSON log file per trial)
  - Retry periodically (exponential backoff)
  - Emit WARNING when buffer exceeds 100MB (FR-017)
  - No hard limit (training continues)
- On tracking backend restore, flush buffered metrics

**Rationale**:
- Local SQLite can have file lock contention or corruption during crashes
- Buffering ensures no metrics lost even if tracking backend unavailable
- 100MB threshold ~1M metric entries (unlikely to exceed during single trial)

**Implementation**:
- `MetricsBuffer` class with disk-backed queue
- Periodic background flush task
- MLflow custom callback to route through buffer

---

## 10. Docker Container Environment

### Decision: PyTorch + Hugging Face CUDA Image with Mounted Data

**Base Image**: `pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime`

**Installed Dependencies** (pinned in `requirements.txt`):
- transformers==4.35.2
- datasets==2.15.0
- mlflow==2.8.1
- optuna==3.4.0
- textattack==0.3.8
- torchcrf==1.1.0
- pyyaml, pandas, scikit-learn

**Container Setup**:
- Mount `Data/` directory as read-only volume
- Mount `experiments/` directory as read-write volume (for MLflow DB and trial outputs)
- Expose MLflow UI on port 5000 (optional)
- Environment variables:
  - `HF_HOME=/workspace/.cache/huggingface` (Hugging Face cache)
  - `CUDA_VISIBLE_DEVICES` (GPU selection)

**Reproducibility**:
- Pin all package versions
- Set deterministic CUDA flags (`torch.backends.cudnn.deterministic=True`)
- Document base image digest in Dockerfile

**Rationale**:
- Portable across machines (A-004)
- Isolates dependencies from host system
- CUDA image supports GPU acceleration
- Mounting data prevents duplication in container layers

---

## 11. Model Architecture Catalog

### Decision: 30 Pretrained Transformers from Hugging Face

**Model Categories**:

1. **General NLP (BERT family)**:
   - google-bert/bert-base-uncased
   - google-bert/bert-large-uncased
   - google-bert/bert-large-uncased-whole-word-masking-finetuned-squad

2. **DeBERTa**:
   - nvidia/quality-classifier-deberta
   - microsoft/deberta-v3-base
   - microsoft/deberta-v3-large

3. **SpanBERT** (optimized for span tasks):
   - SpanBERT/spanbert-base-cased
   - SpanBERT/spanbert-large-cased

4. **XLM-RoBERTa** (multilingual, though data is English):
   - FacebookAI/xlm-roberta-base
   - FacebookAI/roberta-large
   - FacebookAI/xlm-roberta-large
   - FacebookAI/xlm-roberta-large-finetuned-conll03-english

5. **ELECTRA**:
   - google/electra-base-discriminator
   - OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-109M
   - OpenMed/OpenMed-NER-ChemicalDetect-ElectraMed-33M

6. **Longformer** (long documents):
   - allenai/longformer-base-4096
   - allenai/longformer-base-4096-extra.pos.embd.only
   - allenai/longformer-large-4096-extra.pos.embd.only
   - allenai/longformer-large-4096-finetuned-triviaqa
   - allenai/longformer-large-4096
   - allenai/longformer-scico

7. **BigBird** (long documents):
   - google/bigbird-roberta-base
   - google/bigbird-roberta-large

8. **BioBERT** (biomedical domain):
   - dmis-lab/biobert-large-cased-v1.1
   - dmis-lab/biobert-v1.1

9. **ClinicalBERT** (clinical domain):
   - medicalai/ClinicalBERT

10. **Mental Health Domain**:
    - mental/mental-bert-base-uncased
    - mnaylor/psychbert-cased
    - mnaylor/psychbert-finetuned-mentalhealth
    - mnaylor/psychbert-finetuned-multiclass

**Rationale**:
- **Domain coverage**: Mental health (PsychBERT, MentalBERT), clinical (ClinicalBERT), biomedical (BioBERT), general NLP
- **Architecture diversity**: Standard BERT, DeBERTa (disentangled attention), SpanBERT (span tasks), Longformer/BigBird (long contexts)
- **Scale range**: Base (110M params) to Large (340M params) fits 1-10GB memory constraint
- **Hypothesis**: Mental health domain models may capture relevant semantic patterns

**Implementation**:
- Load via `transformers.AutoModel.from_pretrained(model_id)`
- Cache models locally to avoid repeated downloads (Hugging Face cache)
- HPO selects one model ID per trial

---

## 12. Data Processing Pipeline

### Decision: Dynamic Dataset Generation from RedSM5 CSV Files

**Input Files** (read-only):
- `Data/redsm5/redsm5_posts.csv`: Post text and metadata
- `Data/redsm5/redsm5_annotations.csv`: Criteria labels and evidence sentence annotations

**Processing Steps**:

1. **Load & Merge**:
   - Read both CSVs with pandas
   - Join on post_id to align labels with posts

2. **Format Selection** (HPO-driven):
   - If binary pairs: Create 9 examples per post (one per criterion)
   - If multi-label: Create 1 example per post with 9-dim label

3. **Evidence Extraction**:
   - Parse `sentence` column in annotations CSV for evidence spans
   - Map evidence to character offsets in post text

4. **Augmentation** (HPO-driven):
   - Apply selected TextAttack methods to evidence sentences
   - Replace evidence in post with augmented version

5. **Train/Val/Test Split**:
   - 70/15/15 split at post level (not example level to avoid leakage)
   - Save processed splits to `Data/redsm5/processed/` for caching
   - Use same splits across trials for fair comparison

6. **Tokenization**:
   - Model-specific tokenizer from Hugging Face
   - Handle [CLS], [SEP] tokens per format
   - Truncate to max model length (512 for BERT, 4096 for Longformer)

**Output Format**:
- PyTorch `Dataset` yielding:
  - `input_ids`, `attention_mask`, `token_type_ids`
  - `criteria_labels`: Binary (1,) or multi-label (9,)
  - `evidence_spans`: Start/end indices or BIO tags
  - `post_id`, `criterion_id` (for tracking)

**Rationale**:
- Single split ensures all trials compare on same data
- Augmentation on evidence only preserves post context (user requirement)
- Caching avoids re-processing CSVs every trial

---

## 13. Evaluation & Reporting

### Decision: Per-Trial JSON Reports with Test Set Metrics

**Test Set Evaluation**:
- After each trial, load best checkpoint(s)
- Evaluate on held-out test set (never seen during training/validation)
- Compute metrics:
  - **Criteria matching**: Accuracy, F1 (macro/micro), Precision, Recall per criterion
  - **Evidence binding**: Exact match, F1 (span overlap), Character-level F1
- If multiple co-best checkpoints, evaluate all and include all results

**JSON Report Schema** (per trial):
```json
{
  "trial_id": "uuid",
  "timestamp": "2025-10-10T12:34:56Z",
  "config": {
    "model_id": "mental/mental-bert-base-uncased",
    "input_format": "binary_pairs",
    "criteria_head": "mlp_residual",
    "evidence_head": "biaffine",
    "loss_function": "hybrid_bce_focal",
    "augmentation_methods": ["synonym", "back_translation"],
    ...
  },
  "optimization_metric": "val_f1_macro",
  "checkpoints": [
    {
      "path": "experiments/trial_123/checkpoints/epoch_10.pt",
      "epoch": 10,
      "validation_metric": 0.856,
      "co_best": false
    }
  ],
  "test_metrics": {
    "criteria_matching": {
      "accuracy": 0.842,
      "f1_macro": 0.834,
      "f1_micro": 0.851,
      "per_criterion": {...}
    },
    "evidence_binding": {
      "exact_match": 0.412,
      "f1": 0.678
    }
  }
}
```

**Rationale**:
- JSON format enables automated analysis across 1000 trials
- Full config snapshot ensures reproducibility
- Co-best checkpoint tracking supports tie scenarios
- Test metrics provide unbiased performance estimate

---

## 14. HPO Search Space Summary

**Total Search Space Dimensions** (Categorical + Continuous):

1. **Model Architecture** (30 choices)
2. **Input Format** (2 choices: binary pairs, multi-label)
3. **Criteria Matching Head** (5 architectures × 4 pooling strategies = 20 combinations)
4. **Criteria Head Hidden Dim** (continuous: 128-1024)
5. **Evidence Binding Head** (5 architectures)
6. **Task Coupling** (2 choices: independent, coupled)
   - If coupled: 5 combination methods × 3 pooling strategies = 15 sub-choices
7. **Loss Function** (5 base + hybrid combinations ≈ 10 choices)
   - If hybrid: weight α (continuous: 0.1-0.9)
   - If focal: γ focusing parameter (continuous: 1.0-5.0)
8. **Augmentation Methods** (2^5 = 32 subsets of 5 methods)
   - Augmentation probability (continuous: 0.0-0.5)
9. **Activation Function** (7 choices)
10. **Dropout Rate** (continuous: 0.1-0.5)
11. **Label Smoothing** (continuous: 0.0-0.2)
12. **Learning Rate** (continuous: 1e-6 to 1e-4, log scale)
13. **Layer-Wise LR Decay** (continuous: 0.8-0.99)
14. **Differential LR Ratio** (continuous: 0.1-1.0, encoder LR / head LR)
15. **Warmup Ratio** (continuous: 0.0-0.2)
16. **Adversarial Perturbation** (continuous: 0.0-0.01)
17. **Batch Size** (categorical: 8, 16, 32)
18. **Epochs** (categorical: 10, 20, 30)

**Estimated Search Space Size**: O(10^15) combinations (combinatorial explosion)

**Optimization Strategy**:
- Optuna TPE (Tree-structured Parzen Estimator) sampler to focus on promising regions
- Multi-objective optimization: Maximize validation F1, minimize checkpoint disk usage
- Early stopping: Prune trials with validation F1 < 0.5 after 3 epochs

---

## 15. Logging & Observability

### Decision: Dual Logging (Structured JSON + Human-Readable)

**Logging Levels**:
- DEBUG: Detailed model internals, batch-level metrics
- INFO: Epoch summaries, checkpoint saves, disk space checks
- WARNING: Buffer size exceeds 100MB, approaching disk threshold
- ERROR: Training failures, checkpoint corruptions
- CRITICAL: Irrecoverable errors (out of disk space after pruning)

**Log Outputs**:
1. **Structured JSON Log** (`experiments/trial_<id>/logs/training.jsonl`):
   - One JSON object per line
   - Fields: `timestamp`, `level`, `message`, `trial_id`, `epoch`, `step`, `component`
   - Machine-parseable for automated analysis

2. **Human-Readable Stdout**:
   - Colorized (if TTY) progress bars (tqdm)
   - Epoch summaries with metrics tables
   - Critical errors immediately visible

**MLflow Tracking**:
- Log hyperparameters as `mlflow.log_params(config)`
- Log metrics as `mlflow.log_metrics({"val_f1": 0.8, "epoch": 5})`
- Log artifacts: Best checkpoint, JSON report, plots

**Rationale**:
- JSON logs support post-hoc analysis (e.g., which configs cause OOM)
- Stdout provides real-time monitoring
- MLflow centralizes experiment comparison

---

## 16. Testing Strategy

### Decision: Layered Testing (Unit → Integration → Contract)

**Unit Tests**:
- `test_checkpoint_manager.py`: Retention policy logic, pruning scenarios
- `test_augmentation.py`: Evidence-only augmentation correctness
- `test_heads.py`: Forward pass shapes, activation functions
- `test_losses.py`: Loss function gradients, numerical stability

**Integration Tests**:
- `test_full_trial.py`: End-to-end trial execution with mock dataset
- `test_checkpoint_resume.py`: Interrupt trial, resume from checkpoint, verify metric continuity

**Contract Tests**:
- `test_config_schema.py`: Validate HPO config against schema
- `test_output_formats.py`: Validate JSON report against schema

**Property-Based Tests** (hypothesis):
- Augmentation preserves non-evidence text
- Checkpoint pruning never deletes last or best
- Input format conversion is invertible

**Rationale**:
- Complex HPO pipeline requires high test coverage
- Property-based tests catch edge cases in retention policy
- Contract tests ensure JSON schema compatibility across trials

---

## 17. Resume & Fault Tolerance

### Decision: Checkpoint-Based Resume with Idempotent Trials

**Resume Mechanism**:
- Each trial has unique ID (UUID)
- Checkpoint metadata includes: epoch, step, optimizer state, RNG state
- On resume:
  1. Detect existing trial directory
  2. Load latest valid checkpoint
  3. Restore optimizer, LR scheduler, RNG states
  4. Skip already-logged metrics (MLflow deduplication by step)
  5. Continue from next epoch/step

**Fault Scenarios**:
1. **Crash during checkpoint save**: Atomic write (temp → rename) ensures partial files discarded
2. **MLflow backend unavailable**: Metrics buffered to disk, flushed on restore
3. **Out of disk space**: Pruning triggered at 10% threshold; if still fails, trial aborted with ERROR log, but already-logged metrics preserved

**Idempotency**:
- Re-running same trial ID resumes, doesn't restart
- Useful for preemptible cloud instances

**Rationale**:
- Long-running 1000-trial HPO workloads will experience failures
- Resume capability prevents wasted compute
- Idempotency simplifies retry logic

---

## 18. Performance Optimizations

### Decision: Gradient Accumulation + Mixed Precision Training

**Techniques**:

1. **Gradient Accumulation**:
   - Accumulate gradients over N micro-batches before optimizer step
   - Simulates larger batch sizes without OOM
   - HPO parameter: accumulation_steps (1, 2, 4)

2. **Mixed Precision (FP16)**:
   - Use PyTorch AMP (`torch.cuda.amp`) for faster training
   - Automatic loss scaling to prevent underflow
   - Compatible with modern GPUs (V100, A100)

3. **Gradient Checkpointing**:
   - Trade compute for memory (recompute activations during backward pass)
   - Enables larger models (e.g., Longformer-large) on limited VRAM

4. **DataLoader Optimizations**:
   - Pin memory (`pin_memory=True`) for faster GPU transfer
   - Multi-worker data loading (`num_workers=4`)
   - Prefetch batches

**Rationale**:
- 1-10GB models strain GPU memory
- Mixed precision cuts memory usage ~50%, training time ~30%
- Gradient checkpointing allows large models without distributed training

---

## Summary of Key Design Decisions

| Aspect | Decision | Key Rationale |
|--------|----------|---------------|
| HPO Framework | Optuna | Sequential execution, conditional search spaces, MLflow integration |
| Multi-Task Architecture | Flexible coupling, HPO-searchable | Explore independent vs evidence-guided matching |
| Input Formatting | Binary pairs vs multi-label, HPO-searchable | Task formulation impacts performance |
| Loss Functions | 5 base + hybrid combinations | Handle class imbalance, focus on hard examples |
| Data Augmentation | TextAttack on evidence only | Preserve post context, explore augmentation impact |
| Regularization | 7 techniques, all HPO-tunable | Prevent overfitting on small dataset |
| Checkpoint Management | Epoch-based retention with 10% disk threshold | Balance resume granularity vs storage efficiency |
| Logging | Dual JSON + stdout | Machine-parseable + human-readable |
| Containerization | PyTorch CUDA Docker image | Portability, reproducibility, GPU support |
| Model Catalog | 30 transformers, domain-specific included | Cover general, clinical, mental health domains |

**Next Steps**: Proceed to Phase 1 (Design & Contracts) to define data models and API contracts.
