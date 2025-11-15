# Implementation Plan: BERT Training Pipeline for Evidence Sentence Classification

**Branch**: `002-bert-training-pipeline` | **Date**: 2025-11-12 | **Spec**: spec.md
**Input**: Feature specification from `/specs/002-bert-training-pipeline/spec.md`

## Summary

Train and evaluate a binary classifier using configurable BERT-family models from Hugging Face
with a sequence classification head on NSP-style criterion–sentence pairs
(`[CLS] <criterion> [SEP] <sentence> [SEP]`). Use post-level 70/15/15 train/val/test splits
with stratification to prevent data leakage. Loss function combines weighted focal loss with
class frequency-based weights. Implement full PyTorch optimization suite (mixed precision, TF32,
SDPA/FlashAttention, torch.compile, gradient checkpointing, fused AdamW). Parameters managed
via Hydra; experiments tracked in MLflow (`sqlite:///mlflow.db`, `./mlruns`). Optional Optuna
HPO for hyperparameter tuning.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: PyTorch >=2.2, Transformers >=4.40, Pandas, scikit-learn, Hydra,
MLflow >=2.8, Optuna >=3.4
**Storage**: Local files for data; MLflow SQLite DB `mlflow.db`, artifacts under `mlruns/`;
Optuna SQLite `optuna.db`
**Testing**: pytest (unit tests for data parsing, metric computation, splitting, focal loss)
**Target Platform**: Linux/macOS dev; GPU strongly recommended for full optimizations
**Project Type**: single (library + CLI)
**Performance Goals**: <15 min/epoch with optimizations on GPU; 2x speedup over baseline;
30% memory reduction with mixed precision
**Constraints**: Enforce reproducibility (seed, deterministic backends); honor NSP input format;
only use annotated sentence-symptom pairs (no synthetic negatives)
**Scale/Scope**: 1,484 posts, ~1,547 annotations; batch size tuned to memory

## Constitution Check

Gates (all PASS):
- P1 BERT-based classifier (HF): using configurable BERT-family models ✓
- P2 NSP input format: criterion–sentence pairs from annotations ✓
- P3 Hydra config: configs/ with full override support ✓
- P4 MLflow local: sqlite:///mlflow.db + ./mlruns ✓
- P5 Optuna if HPO: optional Optuna HPO with sqlite:///optuna.db ✓
- P6 Reproducibility: seeds, env snapshot, Hydra config logging, Git SHA ✓

## Project Structure

### Documentation (this feature)

```text
specs/002-bert-training-pipeline/
├── spec.md             # Feature specification with clarifications
├── plan.md             # This implementation plan
├── tasks.md            # Task breakdown (to be generated)
├── research.md         # Technical research notes (to be created)
├── data-model.md       # Entity definitions (to be created)
└── quickstart.md       # End-to-end usage guide (to be created)
```

### Source Code (repository root)

```text
src/Project/SubProject/
├── data/
│   └── dataset.py           # ReDSM5Dataset, data loading, splitting
├── models/
│   └── model.py             # BERTBinaryClassifier (refactor existing)
├── engine/
│   ├── train_engine.py      # Training loop with optimizations
│   └── eval_engine.py       # Evaluation and metrics
├── losses/
│   └── focal_loss.py        # Weighted focal loss implementation
└── utils/
    ├── log.py               # Existing logger
    ├── seed.py              # Existing seed utilities
    ├── mlflow_utils.py      # Existing MLflow helpers
    └── optimization.py      # PyTorch optimization utilities

configs/
├── config.yaml              # Main Hydra config
├── data/
│   └── redsm5.yaml          # Dataset configuration
├── model/
│   ├── bert_base.yaml       # bert-base-uncased config
│   ├── bert_large.yaml      # bert-large-uncased config
│   └── deberta_v3.yaml      # microsoft/deberta-v3-base config
├── training/
│   ├── default.yaml         # Default training params
│   └── optimized.yaml       # Full optimization suite
└── optuna/
    └── default.yaml         # HPO search space

scripts/
├── train.py                 # Main training script with Hydra
├── eval.py                  # Evaluation script
└── hpo.py                   # Optional Optuna HPO script

tests/
├── unit/
│   ├── test_dataset.py
│   ├── test_focal_loss.py
│   ├── test_splitting.py
│   └── test_optimization.py
└── integration/
    └── test_training_pipeline.py
```

**Structure Decision**: Extend existing `src/Project/SubProject/` layout; add `data/dataset.py`,
`losses/`, refactor `models/model.py`, implement `engine/`. New Hydra configs under `configs/`.
Training scripts in `scripts/`. Tests mirror under `tests/`.

## Phase 0: Research (resolve unknowns)

Topics and decisions to document in research.md:

1. **Weighted Focal Loss Implementation**
   - Combine focal loss (γ=2.0 default) with class frequency weights (α)
   - Formula: `FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)`
   - α computed as inverse class frequencies, normalized
   - Configurable γ (gamma) and α (alpha) via Hydra
   - Compatible with PyTorch autocast for mixed precision

2. **Post-Level Stratified Splitting**
   - Split 1,484 posts into 70/15/15 (train/val/test)
   - Use stratification on label distribution to maintain balance
   - All sentences from same post stay together (prevent leakage)
   - Implementation: group by post_id, then stratified split on aggregated labels

3. **PyTorch Optimizations**
   - Mixed precision: detect bf16 support, fallback to fp16 with GradScaler
   - TF32: `torch.backends.cuda.matmul.allow_tf32 = True`
   - Attention: prefer FlashAttention2 if installed, else SDPA (built-in)
   - Gradient checkpointing: `model.gradient_checkpointing_enable()`
   - Fused AdamW: `torch.optim.AdamW(..., fused=True)` with fallback
   - torch.compile: optional, configurable, disable for debugging
   - DataLoader: `pin_memory=True`, `num_workers=4`, `persistent_workers=True`
   - Gradient accumulation: configurable for larger effective batch size

4. **Criterion-Sentence Pairing Strategy**
   - Only use annotated pairs from `redsm5_annotations.csv`
   - Each row: (post_id, sentence_id, sentence_text, DSM5_symptom, status, explanation)
   - Pair sentence_text with corresponding DSM-5 criterion from `MDD_Criteira.json`
   - No synthetic negatives - use status=0 annotations as negatives
   - Total pairs = number of annotation rows (~1,547)

5. **Metrics and Logging**
   - Per-epoch: loss, accuracy, precision, recall, F1 (binary)
   - Per-symptom metrics during evaluation (9 symptoms + SPECIAL_CASE)
   - Log to MLflow: all metrics, parameters, Hydra config, model artifacts
   - Save best model based on validation F1

## Phase 1: Design & Contracts

Artifacts to generate:

### data-model.md
Define entities:
- **CriterionSentencePair**: (criterion_text, sentence_text, label, metadata)
- **ReDSM5Dataset**: PyTorch Dataset with tokenization and collation
- **DataSplit**: Train/val/test manifests with post_ids
- **TrainingRun**: MLflow run with params, metrics, artifacts
- **HydraConfig**: Complete config structure

### quickstart.md
End-to-end commands:
```bash
# Train with default config
python scripts/train.py

# Train with model override
python scripts/train.py model=bert_large

# Train with optimizations disabled (debugging)
python scripts/train.py training=default training.use_compile=false

# Run HPO
python scripts/hpo.py

# Evaluate saved model
python scripts/eval.py model_uri=runs:/RUN_ID/model

# MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### contracts/
N/A - CLI-based, no external API contracts needed

Agent context updated via `.specify/scripts/bash/update-agent-context.sh codex`.

## Phase 2: Implementation Strategy (detailed)

### 2.1 Data Loading & Preprocessing

**File**: `src/Project/SubProject/data/dataset.py`

**Tasks**:
1. Load ReDSM5 posts from `data/redsm5/redsm5_posts.csv`
2. Load annotations from `data/redsm5/redsm5_annotations.csv`
3. Load DSM-5 criteria from `data/data/DSM5/MDD_Criteira.json`
4. Map DSM5_symptom names to criterion texts:
   - DEPRESSED_MOOD → A.1 criterion
   - ANHEDONIA → A.2 criterion
   - etc.
5. Create CriterionSentencePair for each annotation row
6. Implement post-level stratified split (70/15/15):
   - Group annotations by post_id
   - Compute label distribution per post
   - Use stratified split on post level
   - Save split manifests (post_ids per split)
7. Implement ReDSM5Dataset(torch.utils.data.Dataset):
   - Load pairs for specified split (train/val/test)
   - Tokenize with NSP format: `[CLS] criterion [SEP] sentence [SEP]`
   - Return: input_ids, attention_mask, labels, metadata
8. Implement DataLoader collation function with padding
9. Compute class weights from training set for weighted focal loss

**Acceptance**:
- Dataset loads all 1,547 annotations correctly
- Split respects post-level grouping (no leakage)
- Splits maintain label balance (stratified)
- Tokenization produces correct NSP format
- Class weights computed and logged

### 2.2 Model Architecture

**File**: `src/Project/SubProject/models/model.py`

**Tasks**:
1. Refactor existing Model class
2. Implement BERTBinaryClassifier(torch.nn.Module):
   - Load BERT-family model from HF (configurable)
   - AutoModelForSequenceClassification with num_labels=2
   - Configure attention implementation (sdpa or flash_attention_2)
   - Disable key-value cache for training (`use_cache=False`)
   - Support gradient checkpointing
3. Add model configuration validation
4. Add device placement helper
5. Implement model loading from MLflow

**Acceptance**:
- Model loads different BERT variants via config
- Classification head outputs (batch_size, 2) logits
- Gradient checkpointing works without errors
- Model compatible with mixed precision training
- Can save/load from MLflow

### 2.3 Loss Function

**File**: `src/Project/SubProject/losses/focal_loss.py`

**Tasks**:
1. Implement WeightedFocalLoss(torch.nn.Module):
   - Takes alpha (class weights), gamma (focal parameter)
   - Combines both weighted and focal loss
   - Formula: `FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)`
   - Support for multi-class (though we use binary)
   - Compatible with autocast (no float64 ops)
2. Add unit tests for loss computation
3. Add gradient flow tests
4. Document hyperparameters (default γ=2.0)

**Acceptance**:
- Loss computes correctly on simple examples
- Gradients flow properly
- Works with mixed precision (bf16/fp16)
- Configurable via Hydra

### 2.4 Optimization Utilities

**File**: `src/Project/SubProject/utils/optimization.py`

**Tasks**:
1. Implement `setup_precision()`:
   - Detect bf16 support: `torch.cuda.is_bf16_supported()`
   - Set autocast dtype (bf16 > fp16 > fp32)
   - Create GradScaler if fp16
   - Enable TF32: `torch.backends.cuda.matmul.allow_tf32 = True`
2. Implement `setup_deterministic()`:
   - Set seeds via existing seed.py
   - Configure cudnn: `deterministic=True`, `benchmark=False`
3. Implement `setup_optimizer()`:
   - Try fused AdamW: `torch.optim.AdamW(..., fused=True)`
   - Fallback to regular AdamW if fused not available
   - Log which optimizer is used
4. Implement `compile_model()`:
   - Wrap model with torch.compile if enabled
   - Configurable mode (default, reduce-overhead, max-autotune)
   - Handle compilation failures gracefully
5. Document optimization trade-offs in comments

**Acceptance**:
- Precision setup detects hardware capabilities correctly
- Optimizer selection works on different platforms
- Deterministic mode reduces variance
- torch.compile improves speed (measured)

### 2.5 Training Engine

**File**: `src/Project/SubProject/engine/train_engine.py`

**Tasks**:
1. Implement `train_one_epoch()`:
   - Iterate over DataLoader with tqdm progress
   - Forward pass with autocast
   - Compute weighted focal loss
   - Backward pass (handle GradScaler for fp16)
   - Gradient clipping (max_norm=1.0)
   - Optimizer step with fused operations
   - Accumulate metrics (loss, accuracy)
   - Optional gradient accumulation
2. Implement `validate()`:
   - Evaluation mode (no grad)
   - Compute val loss and metrics
   - Return dict with all metrics
3. Implement `train()`:
   - Main training loop over epochs
   - Call train_one_epoch() and validate()
   - MLflow logging per epoch
   - Early stopping based on val F1
   - Save checkpoints
   - Log best model to MLflow with model registry
4. Implement efficient DataLoader setup:
   - pin_memory=True
   - num_workers configurable (default 4)
   - persistent_workers=True if workers > 0
5. Add training state recovery (resume from checkpoint)

**Acceptance**:
- Training completes without errors
- Mixed precision reduces memory by ~30%
- Optimizations provide 2x speedup
- Metrics logged to MLflow correctly
- Model saved and recoverable

### 2.6 Evaluation Engine

**File**: `src/Project/SubProject/engine/eval_engine.py`

**Tasks**:
1. Implement `evaluate_model()`:
   - Load model from MLflow
   - Run inference on test set
   - Compute overall metrics (accuracy, P, R, F1)
   - Compute per-symptom metrics
   - Generate confusion matrix
   - Log all metrics to MLflow
2. Implement `compute_metrics()`:
   - Binary classification metrics via sklearn
   - Per-class precision, recall, F1
   - ROC-AUC, PR-AUC (optional)
3. Support batch inference for efficiency

**Acceptance**:
- Evaluation produces all required metrics
- Per-symptom metrics computed correctly
- Results logged to MLflow
- Evaluation script works standalone

### 2.7 Hydra Configuration

**Files**: `configs/*.yaml`

**Tasks**:
1. Create main config.yaml with defaults groups
2. Create data/redsm5.yaml:
   - Dataset paths
   - Split ratios (70/15/15)
   - Tokenizer settings (max_length=512, truncation=True)
3. Create model configs (bert_base, bert_large, deberta_v3):
   - Model name/path
   - Attention implementation
   - Gradient checkpointing flag
4. Create training/default.yaml:
   - Epochs, batch size, learning rate
   - Optimizer settings (fused=True)
   - LR scheduler (linear with warmup)
   - Loss params (gamma=2.0, auto class weights)
5. Create training/optimized.yaml (extends default):
   - Enable all optimizations
   - use_compile=True
   - use_gradient_checkpointing=True
   - Mixed precision settings
6. Create optuna/default.yaml:
   - Search space for learning rate, batch size, gamma
   - Number of trials
   - Pruning strategy

**Acceptance**:
- Config structure is clean and modular
- Overrides work via CLI
- All parameters accessible in code
- Config logged to MLflow

### 2.8 Training Script

**File**: `scripts/train.py`

**Tasks**:
1. Setup Hydra decorator
2. Load and validate config
3. Setup MLflow (tracking URI, experiment name)
4. Setup logging and seeds
5. Setup optimizations (precision, deterministic, compile)
6. Load data and create DataLoaders
7. Initialize model and optimizer
8. Create weighted focal loss
9. Start MLflow run:
   - Log all parameters from config
   - Log Git commit SHA (if available)
   - Log environment info (pip freeze)
10. Call training engine
11. Log final model to registry
12. Print summary and MLflow run URI

**Acceptance**:
- Script runs end-to-end successfully
- All config overrides work
- MLflow run contains all artifacts
- Reproducible with same seed

### 2.9 Evaluation Script

**File**: `scripts/eval.py`

**Tasks**:
1. Accept model URI as argument (MLflow runs URI or path)
2. Load model and config from MLflow
3. Load test data
4. Run evaluation
5. Print metrics table
6. Optionally log to MLflow (new eval run)

**Acceptance**:
- Can load and evaluate any saved model
- Metrics match training time validation
- Clear output format

### 2.10 HPO Script (Optional)

**File**: `scripts/hpo.py`

**Tasks**:
1. Setup Optuna study with SQLite storage
2. Define objective function:
   - Sample hyperparameters from search space
   - Run training with sampled params
   - Return validation F1 as objective
3. Integrate with MLflow (log each trial)
4. Run optimization for N trials
5. Print best params and metric
6. Train final model with best params

**Acceptance**:
- Optuna study completes without errors
- Best params identified and logged
- Compatible with MLflow tracking

### 2.11 Testing

**Files**: `tests/unit/*.py`, `tests/integration/*.py`

**Tasks**:
1. Unit test: data loading and splitting
2. Unit test: tokenization and collation
3. Unit test: weighted focal loss computation
4. Unit test: class weight calculation
5. Unit test: optimization setup (mock GPU)
6. Integration test: full training on tiny subset (1 epoch, 10 samples)
7. Integration test: evaluation pipeline
8. Add pytest fixtures for sample data

**Acceptance**:
- All unit tests pass
- Integration test completes quickly
- Tests run in CI (if available)

### 2.12 Documentation

**Files**: `specs/002-bert-training-pipeline/*.md`

**Tasks**:
1. Create research.md with technical decisions
2. Create data-model.md with entity definitions
3. Create quickstart.md with usage examples
4. Update README.md with feature info (optional)

**Acceptance**:
- Documentation is clear and complete
- Quickstart commands work as written
- Research notes capture all key decisions

## Phase 3: Validation & Quality Gates

### 3.1 Functional Validation

- [ ] Dataset loads 1,547 annotations correctly
- [ ] Post-level split prevents leakage (manual verification)
- [ ] Model trains on criterion-sentence pairs
- [ ] Weighted focal loss computes without errors
- [ ] Mixed precision works (bf16 or fp16)
- [ ] All optimizations apply correctly
- [ ] MLflow tracks all parameters and metrics
- [ ] Evaluation produces per-symptom metrics
- [ ] Config overrides work via Hydra CLI

### 3.2 Performance Validation

- [ ] Training completes <15 min/epoch on GPU (baseline BERT)
- [ ] Optimizations provide ≥2x speedup vs baseline
- [ ] Mixed precision reduces memory by ≥30%
- [ ] Model achieves F1 > 0.6 on validation (baseline)
- [ ] Target F1 > 0.75 with tuning

### 3.3 Reproducibility Validation

- [ ] Same seed produces identical results (within FP precision)
- [ ] Git SHA logged to MLflow
- [ ] Environment snapshot logged (pip freeze)
- [ ] Full Hydra config logged
- [ ] Model can be reloaded from MLflow

### 3.4 Code Quality Gates

- [ ] Black formatting (line length 100): `black src tests scripts`
- [ ] Ruff linting: `ruff check src tests scripts`
- [ ] MyPy type checking: `mypy src tests scripts`
- [ ] Pytest tests pass: `pytest tests/`
- [ ] No TODO markers in merged code
- [ ] All functions have docstrings with type hints

## Phase 4: Deployment & Handoff

### 4.1 Deployment Checklist

- [ ] All code merged to feature branch
- [ ] All tests passing
- [ ] Documentation complete
- [ ] MLflow contains at least one trained model
- [ ] Quickstart guide validated by fresh user

### 4.2 Handoff Artifacts

1. **Code**: Complete implementation in `src/`, `scripts/`, `configs/`
2. **Tests**: Full test suite in `tests/`
3. **Docs**: All markdown files in `specs/002-bert-training-pipeline/`
4. **Models**: At least one trained model in MLflow registry
5. **Config Examples**: Working Hydra configs for common scenarios

### 4.3 Known Limitations

1. **Single-GPU only**: Multi-GPU training not implemented in this phase
2. **No distributed training**: DDP not configured
3. **Limited HPO**: Basic Optuna setup, advanced strategies (pruning, multi-objective) not implemented
4. **No model serving**: Inference API not provided
5. **No continuous training**: Pipeline assumes one-time training

### 4.4 Future Enhancements (Out of Scope)

- Multi-GPU training with DDP
- Advanced HPO with Optuna pruners and multi-objective optimization
- Model serving API (FastAPI/Flask)
- Continuous training pipeline
- Automated model monitoring
- Advanced data augmentation
- Ensemble methods

## Complexity Tracking

N/A — All clarifications align with constitution. No violations or workarounds needed.

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU OOM during training | High | Gradient checkpointing, smaller batch size, gradient accumulation |
| torch.compile failures | Medium | Make optional via config, provide clear error messages |
| FlashAttention not available | Low | Automatic fallback to SDPA (built-in) |
| Class imbalance too severe | Medium | Weighted focal loss addresses this; can tune gamma/alpha |
| Slow data loading | Medium | Pinned memory, multiple workers, persistent workers |
| Non-deterministic results | Medium | Deterministic mode enabled, seeds set, document known sources |

## Success Metrics (from spec)

- **SC-001**: <15 min/epoch with optimizations ✓
- **SC-002**: F1 > 0.6 baseline, target > 0.75 ✓
- **SC-003**: All runs logged to MLflow ✓
- **SC-004**: Reproducible with same seed ✓
- **SC-005**: Config overridable via Hydra ✓
- **SC-006**: Passes Black, Ruff, MyPy ✓
- **SC-007**: Handles all 1,484 posts ✓
- **SC-008**: Model in MLflow registry ✓
- **SC-009**: Per-symptom metrics ✓
- **SC-010**: CPU/GPU without code changes ✓
- **SC-011**: 2x speedup with optimizations ✓
- **SC-012**: 30% memory reduction ✓

---

**Plan Version**: 1.0
**Last Updated**: 2025-11-12
**Ready for Implementation**: Yes
