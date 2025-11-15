# Tasks: BERT Training Pipeline for Evidence Sentence Classification

**Input**: Design documents from `/specs/002-bert-training-pipeline/`
**Prerequisites**: plan.md ✓, spec.md ✓

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4, US5)
- Include exact file paths in descriptions

## Path Conventions

Single project layout: `src/`, `tests/`, `configs/`, `scripts/` at repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create directory structure per implementation plan (src/Project/SubProject/{data,losses,engine}, configs/, scripts/train.py, tests/{unit,integration}/)
- [ ] T002 [P] Add missing dependencies to pyproject.toml (scikit-learn, hydra-core)
- [ ] T003 [P] Configure pre-commit hooks for Black (line 100), Ruff, MyPy

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Research & Design Artifacts

- [ ] T004 [P] Create research.md documenting weighted focal loss formula, post-level splitting strategy, PyTorch optimizations, criterion-sentence pairing
- [ ] T005 [P] Create data-model.md defining CriterionSentencePair, ReDSM5Dataset, DataSplit, TrainingRun, HydraConfig entities
- [ ] T006 [P] Create quickstart.md with end-to-end usage commands (train, eval, HPO, MLflow UI)

### Core Utilities (Shared Across All Stories)

- [ ] T007 [P] Implement weighted focal loss in src/Project/SubProject/losses/focal_loss.py with WeightedFocalLoss class (α, γ params, autocast-compatible)
- [ ] T008 [P] Implement optimization utilities in src/Project/SubProject/utils/optimization.py (setup_precision, setup_deterministic, setup_optimizer, compile_model functions)
- [ ] T009 [P] Add unit tests for focal loss in tests/unit/test_focal_loss.py (correctness, gradient flow, mixed precision compatibility)
- [ ] T010 [P] Add unit tests for optimization setup in tests/unit/test_optimization.py (precision detection, optimizer fallback, deterministic mode)

### Configuration Framework

- [ ] T011 [P] Create main Hydra config in configs/config.yaml with defaults groups (data, model, training, optuna)
- [ ] T012 [P] Create data config in configs/data/redsm5.yaml (paths, split ratios, tokenizer settings, max_length=512)
- [ ] T013 [P] Create model configs: configs/model/{bert_base.yaml, bert_large.yaml, deberta_v3.yaml} (model names, attention impl, gradient checkpointing)
- [ ] T014 [P] Create training configs: configs/training/{default.yaml, optimized.yaml} (epochs, batch size, LR, loss params, optimizer, scheduler, optimizations)
- [ ] T015 [P] Create Optuna config in configs/optuna/default.yaml (search space for LR, batch size, gamma, num trials, pruning)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 5 - Load and Preprocess ReDSM5 Dataset (Priority: P1) 🎯 FIRST

**Goal**: Load ReDSM5 posts/annotations, pair with DSM-5 criteria, create NSP-style inputs, implement post-level stratified splits

**Independent Test**: Run dataset loader and verify outputs match NSP format `[CLS] <criterion> [SEP] <sentence> [SEP]` with correct labels

**Why First**: Data loading is a prerequisite for training (US1), evaluation (US2), and configuration testing (US3). Must be completed before any other stories.

### Implementation for User Story 5

- [ ] T016 [US5] Implement data loading functions in src/Project/SubProject/data/dataset.py:
  - load_redsm5_posts() from data/redsm5/redsm5_posts.csv
  - load_redsm5_annotations() from data/redsm5/redsm5_annotations.csv
  - load_dsm5_criteria() from data/data/DSM5/MDD_Criteira.json
- [ ] T017 [US5] Implement DSM5 symptom name to criterion text mapping (DEPRESSED_MOOD→A.1, ANHEDONIA→A.2, etc.) in src/Project/SubProject/data/dataset.py
- [ ] T018 [US5] Implement CriterionSentencePair creation from annotations (pair sentence_text with corresponding criterion, extract label from status field) in src/Project/SubProject/data/dataset.py
- [ ] T019 [US5] Implement post-level stratified split (70/15/15) in src/Project/SubProject/data/dataset.py:
  - Group annotations by post_id
  - Compute label distribution per post
  - Use stratified split on post level
  - Save split manifests (train_posts.json, val_posts.json, test_posts.json)
- [ ] T020 [US5] Implement class weight computation from training set (inverse frequency, normalized) in src/Project/SubProject/data/dataset.py
- [ ] T021 [US5] Implement ReDSM5Dataset(torch.utils.data.Dataset) class in src/Project/SubProject/data/dataset.py:
  - __init__(split, tokenizer, max_length, pairs)
  - __len__() returns number of pairs
  - __getitem__(idx) returns tokenized inputs in NSP format with labels and metadata
- [ ] T022 [US5] Implement collate_fn for DataLoader with dynamic padding in src/Project/SubProject/data/dataset.py
- [ ] T023 [P] [US5] Add unit tests for data loading in tests/unit/test_dataset.py (CSV parsing, criterion mapping, pair creation)
- [ ] T024 [P] [US5] Add unit tests for splitting in tests/unit/test_splitting.py (post-level grouping, no leakage, stratification, split ratios)
- [ ] T025 [P] [US5] Add unit tests for dataset in tests/unit/test_dataset.py (tokenization, NSP format, collation, class weights)

**Checkpoint**: Data pipeline complete - 1,547 pairs loaded, splits created, NSP format verified

---

## Phase 4: User Story 1 - Train Binary Classifier for Evidence Detection (Priority: P1) 🎯 MVP

**Goal**: Train BERT-based binary classifier on criterion-sentence pairs with weighted focal loss, mixed precision, and full optimizations

**Independent Test**: Run training script with small config, verify model saved to MLflow with logged metrics (loss, accuracy)

### Implementation for User Story 1

- [ ] T026 [US1] Refactor BERTBinaryClassifier in src/Project/SubProject/models/model.py:
  - Use AutoModelForSequenceClassification with num_labels=2
  - Configure attn_implementation (sdpa or flash_attention_2)
  - Set use_cache=False for training
  - Support gradient_checkpointing_enable()
- [ ] T027 [US1] Add model configuration validation and device placement helpers in src/Project/SubProject/models/model.py
- [ ] T028 [US1] Implement train_one_epoch() in src/Project/SubProject/engine/train_engine.py:
  - Iterate over DataLoader with tqdm
  - Forward pass with autocast
  - Compute weighted focal loss
  - Backward pass (handle GradScaler for fp16)
  - Gradient clipping (max_norm=1.0)
  - Optimizer step with fused operations
  - Optional gradient accumulation
  - Return epoch metrics (loss, accuracy)
- [ ] T029 [US1] Implement validate() in src/Project/SubProject/engine/train_engine.py:
  - Evaluation mode (no grad)
  - Compute validation loss and metrics
  - Return metrics dict
- [ ] T030 [US1] Implement train() main loop in src/Project/SubProject/engine/train_engine.py:
  - Loop over epochs
  - Call train_one_epoch() and validate()
  - MLflow logging per epoch
  - Early stopping based on validation F1
  - Save checkpoints
  - Log best model to MLflow with model registry
- [ ] T031 [US1] Implement efficient DataLoader setup in src/Project/SubProject/engine/train_engine.py (pin_memory=True, num_workers=4, persistent_workers=True)
- [ ] T032 [US1] Implement training script in scripts/train.py:
  - Hydra decorator for config
  - Setup MLflow (tracking URI, experiment name)
  - Setup logging and seeds
  - Setup optimizations (precision, deterministic, compile)
  - Load data and create DataLoaders
  - Initialize model and optimizer
  - Create weighted focal loss
  - Start MLflow run with full logging
  - Call train() engine
  - Log final model to registry
- [ ] T033 [P] [US1] Add integration test for training pipeline in tests/integration/test_training_pipeline.py (1 epoch, 10 samples, verify model saved)

**Checkpoint**: Training pipeline complete - can train models end-to-end with all optimizations and MLflow tracking

---

## Phase 5: User Story 2 - Evaluate Model Performance (Priority: P1)

**Goal**: Evaluate trained models on test data with standard metrics (P, R, F1) and per-symptom breakdown

**Independent Test**: Load trained model from MLflow, run evaluation on test set, verify metrics computed and logged

### Implementation for User Story 2

- [ ] T034 [US2] Implement compute_metrics() in src/Project/SubProject/engine/eval_engine.py:
  - Binary classification metrics via sklearn (accuracy, precision, recall, F1)
  - Per-class metrics
  - Optional ROC-AUC, PR-AUC
- [ ] T035 [US2] Implement per-symptom metrics computation in src/Project/SubProject/engine/eval_engine.py (group predictions by DSM5_symptom, compute metrics per symptom)
- [ ] T036 [US2] Implement evaluate_model() in src/Project/SubProject/engine/eval_engine.py:
  - Load model from MLflow
  - Run batch inference on test set
  - Compute overall metrics
  - Compute per-symptom metrics (9 symptoms + SPECIAL_CASE)
  - Generate confusion matrix
  - Log all metrics to MLflow
- [ ] T037 [US2] Implement evaluation script in scripts/eval.py:
  - Accept model URI as argument (MLflow runs URI or path)
  - Load model and config from MLflow
  - Load test data
  - Call evaluate_model()
  - Print metrics table
  - Optionally log to MLflow (new eval run)
- [ ] T038 [P] [US2] Add integration test for evaluation pipeline in tests/integration/test_training_pipeline.py (load model, eval on subset, verify metrics)

**Checkpoint**: Evaluation pipeline complete - can load and evaluate any saved model with per-symptom metrics

---

## Phase 6: User Story 3 - Configure Experiments via Hydra (Priority: P1)

**Goal**: Ensure all training parameters configurable via YAML or CLI overrides without code changes

**Independent Test**: Modify config file or pass CLI overrides, verify training run uses those parameters (visible in MLflow logs)

### Implementation for User Story 3

- [ ] T039 [US3] Validate Hydra config structure matches implementation (verify all params accessible in training script)
- [ ] T040 [US3] Add config validation in scripts/train.py (check required fields, validate ranges, log warnings)
- [ ] T041 [US3] Test CLI overrides in scripts/train.py (model=bert_large, training.batch_size=32, etc.) and verify MLflow logs
- [ ] T042 [US3] Document all config parameters in quickstart.md with examples
- [ ] T043 [P] [US3] Add unit tests for config loading in tests/unit/test_config.py (defaults, overrides, validation)

**Checkpoint**: Configuration system complete - all parameters accessible via Hydra with override support

---

## Phase 7: User Story 4 - Hyperparameter Optimization with Optuna (Priority: P2)

**Goal**: Automatically search for optimal hyperparameters using Optuna with MLflow integration

**Independent Test**: Run Optuna study with small search space, verify trials logged to both Optuna DB and MLflow

### Implementation for User Story 4

- [ ] T044 [US4] Implement objective function in scripts/hpo.py:
  - Sample hyperparameters from Hydra-defined search space
  - Run training with sampled params
  - Return validation F1 as objective
  - Log trial to MLflow
- [ ] T045 [US4] Setup Optuna study with SQLite storage in scripts/hpo.py (sqlite:///optuna.db, maximize F1, pruning strategy)
- [ ] T046 [US4] Implement HPO script in scripts/hpo.py:
  - Parse Hydra config for search space
  - Create Optuna study
  - Run N trials
  - Print best params and metric
  - Optionally train final model with best params
- [ ] T047 [P] [US4] Add integration test for HPO in tests/integration/test_hpo.py (2 trials, verify study created, best params identified)

**Checkpoint**: HPO pipeline complete - can automatically tune hyperparameters with Optuna

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

### Documentation

- [ ] T048 [P] Verify research.md contains all technical decisions (focal loss, splitting, optimizations, pairing strategy)
- [ ] T049 [P] Verify data-model.md defines all entities (CriterionSentencePair, ReDSM5Dataset, etc.) with fields and relationships
- [ ] T050 [P] Verify quickstart.md has working examples (train, eval, HPO, MLflow UI, config overrides)
- [ ] T051 [P] Update README.md with feature description and quickstart link

### Code Quality

- [ ] T052 Run Black formatting on all code: `black src tests scripts --line-length 100`
- [ ] T053 Run Ruff linting: `ruff check src tests scripts --fix`
- [ ] T054 Run MyPy type checking: `mypy src tests scripts` and fix all errors
- [ ] T055 Run pytest test suite: `pytest tests/` and ensure all pass
- [ ] T056 Remove all TODO markers and placeholder comments
- [ ] T057 Add Google-style docstrings with type hints to all functions

### Reproducibility Validation

- [ ] T058 [P] Test seed reproducibility (run training twice with same seed, verify metrics match within FP precision)
- [ ] T059 [P] Verify Git SHA logged to MLflow runs
- [ ] T060 [P] Verify pip freeze logged to MLflow runs
- [ ] T061 [P] Verify full Hydra config logged to MLflow runs

### Performance Validation

- [ ] T062 [P] Benchmark training with optimizations vs baseline (target: 2x speedup)
- [ ] T063 [P] Benchmark memory usage with mixed precision vs FP32 (target: 30% reduction)
- [ ] T064 [P] Verify training completes <15 min/epoch on GPU for bert-base-uncased
- [ ] T065 [P] Verify model achieves F1 > 0.6 on validation set (baseline)

### Edge Case Handling

- [ ] T066 Test GPU OOM handling (reduce batch size, enable gradient checkpointing, test gradient accumulation)
- [ ] T067 Test torch.compile failures (make optional via config, verify graceful fallback)
- [ ] T068 Test FlashAttention unavailable (verify automatic fallback to SDPA)
- [ ] T069 Test bf16 unsupported GPU (verify fallback to fp16 with GradScaler)
- [ ] T070 Test MLflow DB unavailable (verify graceful error with informative message)
- [ ] T071 Test missing DSM-5 criteria files (verify fail-fast with clear error)
- [ ] T072 Test very long sentences >512 tokens (verify truncation works correctly)

### Final Validation

- [ ] T073 Run end-to-end quickstart validation (follow quickstart.md commands on fresh setup)
- [ ] T074 Verify at least one trained model exists in MLflow registry
- [ ] T075 Verify evaluation produces per-symptom metrics for all 10 categories
- [ ] T076 Verify constitution compliance checklist (P1-P6) all pass

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 5 (Phase 3)**: Depends on Foundational - BLOCKS US1, US2, US3 (data prerequisite)
- **User Story 1 (Phase 4)**: Depends on US5 completion (needs data)
- **User Story 2 (Phase 5)**: Depends on US1 completion (needs trained model)
- **User Story 3 (Phase 6)**: Depends on US1 completion (tests config during training)
- **User Story 4 (Phase 7)**: Depends on US1, US3 completion (needs training + config)
- **Polish (Phase 8)**: Depends on all user stories being complete

### Critical Path

```
Setup → Foundational → US5 (Data) → US1 (Train) → US2 (Eval)
                                                  → US3 (Config)
                                                  → US4 (HPO)
```

### Parallel Opportunities

**Phase 2 (Foundational):**
- T004, T005, T006 (docs) can run in parallel
- T007, T008 (utilities) can run in parallel
- T009, T010 (tests) can run in parallel after respective utilities
- T011-T015 (all configs) can run in parallel

**Phase 3 (US5):**
- T023, T024, T025 (all tests) can run in parallel after implementation

**Phase 4 (US1):**
- T033 can run in parallel with other phases once US1 implementation complete

**Phase 5 (US2):**
- T038 can run in parallel with other phases once US2 implementation complete

**Phase 6 (US3):**
- T043 can run in parallel after T039-T042

**Phase 7 (US4):**
- T047 can run in parallel after T044-T046

**Phase 8 (Polish):**
- T048, T049, T050, T051 (docs) can run in parallel
- T052, T053, T054, T055 (quality) can run sequentially (fix issues between runs)
- T058, T059, T060, T061 (reproducibility) can run in parallel
- T062, T063, T064, T065 (performance) can run in parallel
- T066-T072 (edge cases) can run in parallel

---

## Implementation Strategy

### MVP First (US5 + US1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 5 (Data loading - prerequisite)
4. Complete Phase 4: User Story 1 (Training)
5. **STOP and VALIDATE**: Train a model end-to-end, verify MLflow tracking
6. Basic MVP ready: can load data and train models

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add US5 (Data) → Test independently → Data loading verified
3. Add US1 (Train) → Test independently → MVP: Can train models!
4. Add US2 (Eval) → Test independently → Can evaluate models
5. Add US3 (Config) → Test independently → Full configuration flexibility
6. Add US4 (HPO) → Test independently → Automated hyperparameter tuning
7. Polish → Final validation → Production ready

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. One developer focuses on US5 (Data) - everyone else blocked
3. Once US5 done:
   - Developer A: US1 (Train) - critical path
   - Developer B: US3 (Config) - can work on config structure
   - Developer C: Documentation (research.md, data-model.md, quickstart.md)
4. Once US1 done:
   - Developer A: US2 (Eval)
   - Developer B: US4 (HPO)
   - Developer C: Tests and polish
5. Final validation together

---

## Task Count Summary

- **Phase 1 (Setup)**: 3 tasks
- **Phase 2 (Foundational)**: 12 tasks
- **Phase 3 (US5 - Data)**: 10 tasks
- **Phase 4 (US1 - Train)**: 8 tasks
- **Phase 5 (US2 - Eval)**: 5 tasks
- **Phase 6 (US3 - Config)**: 5 tasks
- **Phase 7 (US4 - HPO)**: 4 tasks
- **Phase 8 (Polish)**: 29 tasks

**Total**: 76 tasks

**Estimated Duration**:
- MVP (Setup + Foundational + US5 + US1): ~40 tasks → 5-7 days solo, 3-4 days with team
- Full Feature (All phases): 76 tasks → 10-14 days solo, 6-8 days with team

---

## Notes

- [P] tasks = different files, no dependencies, can run in parallel
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate progress
- US5 must be completed first as it's a prerequisite for all other stories
- Focus on MVP (US5 + US1) before expanding to other stories
