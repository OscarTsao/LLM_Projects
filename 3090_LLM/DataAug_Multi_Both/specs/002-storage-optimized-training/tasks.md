# Implementation Tasks: Storage-Optimized Training & HPO Pipeline

**Feature Branch**: `002-storage-optimized-training`
**Generated**: 2025-10-10
**Updated**: 2025-10-10 (updated to per-study evaluation per constitution v1.1.0; added T027a, T047a for FR-033, FR-034)
**Total Tasks**: 49
**Estimated Duration**: 8-10 weeks (1 developer)

**Recent Updates**:
- User Story 3 updated to **per-study test evaluation** (constitution v1.1.0 + spec). Hybrid approach: each trial evaluates on validation set during training; test set evaluation occurs once per study after all trials complete, evaluating only the best model from the entire study. Study-level JSON report saved in study directory.

---

## Task Organization

Tasks are organized by user story to enable independent implementation and testing:

- **Phase 1**: Setup & Infrastructure (T001-T008)
- **Phase 2**: Foundational Components (T009-T018) - Must complete before user stories
- **Phase 3**: User Story 1 - Storage-Optimized Training/HPO with Resume (T019-T030) [P1]
- **Phase 4**: User Story 2 - Portable Environment (T031-T036) [P2]
- **Phase 5**: User Story 3 - Per-Study Test Evaluation & JSON Reports (T037-T042) [P3]
- **Phase 6**: Polish & Integration (T043-T047)

**Legend**:
- `[P]` = Parallelizable (can work on simultaneously with other [P] tasks)
- `[US1]`, `[US2]`, `[US3]` = User Story labels
- `[BLOCKING]` = Must complete before dependent tasks

---

## Phase 1: Setup & Infrastructure

### T001: Initialize Project Structure [BLOCKING] ✅ COMPLETE
**Files**: Repository root structure  
**Description**: Create complete project directory structure per plan.md
- Create `src/` with subdirectories: `models/`, `data/`, `training/`, `hpo/`, `utils/`, `cli/`
- Create `tests/` with subdirectories: `contract/`, `integration/`, `unit/`
- Create `configs/` with subdirectories: `model/`, `data/`, `retention_policy/`
- Create `docker/` directory
- Create `experiments/mlflow_db/` directory
- Initialize `__init__.py` files in all Python packages

**Acceptance**: All directories exist, Python packages importable

---

### T002: Set Up Poetry Dependency Management [BLOCKING] [P] ✅ COMPLETE
**Files**: `pyproject.toml`, `poetry.lock`  
**Description**: Configure Poetry for dependency management per constitution
- Create `pyproject.toml` with project metadata and dependencies:
  - PyTorch 2.0+, transformers, datasets (Hugging Face)
  - MLflow, Optuna, TextAttack, torchcrf, peft
  - hydra-core, pytest, pytest-cov, hypothesis
  - ruff, black, mypy
- Run `poetry lock` to generate exact version pins
- Add Poetry installation instructions to README

**Acceptance**: `poetry install` succeeds, all dependencies installed

---

### T003: Create Makefile [BLOCKING] [P] ✅ COMPLETE
**Files**: `Makefile`  
**Description**: Implement Makefile-driven operations per constitution Principle VII
- Add targets: `help`, `train`, `resume`, `evaluate`, `cleanup`, `test`, `lint`, `format`
- Each target should use Docker Compose for containerized execution
- Add self-documenting help text for each target
- Test all targets execute without errors

**Acceptance**: `make help` shows all targets, `make test` runs successfully

---

### T004: Set Up Hydra Configuration Framework [BLOCKING] [P] ✅ COMPLETE
**Files**: `configs/hpo_study.yaml`, `src/utils/config.py`  
**Description**: Implement Hydra-based configuration management
- Create base HPO study config template in `configs/hpo_study.yaml`
- Implement `src/utils/config.py` with Hydra config loading and schema validation
- Add config composition support for model/data/retention_policy overrides
- Validate against `contracts/config_schema.yaml`

**Acceptance**: Config loads successfully, validation catches schema violations

---

### T005: Implement Exponential Backoff Retry Utility [BLOCKING] [P] ✅ COMPLETE
**Files**: `src/utils/retry.py`  
**Description**: Create retry decorator for fault tolerance (FR-005, FR-017)
- Implement `@exponential_backoff` decorator with configurable max_attempts, base_delay, max_delay
- Default: delays [1s, 2s, 4s, 8s, 16s], max 5 attempts
- Log retry attempts at WARNING level
- Raise original exception after exhaustion
- Add unit tests for retry logic

**Acceptance**: Decorator retries with exponential backoff, tests pass

---

### T006: Implement Dual Logging System [BLOCKING] [P] ✅ COMPLETE
**Files**: `src/utils/logging.py`  
**Description**: Create structured JSON + human-readable stdout logging (FR-020)
- Implement `LogEvent` dataclass with fields: timestamp, level, message, component, trial_id, extra
- Create dual logger that writes to both JSONL file and stdout
- Implement `format_storage_exhaustion_error()` with detailed error context (FR-014)
- Add log rotation to prevent unbounded growth (daily or at 1GB, retain last 14 files)
- Implement comprehensive log sanitization (FR-032): mask tokens, API keys, bearer tokens, passwords, email addresses in JSON/stdout/exception messages/MLflow tags; include unit tests for masking patterns

**Acceptance**: Logs written to both JSON and stdout, error formatter includes artifact enumeration

---

### T007: Implement Storage Monitoring Background Thread [BLOCKING] [P] ✅ COMPLETE
**Files**: `src/utils/storage_monitor.py`  
**Description**: Create background disk space monitoring (FR-018, Principle II)
- Implement background thread checking disk usage every 60 seconds using `shutil.disk_usage()`
- Monitor filesystem containing `experiments/` directory
- Set `STORAGE_CRITICAL` flag when available space < 10%
- Thread-safe communication via `threading.Event`
- Log: INFO every 10 minutes, WARNING at <10%, ERROR at <5%
- Lifecycle: start at training init, stop at trial completion

**Acceptance**: Thread monitors disk, sets flag at 10% threshold, logs appropriately

---

### T008: Create Contract Validation Test Framework [P] ✅ COMPLETE
**Files**: `tests/contract/test_config_schema.py`, `tests/contract/test_output_formats.py`, `tests/contract/test_checkpoint_metadata.py`  
**Description**: Implement contract validation tests
- Test TrialConfig validation against `contracts/config_schema.yaml`
- Test EvaluationReport validation against `contracts/trial_output_schema.json`
- Test checkpoint metadata validation against `contracts/checkpoint_metadata.json`
- Use jsonschema library for validation

**Acceptance**: All contract tests pass, invalid schemas rejected

---

## Phase 2: Foundational Components (Must Complete Before User Stories)

### T009: Implement Data Loading & Preprocessing [BLOCKING] ✅ COMPLETE
**Files**: `src/data/dataset.py`, `src/data/preprocessing.py`  
**Description**: Load dataset strictly from Hugging Face Datasets with explicit splits
- Load from HF Hub using a concrete dataset identifier with `split="train"`, `split="validation"`, and `split="test"` (default id: `irlab-udc/redsm5`, optional revision: `main`)
- Do not rely on local CSV fallbacks; adhere to constitution Data Handling
- Support both `binary_pairs` and `multi_label` input formats
- Validate split disjointness (no `post_id` in multiple splits) and determinism across runs
- Validate dataset identifier format and required splits presence; if `dataset.revision` set, pin exact revision/tag/commit and log resolved hash to MLflow (FR-026)
- Failure handling (FR-029): On invalid id or missing/corrupt splits, abort with actionable error including attempted id/revision, required vs detected splits, and remediation instructions; add negative tests

**Acceptance**: Data loads from HF Datasets with explicit splits; id/revision validated; negative tests cover invalid id/missing splits; MLflow logs include resolved dataset revision/hash

---

### T010: Implement TextAttack Data Augmentation [P] ✅ COMPLETE
**Files**: `src/data/augmentation.py`  
**Description**: Integrate TextAttack for evidence augmentation
- Implement augmentation methods: synonym, insert, swap, back_translation, char_perturb
- Apply augmentation only to evidence text, preserve non-evidence regions
- Configurable per-example probability (0.0-0.5)
- Track augmentation metadata (original vs augmented evidence)
- Add property-based tests: augmentation preserves non-evidence text

**Acceptance**: Augmentation applies correctly, tests pass

---

### T011: Implement Hugging Face Model Encoder Wrapper [BLOCKING] [P] ✅ COMPLETE
**Files**: `src/models/encoders/hf_encoder.py`
**Description**: Create wrapper for loading Hugging Face transformer models from catalog
- Implement `HFEncoder` class that loads models by `model_id`
- Initial catalog: 5 validated models (mental-bert, psychbert, clinicalbert, bert-base, roberta-base)
- Architecture supports expansion to 30+ models after validation
- Check local cache first, retry download with exponential backoff (FR-005)
- Support gradient checkpointing for memory efficiency
- Handle model-specific tokenizers
- Authentication failure handling (FR-032): actionable errors for missing/expired tokens with login guidance; ensure sensitive data is not logged (rely on T006 sanitization)
- Add unit tests for cache-first loading and retry logic

**Acceptance**: Models load from initial catalog (5 models), cache and retry work, auth failures produce actionable errors without leaking secrets, tests pass, architecture supports future expansion

---

### T012: Implement Criteria Matching Heads [BLOCKING] [P] ✅ COMPLETE
**Files**: `src/models/heads/criteria_matching.py`
**Description**: Create task-specific heads for criteria classification
- Implement head types: `linear`, `mlp`, `mlp_residual`, `gated`, `multi_sample_dropout`
- Implement pooling strategies: `cls`, `mean`, `attention`, `last_2_layer_mix`
- Support both binary (9 separate classifiers) and multi-label (single 9-output classifier)
- Configurable hidden dimensions (128-1024) and dropout (0.1-0.5)
- Add unit tests for each head type

**Acceptance**: All head types work, pooling strategies tested

---

### T013: Implement Evidence Binding Heads [BLOCKING] [P] ✅ COMPLETE
**Files**: `src/models/heads/evidence_binding.py`
**Description**: Create span extraction heads for evidence binding
- Implement head types: `start_end_linear`, `start_end_mlp`, `biaffine`, `bio_crf`, `sentence_reranker`
- Support null span prediction (no evidence cases)
- Integrate torchcrf for BIO+CRF variant
- Add unit tests for span prediction logic

**Acceptance**: All head types predict spans correctly, null spans handled

---

### T014: Implement Multi-Task Model Architecture [BLOCKING] ✅ COMPLETE
**Files**: `src/models/multi_task.py`
**Description**: Combine encoder and heads with optional task coupling
- Implement `MultiTaskModel` with shared encoder
- Support `independent` and `coupled` task modes
- For coupled mode: implement coupling methods (concat, add, mean, weighted_sum, weighted_mean)
- Implement pooling before combination (mean, max, attention)
- Add `from_config()` factory method for TrialConfig
- Add integration tests for both coupling modes

**Acceptance**: Model supports both modes, coupling methods work, tests pass

---

### T015: Implement Loss Functions [BLOCKING] [P] ✅ COMPLETE
**Files**: `src/models/losses.py`
**Description**: Create loss functions for training
- Implement: `bce`, `weighted_bce`, `focal`, `adaptive_focal`, `hybrid`
- Focal loss with configurable gamma (1.0-5.0)
- Hybrid loss with alpha weighting (0.1-0.9)
- Label smoothing support (0.0-0.2)
- Class weighting support for imbalanced data
- Add unit tests for each loss function

**Acceptance**: All loss functions compute correctly, tests pass

---

### T016: Implement Checkpoint Manager with Integrity Validation [BLOCKING] ✅ COMPLETE (partial tests)
**Files**: `src/training/checkpoint_manager.py`
**Description**: Manage checkpoint retention and pruning with SHA256 validation (FR-004, FR-024, Principle V)
- Implement atomic checkpoint writes (temp file → rename)
- Compute SHA256 hash on save, store in metadata and `.hash` file
- Validate hash on load, fall back to previous checkpoint if corrupted
- Implement retention policy: keep_last_n, keep_best_k, max_total_size_gb
- Prune non-protected checkpoints when limits exceeded or disk < 10%
- Track co-best checkpoints (tied for max optimization metric)
- Add integration tests: corruption recovery, retention policy enforcement

 - Preflight storage estimation (FR-034): estimate checkpoint size and required free space before training and before each save; abort early with actionable error if capacity insufficient even after quantified aggressive pruning
 - Embed compatibility metadata (FR-025): code version (git commit or version), model architecture signature, head configurations to support resume-time compatibility validation

**Acceptance**: Checkpoints saved atomically, integrity validated, preflight storage checks enforced, compatibility metadata embedded, pruning works, tests pass

---

### T017: Implement MLflow Metrics Buffering [BLOCKING] [P] ✅ COMPLETE
**Files**: `src/hpo/metrics_buffer.py`
**Description**: Buffer metrics during MLflow outages (FR-017, Principle IV)
- Implement disk-backed JSONL buffer for metrics
- Automatically replay with exponential backoff when backend available
- Keep buffer file until successful upload confirmed
- Warn when buffer exceeds 100MB (no hard limit)
- Add integration tests: outage scenario, replay with backoff

**Acceptance**: Metrics buffered to disk, replayed on recovery, tests pass

---

### T018: Implement Optuna Search Space Definition [BLOCKING] [P] ✅ COMPLETE
**Files**: `src/hpo/search_space.py`
**Description**: Define HPO search space mapping to TrialConfig
- Map all hyperparameters from `contracts/config_schema.yaml` to Optuna suggestions
- Support categorical, int, float (log scale), and conditional parameters
- Implement conditional logic (e.g., if task_coupling=coupled, suggest coupling_method)
- Validate suggested configs against schema
- Add unit tests for search space generation

**Acceptance**: Search space generates valid TrialConfigs, conditionals work, tests pass

---

## Phase 3: User Story 1 - Storage-Optimized Training/HPO with Resume [P1]

**Goal**: Enable ML engineers to run long-running HPO without exhausting storage, with automatic resume capability.

**Independent Test**: Launch HPO with aggressive retention policy; verify metrics logged, checkpoints pruned, interrupted job resumes successfully.

---

### T019: [US1] Implement Training Loop with Epoch-Based Checkpointing ✅ COMPLETE
**Files**: `src/training/trainer.py`
**Description**: Core training loop for single trial
- Implement epoch-based training with validation after each epoch
- Minimum checkpoint interval: 1 epoch (FR-022)
- Track optimization metric (user-specified, always maximize)
- Integrate with checkpoint_manager for atomic saves
- Check `STORAGE_CRITICAL` flag before each checkpoint save
- Support deterministic seeding for reproducibility (FR-011, FR-027): seed Python `random`, NumPy, PyTorch CPU/CUDA, and DataLoader workers; record seeds in MLflow and per-trial report
- Add integration test: full training run with checkpointing

**Acceptance**: Training completes, checkpoints saved per epoch, metrics tracked

---

### T020: [US1] Implement Sequential Trial Executor [P] ✅ COMPLETE
**Files**: `src/hpo/trial_executor.py`
**Description**: Execute HPO trials sequentially with cleanup (FR-021)
- Implement trial execution loop with Optuna integration
- Configure `n_jobs=1` for sequential execution
- Create trial directory structure: `experiments/trial_<uuid>/`
- Execute: train → validate → checkpoint → cleanup
- Handle trial failures gracefully (log error, continue to next trial)
- Emit HPO progress observability (FR-033): trial index/total, completion rate, best-so-far, ETA if available; log to JSON and MLflow params/tags
- Add integration test: multi-trial HPO execution

**Acceptance**: Trials execute sequentially, failures handled, HPO progress signals emitted to logs/MLflow, tests pass

---

### T021: [US1] Implement Resume from Latest Checkpoint ✅ COMPLETE
**Files**: `src/training/trainer.py` (extend)
**Description**: Support automatic resume after interruption (FR-004, Principle V)
- Detect latest valid checkpoint in trial directory
- Validate checkpoint integrity (SHA256 hash)
- Fall back to previous checkpoint if corruption detected
- Restore model, optimizer, scheduler, RNG states
- Continue logging metrics without duplication
- Zero-checkpoint resume (FR-030): if interruption occurs before any checkpoint exists, resume from initial state (epoch 0) without metric duplication
- Idempotent resume on interruption (FR-031): use lock files or atomic state updates to ensure interrupted resume attempts can be retried safely without duplicating metrics
- Compatibility validation (FR-025): validate checkpoint code version/model signature/head types; if incompatible, fail with actionable error; allow optional migration hooks if configured; embed metadata during save (T016) to support validation
- Add integration test: interrupt training, resume, verify no metric duplication
 - Add integration test: resume with zero checkpoints; interrupt resume and retry; resume after code changes (compatibility failure path)

**Acceptance**: Training resumes from latest checkpoint, metrics not duplicated, zero-checkpoint and interrupted-resume scenarios handled, compatibility validated, tests pass

---

### T022: [US1] Implement Proactive Retention Pruning ✅ COMPLETE (partial tests)
**Files**: `src/training/checkpoint_manager.py` (extend)
**Description**: Trigger aggressive pruning when disk < 10% (FR-018)
- Integrate with storage_monitor background thread
- When `STORAGE_CRITICAL` flag set, trigger aggressive pruning
- Prune oldest non-protected checkpoints first
- Quantified policy transitions (FR-028): if pruning non-protected is insufficient, reduce to `keep_best_k=1` and `keep_last_n=0` for subsequent trials; as a last resort, preserve only the single best checkpoint across the entire study
- If still insufficient space, generate detailed error (FR-014)
- Error includes: current usage, space needed, largest artifacts, cleanup commands
- Add integration test: simulate low disk, verify pruning triggered

**Acceptance**: Pruning triggered at 10%, quantified policy transitions enforced, detailed error on failure, property tests cover policy invariants, tests pass

---

### T023: [US1] Implement CLI Entry Point ✅ COMPLETE for Training
**Files**: `src/cli/train.py`
**Description**: Main CLI for training and HPO (FR-010)
- Support modes: `single` (one trial), `hpo` (multi-trial), `dry-run` (validation only)
- Accept Hydra config via `--config` flag
- Support `--resume` flag for interrupted HPO
- Add `--dry-run` flag to disable checkpointing (metrics still logged, for evaluation-only runs per FR-010)
- Add `--no-checkpoint` flag as alias for dry-run mode
- Integrate with MLflow tracking (local database)
- Display progress bars (tqdm) for epochs and trials
- Add integration test: CLI invocation with various flags

**Acceptance**: CLI runs training/HPO, resume works, progress displayed, --dry-run disables checkpointing while preserving metrics logging

---

### T024: [US1] Integration Test - Full HPO with Resume
**Files**: `tests/integration/test_full_hpo_resume.py`
**Description**: End-to-end test for User Story 1
- Launch HPO with 5 trials, aggressive retention policy
- Interrupt after trial 2 completes
- Resume HPO, verify trials 3-5 execute
- Verify metrics logged for all trials
- Verify only latest N and best K checkpoints retained
- Verify no metric duplication after resume

**Acceptance**: Test passes, all US1 acceptance criteria met

---

### T025: [US1] Integration Test - Storage Exhaustion Scenario
**Files**: `tests/integration/test_storage_exhaustion.py`
**Description**: Test proactive pruning at 10% threshold
- Mock disk usage to simulate <10% available space
- Trigger checkpoint save
- Verify aggressive pruning triggered
- Verify detailed error message if pruning insufficient
- Verify all metrics preserved despite pruning

**Acceptance**: Test passes, pruning triggered, error message detailed

---

### T026: [US1] Integration Test - Checkpoint Corruption Recovery
**Files**: `tests/integration/test_checkpoint_corruption.py`
**Description**: Test integrity validation and fallback
- Save checkpoint, corrupt file (modify bytes)
- Attempt resume
- Verify hash validation detects corruption
- Verify fallback to previous valid checkpoint
- Verify training continues successfully

**Acceptance**: Test passes, corruption detected, fallback works

---

### T027: [US1] Property-Based Test - Retention Invariants
**Files**: `tests/unit/test_checkpoint_retention_properties.py`
**Description**: Use hypothesis to test retention policy invariants
- Property: Always keep ≥1 checkpoint (never delete all)
- Property: Never delete co-best checkpoints
- Property: keep_last_n and keep_best_k respected
- Generate random checkpoint sequences, verify invariants hold

**Acceptance**: Property tests pass for all invariants

---

### T027a: [US1] Implement Preflight Storage Validation
**Files**: `src/training/storage_validator.py`, `src/training/trainer.py`
**Description**: Implement preflight storage checks before training and checkpointing (FR-034)
- Estimate checkpoint size using formula: `model.state_dict() byte size + optimizer.state_dict() byte size + 1MB metadata`
- Calculate required free space: `estimated_checkpoint_size × (keep_last_n + keep_best_k)`
- Check available disk space before training starts
- Check available disk space before each checkpoint save
- If predicted to exceed capacity even after aggressive pruning, abort with actionable error:
  - Current disk usage and available space
  - Projected checkpoint size
  - Required free space for retention policy
  - List of largest artifacts (with sizes and paths)
  - Remediation options (adjust retention policy, clean artifacts, add storage)
- Add unit tests for size estimation and validation logic

**Acceptance**: Preflight checks prevent training when insufficient storage, error messages actionable

---

### T028: [US1] Performance Test - Checkpoint Save Overhead
**Files**: `tests/integration/test_checkpoint_performance.py`
**Description**: Verify checkpoint save overhead ≤30s per epoch
- Train for 3 epochs with 1-10GB model
- Measure time for checkpoint save (including hash computation)
- Verify overhead ≤30s per epoch
- Verify metric logging latency <1s per batch

**Acceptance**: Performance targets met

---

### T029: [US1] Performance Test - Storage Monitoring Thread CPU
**Files**: `tests/integration/test_storage_monitor_performance.py`
**Description**: Verify storage monitoring thread CPU usage <1%
- Run storage monitor for 10 minutes
- Measure CPU usage via psutil
- Verify average CPU usage <1%

**Acceptance**: CPU usage target met

---

### T030: [US1] Documentation - User Story 1 Quickstart
**Files**: `docs/us1_quickstart.md`
**Description**: Document how to use storage-optimized training/HPO
- Provide example config for aggressive retention policy
- Document resume workflow
- Provide troubleshooting guide for storage exhaustion
- Include example MLflow queries for best trials

**Acceptance**: Documentation complete, examples work

---

**CHECKPOINT**: User Story 1 Complete - Storage-optimized training/HPO with resume capability functional

---

## Phase 4: User Story 2 - Portable Environment [P2]

**Goal**: Enable ML engineers to run training consistently across different machines using containerized environment.

**Independent Test**: On fresh machine, start container and run sample training within 15 minutes.

---

### T031: [US2] Create Dockerfile with Poetry
**Files**: `docker/Dockerfile`
**Description**: Multi-stage Docker build with Poetry (Principle VI)
- Base image: `python:3.10-slim`
- Install Poetry 1.7.0
- Copy `pyproject.toml` and `poetry.lock`
- Run `poetry install --no-dev --no-interaction`
- Configure for GPU support (CUDA 11.8+)
- Set up Hugging Face cache mount point
- Add healthcheck for container readiness

**Acceptance**: Image builds successfully, Poetry installs dependencies

---

### T032: [US2] Create Docker Compose Configuration [P]
**Files**: `docker/docker-compose.yml`
**Description**: Orchestrate container with volume mounts
- Define service: `trainer`
- Mount volumes: `Data/` (read-only), `experiments/`, `~/.cache/huggingface/`
- Configure GPU access (`--gpus all`)
- Set environment variables for MLflow, Optuna
- Add profiles for CPU-only vs GPU execution

**Acceptance**: `docker-compose up` starts container with correct mounts

---

### T033: [US2] Integration Test - Fresh Machine Setup
**Files**: `tests/integration/test_portable_environment.py`
**Description**: Test container setup on fresh environment
- Simulate fresh machine (clean Docker cache)
- Build image from scratch
- Start container
- Run sample training (1 epoch, small model)
- Measure total time from `docker build` to training completion
- Verify completes within 15 minutes (moderate network)

**Acceptance**: Test passes, 15-minute target met

---

### T034: [US2] Integration Test - Cross-Machine Consistency
**Files**: `tests/integration/test_cross_machine_consistency.py`
**Description**: Verify reproducibility across environments
- Run same trial config on two different containers
- Use identical seed, data, hyperparameters
- Verify final metrics match (within floating-point tolerance)
- Verify checkpoint hashes match

**Acceptance**: Results reproducible across containers

---

### T035: [US2] Documentation - Container Setup Guide
**Files**: `docs/us2_container_setup.md`
**Description**: Document portable environment setup
- Prerequisites (Docker, GPU drivers)
- Build and run instructions
- Volume mount explanations
- Troubleshooting common issues (GPU access, permissions)

**Acceptance**: Documentation complete, setup instructions work

---

### T036: [US2] Update Makefile for Container Operations
**Files**: `Makefile` (extend)
**Description**: Add container-specific targets
- `make build` - Build Docker image
- `make shell` - Start interactive container shell
- `make train-container` - Run training in container
- All targets use `docker-compose` for consistency

**Acceptance**: Makefile targets work, container operations functional

---

**CHECKPOINT**: User Story 2 Complete - Portable containerized environment functional

---

## Phase 5: User Story 3 - Per-Study Test Evaluation & JSON Reports [P3]

**Goal**: Enable researchers to evaluate the best model from the entire HPO study on the test set and generate a study-level machine-readable JSON report.

**Clarification**: Evaluation is **per-study** (hybrid approach) - each trial evaluates on validation set during training; test set evaluation runs once after all trials complete, evaluating only the best model from the entire study. Co-best checkpoints within the best trial are all evaluated and included in the study report.

**Independent Test**: After HPO completes, verify the study directory contains a valid JSON report with test metrics for the best model from the entire study.

---

### T037: [US3] Implement Test Set Evaluator
**Files**: `src/training/evaluator.py`
**Description**: Evaluate a the best model from an HPO study on the held-out test set (FR-007, FR-008)
- Accept `study_id` as input
- Identify best trial from the study (by optimization metric)
- Load best checkpoint(s) from the best trial (include co-best if tied)
- Evaluate on test set (loaded from Hugging Face datasets, split="test")
- Compute criteria matching metrics: accuracy, F1 (macro/micro), precision, recall, per-criterion (with PR AUC and confusion matrices for Feature 001)
- Compute evidence binding metrics: exact_match, F1, char_F1, null_span_accuracy
- Return structured metrics dict matching `contracts/study_output_schema.json`

**Acceptance**: Evaluator computes all required metrics for best model from study, schema validated

---

### T038: [US3] Implement JSON Report Generator
**Files**: `src/training/evaluator.py` (extend)
**Description**: Generate EvaluationReport JSON per trial
- Create report with fields: report_id, trial_id, generated_at, config, optimization_metric_name, best_validation_score
- Include evaluated_checkpoints array (all co-best checkpoints from the trial)
- Include test_metrics (criteria_matching + evidence_binding)
- Include decision_thresholds (optional, for Feature 001 integration)
- Save to `experiments/trial_<uuid>/evaluation_report.json`
- Validate against `contracts/trial_output_schema.json`

**Acceptance**: JSON reports generated per trial, schema validation passes

---

### T039: [US3] Integrate Evaluation into Trial Completion
**Files**: `src/hpo/trial_executor.py` (extend), `src/cli/evaluate.py` (new)
**Description**: Run evaluation after each trial completes
- After a trial completes, identify best checkpoint(s) by optimization metric
- Call evaluator on that trial’s best checkpoint(s)
- Generate and save JSON report to the trial directory
- Log report path to MLflow as artifact
- Handle co-best checkpoints (evaluate all, include all in report)
- Create CLI command for manual evaluation: `python -m src.cli.evaluate --trial-id <uuid>`

**Acceptance**: Evaluation runs after each trial completion, report saved to trial directory

---

### T040: [US3] Contract Test - EvaluationReport Schema
**Files**: `tests/contract/test_output_formats.py` (extend)
**Description**: Validate generated reports against schema
- Generate sample EvaluationReport
- Validate against `contracts/study_output_schema.json` using jsonschema
- Test edge cases: co-best checkpoints, missing optional fields
- Verify per-criterion metrics structure

**Acceptance**: Contract tests pass, schema violations detected

---

### T041: [US3] Integration Test - Per-Study Test Evaluation
**Files**: `tests/integration/test_per_study_evaluation.py`
**Description**: End-to-end test for User Story 3
- Run HPO with 5 trials
- Verify the study directory contains `evaluation_report.json`
- Verify the report has all required fields (study_id, best_trial_id, test_metrics, config, decision_thresholds)
- Verify test metrics correspond to test set evaluation of the best model from the entire study
- Verify co-best checkpoints within best trial handled correctly
- Verify evaluation runs after all trials complete

**Acceptance**: Test passes, all US3 acceptance criteria met

---

### T042: [US3] Documentation - Report Analysis Guide
**Files**: `docs/us3_report_analysis.md`
**Description**: Document how to analyze JSON reports
- Provide Python examples for loading and aggregating reports
- Show how to query best trials by test metrics
- Provide visualization examples (matplotlib/seaborn)
- Document report schema and field meanings

**Acceptance**: Documentation complete, examples work

---

**CHECKPOINT**: User Story 3 Complete - Per-study test evaluation and JSON reporting functional

---

### T042a: [US3 - Optional] Study-Level Summary Report
**Files**: `src/dataaug_multi_both/cli/evaluate_study.py` (new), `specs/002-storage-optimized-training/contracts/study_summary_schema.json` (new)
**Description**: Generate an optional study-level summary JSON that aggregates trial metadata (FR-035)
- Accept `study_id` (or derive from Optuna study DB)
- Identify `best_trial_id` by the optimization metric and retrieve its evaluation report path
- Create `experiments/study_<uuid>/summary_report.json` with fields:
  - `report_id`, `study_id`, `generated_at`, `best_trial_id`, `best_validation_score`, `best_trial_evaluation_report_path`, `optimization_metric_name`
- Optionally include `trials_count` and `top_trials` (id, validation_score)
- Validate against `contracts/study_summary_schema.json`

**Acceptance**: Study summary report generated successfully; schema validation passes; this is supplementary to the per-study evaluation report (T038)

---

## Phase 6: Polish & Integration

### T043: Implement End-to-End Integration Test
**Files**: `tests/integration/test_e2e_hpo.py`
**Description**: Comprehensive end-to-end test covering all user stories
- Run full HPO workflow: setup → train → interrupt → resume → complete study → evaluate
- Verify all 3 user stories' acceptance criteria
- Verify storage optimization (≥60% reduction vs keep-all)
- Verify portable environment (container-based execution)
- Verify per-trial JSON reports generated (one per trial)
- Verify evaluation runs after each trial completes

**Acceptance**: E2E test passes, all user stories functional

---

### T044: Achieve 80% Test Coverage
**Files**: All test files
**Description**: Ensure ≥80% coverage for core modules (Constitution requirement)
- Measure coverage via pytest-cov
- Add missing unit tests for uncovered code paths
- Focus on `src/training/`, `src/models/`, `src/data/`, `src/hpo/`
- Generate HTML coverage report
- Configure CI to fail if coverage <80%

**Acceptance**: Coverage ≥80%, CI configured

---

### T045: Code Quality - Linting and Formatting
**Files**: All Python files
**Description**: Ensure code passes all quality checks (FR-016)
- Run `ruff check .` and fix all violations
- Run `black .` to format code
- Run `mypy src/` and fix type errors
- Validate `docker/requirements.txt` is in sync with `poetry.lock`: `poetry export -f requirements.txt --output /tmp/req.txt --without-hashes && diff /tmp/req.txt docker/requirements.txt`
- Add pre-commit hooks for automated checks
- Add CI check to fail if requirements.txt is out of sync

**Acceptance**: All linters pass, no errors, requirements.txt validated in sync with poetry.lock

---

### T046: Documentation - Complete README
**Files**: `README.md`
**Description**: Comprehensive project README
- Project overview and key features
- Quick start guide (link to quickstart.md)
- Installation instructions (Poetry + Docker)
- Usage examples for all 3 user stories
- Architecture diagram
- Contributing guidelines
- License information

**Acceptance**: README complete, clear, accurate

---

### T047: Final Validation - Constitution Compliance Check
**Files**: All project files
**Description**: Verify all constitutional principles satisfied
- I. Reproducibility-First: Verify deterministic seeding, Poetry locks, config versioning
- II. Storage-Optimized: Verify retention policies, 10% threshold, metrics preserved
- III. Dual-Agent Architecture: Verify criteria matching + evidence binding functional
- IV. MLflow-Centric: Verify local DB, buffering, exponential backoff
- V. Auto-Resume: Verify SHA256 validation, atomic writes, corruption recovery
- VI. Portable Environment: Verify Docker + Poetry, documented mounts
- VII. Makefile-Driven: Verify all operations accessible via make targets

**Acceptance**: All 7 principles satisfied, no violations

---

### T047a: [US1] Implement HPO Progress Observability
**Files**: `src/hpo/progress_tracker.py`, `src/training/trainer.py`
**Description**: Implement HPO progress tracking and reporting (FR-033)
- Track trial-level progress: trial index/total, completion rate, best-so-far metric
- Calculate and display ETA (estimated time to completion) based on average trial duration
- Emit progress signals to both MLflow (params/tags) and JSON logs
- Display progress in stdout: "Trial 42/1000 (4.2%) | Best: 0.87 | ETA: 3h 15m"
- Log progress events at INFO level to JSON log file
- Add unit tests for progress calculation and ETA estimation

**Acceptance**: Progress tracking functional, ETA accurate within 10%, logged to MLflow and JSON

---

## Task Dependencies

### Critical Path (Must Complete in Order)
```
T001 (Project Structure)
  ↓
T002, T003, T004, T005, T006, T007, T008 [Parallel Setup]
  ↓
T009 (Data Loading) [BLOCKING for all user stories]
  ↓
T010, T011, T012, T013, T015, T017, T018 [Parallel Foundational]
  ↓
T014 (Multi-Task Model) [Depends on T011, T012, T013]
  ↓
T016 (Checkpoint Manager) [Depends on T007]
  ↓
T019 (Training Loop) [Depends on T014, T016]
  ↓
T020 (Trial Executor) [Depends on T019]
  ↓
T021, T022, T023 [US1 Implementation]
  ↓
T024-T030 [US1 Tests & Docs]
  ↓
T031-T036 [US2 - Can start after T023]
  ↓
T037-T042 [US3 - Can start after T020]
  ↓
T043-T047 [Polish - Requires all user stories complete]
```

### User Story Dependencies
- **US1** (T019-T030): Depends on Phase 1 & 2 complete
- **US2** (T031-T036): Depends on US1 T023 (CLI) complete
- **US3** (T037-T042): Depends on US1 T020 (Trial Executor) complete
- **US2 and US3 can be developed in parallel** after US1 core is complete

---

## Parallel Execution Opportunities

### Phase 1 (Setup) - 7 tasks can run in parallel after T001:
- T002 (Poetry), T003 (Makefile), T004 (Hydra), T005 (Retry), T006 (Logging), T007 (Storage Monitor), T008 (Contract Tests)
- **Estimated time**: 1 week (sequential) → 2-3 days (parallel with 3 developers)

### Phase 2 (Foundational) - 7 tasks can run in parallel after T009:
- T010 (Augmentation), T011 (Encoder), T012 (Criteria Heads), T013 (Evidence Heads), T015 (Losses), T017 (Metrics Buffer), T018 (Search Space)
- T014 (Multi-Task Model) depends on T011, T012, T013 completing
- T016 (Checkpoint Manager) depends on T007 completing
- **Estimated time**: 2 weeks (sequential) → 1 week (parallel with 3 developers)

### Phase 3 (US1) - Tests can run in parallel:
- T024-T029 (6 test tasks) can run in parallel after T019-T023 complete
- **Estimated time**: 1 week (sequential) → 2-3 days (parallel)

### Phase 4 (US2) & Phase 5 (US3) - Can run in parallel:
- US2 (T031-T036) and US3 (T037-T042) are independent after US1 core complete
- **Estimated time**: 2 weeks (sequential) → 1 week (parallel with 2 developers)

---

## Implementation Strategy

### MVP Scope (Minimum Viable Product)
**Goal**: Deliver User Story 1 (P1) first for immediate value

**MVP Tasks**: T001-T030 (30 tasks)
- **Duration**: 4-5 weeks (1 developer) or 2-3 weeks (2-3 developers)
- **Deliverable**: Functional storage-optimized HPO with resume capability
- **Value**: Enables long-running experiments without storage exhaustion

### Incremental Delivery Plan
1. **Sprint 1 (Weeks 1-2)**: Phase 1 & 2 (Setup + Foundational) - T001-T018
   - Deliverable: Core components ready for integration
2. **Sprint 2 (Weeks 3-4)**: Phase 3 (US1 Implementation) - T019-T023
   - Deliverable: Working training/HPO pipeline
3. **Sprint 3 (Week 5)**: Phase 3 (US1 Tests) - T024-T030
   - Deliverable: US1 complete, tested, documented
4. **Sprint 4 (Week 6)**: Phase 4 (US2 Portable Environment) - T031-T036
   - Deliverable: Containerized environment
5. **Sprint 5 (Week 7)**: Phase 5 (US3 Test Evaluation) - T037-T042
   - Deliverable: JSON reporting functional
6. **Sprint 6 (Week 8)**: Phase 6 (Polish) - T043-T047
   - Deliverable: Production-ready, all quality gates passed

### Testing Strategy
- **Unit tests**: Written alongside implementation (TDD where feasible)
- **Integration tests**: After each user story phase completes
- **Contract tests**: Validate schemas continuously
- **Property-based tests**: For critical invariants (retention, data splits)
- **Performance tests**: Verify targets (30s checkpoint, <1% CPU)
- **E2E test**: Final validation before release

---

## Task Summary

| Phase | Tasks | Estimated Duration | Parallelizable |
|-------|-------|-------------------|----------------|
| Phase 1: Setup | T001-T008 (8 tasks) | 1 week | 7 tasks after T001 |
| Phase 2: Foundational | T009-T018 (10 tasks) | 2 weeks | 7 tasks after T009 |
| Phase 3: US1 | T019-T030 + T027a (13 tasks) | 2 weeks | 6 test tasks |
| Phase 4: US2 | T031-T036 (6 tasks) | 1 week | 2 tasks |
| Phase 5: US3 | T037-T042 + T042a (7 tasks) | 1 week | Can run parallel with US2 |
| Phase 6: Polish | T043-T047 + T047a (6 tasks) | 1 week | 2 tasks |
| **Total** | **49 tasks** | **8 weeks (1 dev)** | **~4-5 weeks (3 devs)** |

### Task Count by User Story
- **Setup & Foundational**: 18 tasks (37%)
- **User Story 1 (P1)**: 13 tasks (27%)
- **User Story 2 (P2)**: 6 tasks (12%)
- **User Story 3 (P3)**: 7 tasks (14%)
- **Polish & Integration**: 6 tasks (12%)

### Independent Test Criteria
- **US1**: Launch HPO, interrupt, resume; verify metrics logged, checkpoints pruned, resume successful
- **US2**: Fresh machine setup completes within 15 minutes, training runs consistently
- **US3**: Every trial directory contains valid JSON report with test metrics

### Suggested MVP Scope
**Tasks**: T001-T030 (User Story 1 only)
**Duration**: 4-5 weeks (1 developer)
**Value**: Core storage-optimized HPO functionality

---

## Notes

- **Tests are included** because the spec emphasizes testing (quickstart.md shows extensive test scenarios, constitution requires 80% coverage)
- **All tasks are immediately executable** with specific file paths and acceptance criteria
- **User stories are independently testable** with clear checkpoints after each phase
- **Parallel opportunities identified** to accelerate development with multiple developers
- **Constitution compliance verified** in final task (T047)

---

## Next Steps

1. **Review and approve** this task breakdown
2. **Assign tasks** to developers (consider parallel opportunities)
3. **Set up project tracking** (GitHub Projects, Jira, etc.)
4. **Begin Sprint 1** with Phase 1 & 2 (Setup + Foundational)
5. **Deliver MVP** (US1) in 4-5 weeks
6. **Iterate** on US2 and US3 based on feedback

For questions or clarifications, refer to:
- [spec.md](./spec.md) - Feature specification
- [plan.md](./plan.md) - Implementation plan
- [data-model.md](./data-model.md) - Entity definitions
- [contracts/](./contracts/) - API schemas
- [quickstart.md](./quickstart.md) - Usage examples
