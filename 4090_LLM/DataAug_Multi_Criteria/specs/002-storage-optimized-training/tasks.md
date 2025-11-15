# Tasks: Storage-Optimized Training & HPO Pipeline

**Input**: Design documents from `/specs/002-storage-optimized-training/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Tests are NOT explicitly requested in the spec. Unit and integration tests will be added during implementation for critical components (≥80% coverage target from plan.md), but test tasks are not included in this initial breakdown.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, or SHARED)
- Include exact file paths in descriptions

## Path Conventions
- Single project structure: `src/`, `tests/`, `configs/` at repository root
- Runtime artifacts: `experiments/` (gitignored)
- Container setup: `.devcontainer/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure needed by all user stories

- [ ] T001 [P] [SHARED] Create directory structure per plan.md: `src/{data,models,training,hpo,evaluation,tracking,storage,auth,cli,config,utils}/`, `tests/{unit,integration,fixtures}/`, `configs/{model,data,training,retention,hpo}/`, `experiments/`, `.devcontainer/`
- [ ] T002 [P] [SHARED] Initialize Poetry project with `pyproject.toml`: Define Python 3.10+ requirement, add dependencies (PyTorch 2.x, transformers, mlflow, optuna, hydra-core, datasets, huggingface_hub, pyyaml, pytest, ruff), configure tool.ruff linting rules
- [ ] T003 [P] [SHARED] Create `.gitignore`: Exclude `experiments/`, `*.pyc`, `__pycache__/`, `.pytest_cache/`, `poetry.lock` (tracked), `.env`, HF cache
- [ ] T004 [P] [SHARED] Create `Makefile` with targets: `train`, `hpo`, `train-resume`, `hpo-resume`, `evaluate`, `cleanup` (stub implementations pointing to CLI commands)
- [ ] T005 [P] [SHARED] Create `.devcontainer/Dockerfile` multi-stage build: Base stage with PyTorch CUDA image, install Poetry, copy `pyproject.toml` and `poetry.lock`, install dependencies, set working directory
- [ ] T006 [P] [SHARED] Create `.devcontainer/devcontainer.json`: Configure VS Code dev container with GPU support, volume mounts (`Data/`, `experiments/`, HF cache), Python extensions

**Checkpoint**: Project structure and dependency management ready

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T007 [P] [SHARED] Implement deterministic seeding utility in `src/utils/seeding.py`: Function `set_seed_all(seed)` to seed Python random, NumPy, PyTorch CPU/CUDA, DataLoader workers; return worker_init_fn (FR-027, research.md:491-506)
- [ ] T008 [P] [SHARED] Implement path management utility in `src/utils/paths.py`: Class `PathManager` with methods `create_study_dir()` → (study_id, study_path), `create_trial_dir(study_path)` → (trial_id, trial_path), `get_checkpoint_path(trial_path, epoch)` (FR-006, FR-008, research.md:338-363)
- [ ] T009 [P] [SHARED] Implement structured logging in `src/utils/logging.py`: Class `DualLogger` with methods `log(level, message, extra={})` writing to both JSON file (`trial_dir/logs/training.jsonl`) and human-readable stdout with colored formatting (FR-020, data-model.md:520-553)
- [ ] T010 [P] [SHARED] Implement log sanitization in `src/tracking/sanitizer.py`: Class `LogSanitizer` with regex patterns for HF tokens, API keys, bearer tokens, passwords, emails; method `sanitize(text)` applying all patterns (FR-032, research.md:409-445)
- [ ] T011 [P] [SHARED] Create Hydra config schemas in `src/config/hydra_schemas.py`: Pydantic models for `TrialConfig`, `HPOStudyConfig`, `RetentionPolicy`, `DatasetConfig` with validation rules from data-model.md
- [ ] T012 [SHARED] Create base Hydra config in `configs/config.yaml`: Entry point config with defaults for model, data, training, retention, hpo; use Hydra composition syntax
- [ ] T013 [P] [SHARED] Create sample model configs in `configs/model/`: YAML files for `mental-bert.yaml`, `psychbert.yaml`, `clinicalbert.yaml`, `bert-base.yaml`, `roberta-base.yaml` with model_id and architecture-specific params
- [ ] T014 [P] [SHARED] Create dataset config in `configs/data/dataset.yaml`: Hugging Face dataset identifier (`irlab-udc/redsm5`), split mappings (train/validation/test), revision pinning options (FR-019, FR-026)
- [ ] T015 [P] [SHARED] Create retention policy configs in `configs/retention/`: Default policy (`default.yaml` with keep_last_n=1, keep_best_k=1, max_total_size=10GB, disk_space_threshold_percent=10.0) and aggressive policy (`aggressive.yaml`) (FR-002)
- [ ] T016 [SHARED] Implement Hugging Face dataset loader in `src/data/loaders.py`: Class `HFDatasetLoader` with method `load_dataset(dataset_id, split, revision=None)` validating splits exist, handling load failures, logging resolved revision/hash (FR-019, FR-026, FR-029)
- [ ] T017 [SHARED] Implement Hugging Face authentication validator in `src/auth/hf_auth.py`: Function `validate_hf_token()` checking `HfFolder.get_token()`, validating with lightweight API call (`whoami()`); function `poll_for_valid_token(interval_seconds=300)` for token expiration handling (FR-032, research.md:367-386)
- [ ] T018 [SHARED] Implement disk space monitor in `src/storage/monitor.py`: Class `DiskMonitor` with method `check_disk_space(path, threshold_percent)` → (is_low, available_percent) using `shutil.disk_usage()` (FR-018, research.md:447-458)
- [ ] T019 [SHARED] Implement checkpoint size estimator in `src/storage/estimator.py`: Function `estimate_checkpoint_size(model, optimizer)` → size_bytes using formula: `model.state_dict()` byte size + `optimizer.state_dict()` byte size + 1MB metadata overhead (FR-034)
- [ ] T020 [SHARED] Initialize MLflow tracking in `src/tracking/mlflow_wrapper.py`: Class `MLflowWrapper` initializing local SQLite backend at `experiments/mlflow.db`, methods `start_run(trial_config)`, `log_metrics(metrics, step)`, `log_params(config)`, `end_run()`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Run storage-optimized training/HPO with resume (Priority: P1) 🎯 MVP

**Goal**: ML engineer can run training/HPO without exhausting storage; system retains only necessary checkpoints while logging all metrics; interrupted jobs resume from latest checkpoint

**Independent Test**: Launch training with aggressive retention policy; verify metrics fully logged, only latest N + best K checkpoints retained; interrupt and restart job, verify resume from latest checkpoint without metric duplication

### Implementation for User Story 1

#### Core Training Components

- [ ] T021 [P] [US1] Implement atomic checkpoint writer in `src/training/checkpoint.py`: Function `save_checkpoint_atomic(state_dict, checkpoint_path)` writing to temp file then atomically renaming, computing SHA256 integrity hash, returning hash (FR-024, research.md:49-68)
- [ ] T022 [P] [US1] Implement checkpoint loader with validation in `src/training/checkpoint.py`: Function `load_checkpoint_validated(checkpoint_path)` loading checkpoint, validating integrity hash, raising error if corrupted, returning checkpoint dict (FR-004)
- [ ] T023 [P] [US1] Implement checkpoint metadata dataclass in `src/training/checkpoint.py`: Dataclass `CheckpointMetadata` with fields from data-model.md:205-226 (checkpoint_id, trial_id, epoch, step, model_state_dict, optimizer_state_dict, scheduler_state_dict, rng_state, config, metrics, created_at)
- [ ] T024 [US1] Implement retention policy manager in `src/training/retention.py`: Class `RetentionPolicyManager` with attributes from data-model.md:475-515; methods `evaluate_checkpoint(checkpoint, metrics)` marking as is_last_n/is_best_k/co_best, `should_prune(checkpoint)`, `get_prunable_checkpoints()` → list of checkpoints eligible for pruning (FR-001, FR-002)
- [ ] T025 [US1] Implement checkpoint pruner in `src/storage/pruner.py`: Class `CheckpointPruner` with method `prune(trial_dir, retention_policy, disk_monitor)` implementing 3-step progressive pruning from research.md:388-408 (step 1: prune non-protected in current trial; step 2: reduce keep_best_k=1, keep_last_n=0 for future; step 3: preserve only single best across study) (FR-009, FR-028)
- [ ] T026 [US1] Implement model architecture loader in `src/models/dual_agent.py`: Class `DualAgentModel` (stub for now) with classmethod `from_pretrained(model_id, config)` loading transformer encoder from Hugging Face with retry logic (exponential backoff: 1s, 2s, 4s, 8s, 16s, max 5 attempts), checking local cache first (FR-005, research.md:49-68)
- [ ] T027 [US1] Implement core trainer in `src/training/trainer.py`: Class `Trainer` with attributes (model, optimizer, scheduler, retention_policy, mlflow_wrapper, dual_logger); method `train_epoch(dataloader, epoch)` running training loop, computing metrics, saving checkpoint at epoch end via atomic write, invoking pruner if limits exceeded, logging metrics to MLflow with buffering fallback (FR-001, FR-003, FR-009)
- [ ] T028 [US1] Implement resume logic in `src/training/resume.py`: Class `ResumeManager` with method `resume_from_checkpoint(trial_dir)` detecting latest valid checkpoint, loading with validation, restoring model/optimizer/scheduler/RNG states, returning resume_state (epoch, step, metrics_logged); handle zero-checkpoint scenario (start from epoch 0); ensure idempotent operations (FR-004, FR-030, FR-031)
- [ ] T029 [US1] Implement training orchestrator in `src/training/trainer.py`: Method `Trainer.run_training(config, trial_dir, resume=False)` orchestrating full training: initialize model/optimizer, load data, run epochs, handle interruptions, save final checkpoint, return best_checkpoint_ids (FR-001)

#### Metrics Tracking & Buffering

- [ ] T030 [P] [US1] Implement metrics buffer in `src/tracking/buffer.py`: Class `MetricsBuffer` with disk-backed JSONL file, methods `append(metric_data)` with file locking (fcntl), `get_size_mb()`, `replay(log_func)` with exponential backoff, `clear()` after successful replay (FR-017, research.md:507-545)
- [ ] T031 [US1] Extend MLflow wrapper in `src/tracking/mlflow_wrapper.py`: Method `log_metrics(metrics, step)` with try-except catching MLflow errors, falling back to MetricsBuffer on failure, emitting WARNING when buffer exceeds 100MB, attempting periodic replay (FR-017, data-model.md:256-281)

#### HPO Orchestration

- [ ] T032 [US1] Implement HPO search space definition in `src/hpo/search_space.py`: Function `define_search_space()` returning dict mapping TrialConfig fields to Optuna distribution objects (categorical, float/int ranges, log scale) based on research.md:522-557
- [ ] T033 [US1] Implement HPO trial executor in `src/hpo/trial.py`: Class `TrialExecutor` with method `execute_trial(trial: optuna.Trial, study_config: HPOStudyConfig)` → trial_result: (1) sample hyperparameters from search space via `trial.suggest_*`, (2) create TrialConfig, (3) create trial directory, (4) invoke Trainer.run_training, (5) return best validation metric for Optuna (FR-006, FR-021)
- [ ] T034 [US1] Implement HPO study manager in `src/hpo/study.py`: Class `HPOStudyManager` with method `run_study(study_config: HPOStudyConfig)` creating Optuna study with SQLite storage, sequential sampler (n_jobs=1), iterating trials via `study.optimize(TrialExecutor.execute_trial, n_trials)`, logging progress to MLflow and JSON logs (FR-021, FR-033)

#### CLI Interface

- [ ] T035 [US1] Implement training CLI in `src/cli/train.py`: Main function `train_cli()` using argparse for `--config`, `--mode` (single/hpo), `--resume` flag, `--trial-id`; load Hydra config, initialize logger, invoke Trainer or HPOStudyManager based on mode, handle keyboard interrupt gracefully (save checkpoint before exit)
- [ ] T036 [US1] Update Makefile targets in `Makefile`: Implement `train` target calling `python src/cli/train.py --config $(CONFIG) --mode single`, `hpo` target calling `python src/cli/train.py --config $(CONFIG) --mode hpo`, `train-resume` target calling `python src/cli/train.py --resume --trial-id $(TRIAL_ID)`, `hpo-resume` target calling `python src/cli/train.py --config $(CONFIG) --mode hpo --resume`

#### Configuration & Validation

- [ ] T037 [P] [US1] Create sample training configs in `configs/training/`: YAML files for `single_trial_example.yaml` (fixed hyperparameters for debugging), `hpo_study_example.yaml` (full search space with 100 trials)
- [ ] T038 [US1] Implement config validation utility in `src/config/hydra_schemas.py`: Method `TrialConfig.validate()` checking conditional dependencies (e.g., if task_coupling=="coupled", coupling_method must be set), validating model_id exists, ensuring keep_last_n >= 1, keep_best_k >= 1 (data-model.md:98-107)

#### Error Handling & Edge Cases

- [ ] T039 [P] [US1] Implement storage exhaustion handler in `src/storage/pruner.py`: Method `CheckpointPruner.handle_storage_exhaustion(trial_dir, disk_monitor)` detecting when pruning cannot free space, generating detailed error message per FR-014 (current usage, space needed, largest artifacts with paths/sizes, remediation commands), raising exception with sanitized message (FR-014)
- [ ] T040 [P] [US1] Implement dataset loading error handler in `src/data/loaders.py`: Extend `HFDatasetLoader.load_dataset()` to catch invalid dataset_id or missing splits, generate actionable error per FR-029 (attempted dataset id/revision, required splits, detected splits, correction instructions), log error to JSON and stdout (FR-029)
- [ ] T041 [US1] Implement HF token expiration handler in `src/auth/hf_auth.py`: Class `HFAuthMonitor` with method `monitor_and_pause_on_expiration(study_manager)` running in background thread, polling token validity every 5 minutes during HPO study, pausing study execution if token invalid, resuming automatically once valid again, logging all state transitions (FR-032)

**Checkpoint**: User Story 1 complete - training/HPO with storage optimization and resume capability fully functional

---

## Phase 4: User Story 2 - Portable environment across machines (Priority: P2)

**Goal**: ML engineer can spin up containerized environment on any machine (workstation/server/cloud) and run training/HPO consistently within 15 minutes from cold start

**Independent Test**: On fresh machine with Docker installed and moderate network (50-100 Mbps), run `docker build` and `docker run`, execute sample training job with MLflow tracking, complete setup in ≤15 minutes

### Implementation for User Story 2

#### Container Configuration

- [ ] T042 [P] [US2] Complete `.devcontainer/Dockerfile` multi-stage build: Implement full Dockerfile with base stage (PyTorch 2.x CUDA 11.8 image), builder stage (Poetry install, export requirements.txt from poetry.lock), runtime stage (copy only runtime deps, set HF_HOME env var, expose MLflow port 5000), optimize layer caching (FR-013, FR-016, research.md:305-341)
- [ ] T043 [P] [US2] Create `.devcontainer/docker-compose.yml`: Define service with GPU passthrough (`--gpus all`), volume mounts for `Data/` (read-only), `experiments/` (read-write), HF cache (`~/.cache/huggingface`), environment variables (CUDA_VISIBLE_DEVICES, HF_HOME), port mapping for MLflow UI (FR-013)
- [ ] T044 [P] [US2] Create container entrypoint script in `.devcontainer/entrypoint.sh`: Script validating HF authentication (`huggingface-cli whoami`), checking GPU availability (`nvidia-smi`), initializing MLflow DB if not exists, printing setup instructions to stdout
- [ ] T045 [P] [US2] Create Docker build script in `scripts/build_container.sh`: Bash script running `docker build` with cache optimization flags, tagging image as `mental-health-hpo:latest`, outputting build logs to `experiments/container_build.log`

#### Environment Validation

- [ ] T046 [US2] Implement environment validator in `src/cli/validate_env.py`: CLI command checking Python version (≥3.10), PyTorch version (2.x), CUDA availability, HF authentication, MLflow DB accessibility, disk space (≥50GB free), logging validation results as JSON to stdout; exit code 0 if all checks pass, 1 otherwise (SC-006)
- [ ] T047 [US2] Add environment validation to Makefile: Add `validate-env` target calling `python src/cli/validate_env.py`, prepend to `train` and `hpo` targets to fail fast if environment invalid

#### Documentation & Setup

- [ ] T048 [P] [US2] Update quickstart.md Docker instructions: Verify quickstart.md steps match implemented container setup; add troubleshooting section for common issues (GPU not detected, HF auth failure, slow image pull); include cold-start timing breakdown (FR-013, SC-006)
- [ ] T049 [P] [US2] Create container README in `.devcontainer/README.md`: Document container architecture, volume mount requirements, GPU passthrough setup for different runtimes (Docker Desktop, nvidia-docker, Kubernetes), network requirements for HF model downloads

#### Cross-Machine Reproducibility

- [ ] T050 [US2] Implement reproducibility test script in `scripts/test_reproducibility.sh`: Bash script running identical trial config with same seed on current machine, comparing output metrics JSON against baseline, asserting metric deltas ≤ 1e-4, logging comparison results (SC-007)
- [ ] T051 [US2] Extend deterministic seeding in `src/utils/seeding.py`: Add environment variable checks for `CUBLAS_WORKSPACE_CONFIG`, `PYTHONHASHSEED`, set deterministic CUDA flags (`torch.backends.cudnn.deterministic=True`, `torch.use_deterministic_algorithms(True)`), document reproducibility limitations (non-deterministic ops) in docstring (SC-007, research.md:491-506)

**Checkpoint**: User Story 2 complete - portable containerized environment operational on any supported machine

---

## Phase 5: User Story 3 - Per-study test evaluation and JSON report (Priority: P3)

**Goal**: After HPO study completes, evaluate best model from entire study on held-out test set; generate machine-readable JSON report with test metrics, config, and checkpoint references

**Independent Test**: Run HPO study with 10 trials, verify study directory contains JSON report with required fields (study_id, best_trial_id, test_metrics, config, checkpoint references), verify test metrics correspond to best model evaluation on test split

### Implementation for User Story 3

#### Model Evaluation Components

- [ ] T052 [P] [US3] Implement metric computation in `src/evaluation/metrics.py`: Functions `compute_criteria_metrics(predictions, labels)` → dict with accuracy, F1 (macro/micro), precision, recall; `compute_evidence_metrics(pred_spans, true_spans)` → dict with exact_match, token_level_F1, char_level_F1; per-criterion breakdown (data-model.md:429-458)
- [ ] T053 [P] [US3] Implement evaluator in `src/evaluation/evaluator.py`: Class `Evaluator` with method `evaluate_on_test_set(model, test_dataloader, config)` running inference on test split, computing all metrics, returning metrics dict; handle co-best checkpoints by evaluating all and aggregating results (FR-007)
- [ ] T054 [US3] Implement JSON report generator in `src/evaluation/reports.py`: Class `EvaluationReportGenerator` with method `generate_study_report(study_dir, best_trial_id, test_metrics, config, checkpoints)` → report dict conforming to schema in data-model.md:393-471, saving to `study_dir/evaluation_report.json`, validating against JSON schema (FR-008)

#### Study-Level Evaluation Orchestration

- [ ] T055 [US3] Implement best trial selector in `src/hpo/study.py`: Method `HPOStudyManager.select_best_trial(study)` querying Optuna study for best trial by optimization metric, handling ties (select all co-best), returning best_trial_id(s) and checkpoint_paths (FR-007)
- [ ] T056 [US3] Implement study-level evaluator in `src/hpo/study.py`: Method `HPOStudyManager.evaluate_study(study_dir, study_id, best_trial_id)` loading best checkpoint(s), loading test dataset, invoking Evaluator, generating report via EvaluationReportGenerator, logging report path to MLflow (FR-007, FR-008)
- [ ] T057 [US3] Extend HPO study manager in `src/hpo/study.py`: Update `run_study()` to invoke `evaluate_study()` after all trials complete, ensuring test set evaluation occurs exactly once per study (not per trial), emitting progress signals (trial N/total completed, ETA) to logs (FR-007, FR-033)

#### CLI Interface for Evaluation

- [ ] T058 [US3] Implement evaluation CLI in `src/cli/evaluate.py`: Main function `evaluate_cli()` with argparse for `--study-id` or `--trial-id`, `--checkpoint` (path or "best"), `--split` (validation/test); load config, load checkpoint, load dataset split, invoke Evaluator, print metrics table to stdout, save JSON report (contracts/README.md:279-294)
- [ ] T059 [US3] Update Makefile evaluate target in `Makefile`: Implement `evaluate` target calling `python src/cli/evaluate.py --study-id $(STUDY_ID) --checkpoint best --split test`

#### Report Validation & Schema

- [ ] T060 [P] [US3] Create JSON schema in `contracts/evaluation_report_schema.json`: Define JSON Schema Draft-07 for EvaluationReport entity from data-model.md:393-471, including required fields, type constraints, nested structures for test_metrics (contracts/README.md:296-379)
- [ ] T061 [US3] Implement schema validator in `src/evaluation/reports.py`: Function `validate_report_schema(report_dict, schema_path)` using `jsonschema` library, raising ValidationError with detailed message if invalid, integrating into EvaluationReportGenerator (data-model.md:462-470)

#### Test Set Leakage Prevention

- [ ] T062 [US3] Implement split isolation validator in `src/data/loaders.py`: Extend `HFDatasetLoader` with method `validate_split_isolation(train_split, val_split, test_split)` checking no overlap in sample IDs across splits, raising error if leakage detected, logging validation results (edge case: test set leakage prevention)
- [ ] T063 [US3] Enforce test set access restrictions in `src/training/trainer.py`: Modify `Trainer` to never load test split during training/validation; add assertion checking `split != "test"` in training dataloader creation; document in docstring that test split access is prohibited during training (FR-007)

**Checkpoint**: User Story 3 complete - per-study test evaluation and JSON reporting fully functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements affecting multiple user stories, hardening, and final validation

- [ ] T064 [P] [SHARED] Implement cleanup utility in `src/cli/cleanup.py`: CLI command with `--trial-id` (optional), `--dry-run` flag (default true); detect orphaned checkpoints not referenced by MLflow or trial metadata, compute total reclaimable space, delete if not dry-run, log actions to stdout and JSON (RR-002, contracts/README.md:135-148)
- [ ] T065 [P] [SHARED] Update Makefile cleanup target in `Makefile`: Implement `cleanup` target calling `python src/cli/cleanup.py --dry-run $(DRY_RUN)`, document usage in comments
- [ ] T066 [P] [SHARED] Add comprehensive docstrings: Review all modules in `src/`, ensure every class and public method has Google-style docstring with Args, Returns, Raises, Examples; use type hints everywhere
- [ ] T067 [P] [SHARED] Configure ruff linting in `pyproject.toml`: Enable rules for unused imports, undefined names, line length (120), docstring coverage, import sorting; add ruff check to Makefile as `lint` target
- [ ] T068 [P] [SHARED] Add type checking with mypy: Add mypy to dev dependencies, create `mypy.ini` with strict settings, add `type-check` target to Makefile, fix type errors in all modules
- [ ] T069 [SHARED] Implement preflight storage check in `src/storage/estimator.py`: Function `preflight_check(model, optimizer, retention_policy, disk_monitor)` estimating total required space for keep_last_n + keep_best_k checkpoints, comparing against available disk, aborting with actionable error if insufficient even after aggressive pruning (FR-034)
- [ ] T070 [SHARED] Integrate preflight check into Trainer: Call `preflight_check()` before training starts and before each checkpoint save in `Trainer.train_epoch()`, ensuring early failure instead of mid-training crash (FR-034)
- [ ] T071 [SHARED] Implement HPO progress tracking in `src/hpo/study.py`: Extend `run_study()` to log trial progress (N/total, completion %, best-so-far metric, ETA based on avg trial duration) to MLflow tags and JSON logs after each trial (FR-033)
- [ ] T072 [P] [SHARED] Create default retention policies in `configs/retention/`: Additional YAML files for `permissive.yaml` (keep_last_n=5, keep_best_k=10, max_total_size=50GB), `minimal.yaml` (keep_last_n=1, keep_best_k=1, max_total_size=5GB) (FR-002)
- [ ] T073 [P] [SHARED] Validate quickstart.md instructions: Execute every command in quickstart.md on fresh Docker container, verify completion within 15 minutes, fix any outdated paths or commands, update troubleshooting section with encountered issues (SC-006)
- [ ] T074 [P] [SHARED] Generate Poetry lock file: Run `poetry lock` to create `poetry.lock` with exact dependency versions, commit to git for reproducibility (FR-016, Principle I)
- [ ] T075 [P] [SHARED] Export requirements.txt for Docker: Run `poetry export -f requirements.txt --without-hashes -o requirements.txt`, verify sync with poetry.lock, update Dockerfile to use requirements.txt (FR-016)
- [ ] T076 [SHARED] End-to-end smoke test: Write `scripts/smoke_test.sh` running minimal HPO study (3 trials, 2 epochs each) with all user stories enabled (resume, container, evaluation), asserting study report exists and metrics are within expected range, timing execution (should complete in <10 minutes), logging results

**Checkpoint**: All user stories polished and hardened; project ready for production use

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - **BLOCKS all user stories**
- **User Story 1 (Phase 3)**: Depends on Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (Phase 4)**: Depends on Foundational (Phase 2) - **Requires User Story 1 (T027, T029) for training orchestration to test container**
- **User Story 3 (Phase 5)**: Depends on Foundational (Phase 2) - **Requires User Story 1 (T034) for HPO study completion**
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1 - MVP)**: Can start after Foundational - **Independent** (no dependencies on US2 or US3)
- **User Story 2 (P2)**: Can start after US1 trainer components (T027, T029) exist - **Depends on US1 for testing**
- **User Story 3 (P3)**: Can start after US1 HPO manager (T034) exists - **Depends on US1 for study completion**

**Execution Strategy**: Sequential delivery (US1 → US2 → US3) recommended due to dependencies

### Within Each User Story

#### User Story 1 Internal Dependencies
- T021-T023 (checkpoint handling) before T027 (trainer)
- T024-T025 (retention/pruning) before T027 (trainer)
- T026 (model loader) before T027 (trainer)
- T030-T031 (metrics buffering) before T027 (trainer)
- T027-T029 (core training) before T032-T034 (HPO orchestration)
- T032-T034 (HPO) before T035-T036 (CLI)
- T007-T020 (foundational) before all US1 tasks

#### User Story 2 Internal Dependencies
- T042-T045 (container setup) before T046 (validation)
- T051 (enhanced seeding) extends T007 (foundational seeding)

#### User Story 3 Internal Dependencies
- T052-T053 (metrics/evaluator) before T056 (study evaluator)
- T054 (report generator) before T056 (study evaluator)
- T055-T056 (study evaluation) before T058-T059 (CLI)
- T060-T061 (schema validation) integrated into T054 (report generator)

### Parallel Opportunities

**Phase 1 (Setup)**: All tasks T001-T006 can run in parallel [P]

**Phase 2 (Foundational)**: Many tasks can run in parallel:
- T007, T008, T009, T010 [P] (utilities)
- T011, T013, T014, T015 [P] (configs)
- T016, T017, T018, T019 [P] (data/storage utils)
- T012 depends on T011 (config schemas first)
- T020 runs last (MLflow init)

**User Story 1**: Some parallelism possible:
- T021, T022, T023 [P] (checkpoint components)
- T030 [P] (metrics buffer, independent)
- T037, T038, T039, T040 [P] (configs and error handlers)
- But T024-T025, T026, T027-T029, T031-T034 are sequential

**User Story 2**: High parallelism:
- T042, T043, T044, T045 [P] (container files)
- T048, T049, T050 [P] (documentation)

**User Story 3**: Moderate parallelism:
- T052, T053, T054 [P] (evaluation components)
- T060 [P] (schema, independent)

**Polish**: High parallelism:
- T064, T065, T066, T067, T068, T072, T073, T074, T075 [P]

---

## Parallel Example: Foundational Phase

```bash
# Launch all independent foundational utilities together:
Task: "Implement deterministic seeding utility in src/utils/seeding.py"
Task: "Implement path management utility in src/utils/paths.py"
Task: "Implement structured logging in src/utils/logging.py"
Task: "Implement log sanitization in src/tracking/sanitizer.py"

# Launch all independent config files together:
Task: "Create sample model configs in configs/model/"
Task: "Create dataset config in configs/data/dataset.yaml"
Task: "Create retention policy configs in configs/retention/"
```

---

## Parallel Example: User Story 2

```bash
# Launch all container configuration files together:
Task: "Complete .devcontainer/Dockerfile multi-stage build"
Task: "Create .devcontainer/docker-compose.yml"
Task: "Create container entrypoint script in .devcontainer/entrypoint.sh"
Task: "Create Docker build script in scripts/build_container.sh"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete **Phase 1: Setup** (T001-T006)
2. Complete **Phase 2: Foundational** (T007-T020) - **CRITICAL BLOCKER**
3. Complete **Phase 3: User Story 1** (T021-T041)
4. **STOP and VALIDATE**:
   - Run smoke test: `make hpo STUDY_CONFIG=configs/training/hpo_study_example.yaml N_TRIALS=5`
   - Verify storage optimization (checkpoints pruned, metrics intact)
   - Interrupt and resume: `make hpo-resume STUDY_ID=<uuid>`
   - Verify resume successful without metric duplication
5. Deploy/demo if ready (MVP = storage-optimized training with resume)

### Incremental Delivery

1. **Foundation** (Phase 1-2) → Project structure and core utilities ready
2. **US1 MVP** (Phase 3) → Training/HPO with storage optimization ✅ **Deployable**
3. **US2 Add-on** (Phase 4) → Portable containerized environment ✅ **Deployable**
4. **US3 Add-on** (Phase 5) → Per-study evaluation reporting ✅ **Deployable**
5. **Polish** (Phase 6) → Production hardening ✅ **Final Release**

Each phase adds value without breaking previous functionality

### Parallel Team Strategy

**Not recommended** due to sequential dependencies (US2 needs US1 trainer, US3 needs US1 HPO manager)

**Alternative**: Pipeline approach
1. Developer A: Foundation (Phase 1-2)
2. Developer A → B: US1 core training (T021-T029)
3. Developer A: US1 HPO (T032-T034) | Developer B: US2 container setup (T042-T045)
4. Developer A: US1 CLI (T035-T036) | Developer B: US2 validation (T046-T051)
5. Developer A: US3 evaluation (T052-T063) | Developer B: Polish (T064-T076)

---

## Notes

- [P] tasks = different files, no dependencies - **76 total tasks, 41 parallelizable (54%)**
- [Story] label maps task to specific user story for traceability
- **User Story 1 is MVP** - prioritize completion for earliest value delivery
- Commit after each task or logical group of parallel tasks
- Stop at any checkpoint to validate story independently
- Tests are not included in initial implementation (to be added during development based on ≥80% coverage target)
- All paths are absolute or relative to repository root
- Edge cases from spec.md integrated into error handling tasks (T039, T040, T041)
- Constitutional compliance verified at Phase 2 completion (all 7 principles satisfied)

---

## Task Count Summary

- **Setup**: 6 tasks
- **Foundational**: 14 tasks (CRITICAL BLOCKER)
- **User Story 1** (P1 - MVP): 21 tasks
- **User Story 2** (P2): 10 tasks
- **User Story 3** (P3): 12 tasks
- **Polish**: 13 tasks

**Total**: 76 tasks

**Parallelizable**: 41 tasks marked [P] (54%)

**MVP Scope**: Phase 1-2 + Phase 3 (US1) = 41 tasks for minimum viable product

---

## Independent Test Criteria

### User Story 1 (P1)
- Launch HPO with 10 trials using aggressive retention (keep_last_n=2, keep_best_k=3)
- Verify only 2 latest + 3 best checkpoints retained per trial at any time
- Verify all metrics logged to MLflow (100% presence, no gaps)
- Interrupt job at trial 5, restart with `make hpo-resume`
- Verify resume from trial 6 without re-executing completed trials
- Verify no duplicate metric entries in MLflow

### User Story 2 (P2)
- On fresh Ubuntu 22.04 machine with Docker 20.10+, moderate network (50-100 Mbps)
- Clone repository, run `cd .devcontainer && docker build -t mental-health-hpo:latest .`
- Time image build (should complete in <10 minutes with warm Docker cache)
- Run `docker-compose up` to start container
- Inside container, run `make validate-env` (should pass all checks)
- Run sample training: `make train CONFIG=configs/training/single_trial_example.yaml`
- Verify training completes, MLflow DB populated, checkpoint saved
- Total time from `git clone` to training completion should be ≤15 minutes

### User Story 3 (P3)
- Run HPO study with 10 trials: `make hpo STUDY_CONFIG=configs/training/hpo_study_example.yaml N_TRIALS=10`
- After study completes, verify `experiments/study_<uuid>/evaluation_report.json` exists
- Load JSON report, assert required fields present: study_id, best_trial_id, test_metrics, config, evaluated_checkpoints
- Verify test_metrics.criteria_matching.accuracy is float in range [0, 1]
- Verify test_metrics.evidence_binding.exact_match is float in range [0, 1]
- Verify evaluated_checkpoints array length matches number of co-best checkpoints (≥1)
- Manually evaluate same checkpoint on test set using `make evaluate`, compare metrics (should match within 1e-6)

---

**Next Steps**: Begin implementation with Phase 1 (Setup). After completing Phase 2 (Foundational), proceed with MVP (User Story 1 - Phase 3).
