# Implementation Plan: Storage-Optimized Training & HPO Pipeline

**Branch**: `002-storage-optimized-training` | **Date**: 2025-10-10 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-storage-optimized-training/spec.md`

## Summary

This plan outlines the implementation of a storage-optimized training and hyperparameter optimization (HPO) pipeline. The core requirement is to enable large-scale HPO (1000+ trials) with large models (1-10GB) on limited storage, without losing critical metrics or the ability to resume. The technical approach involves implementing an intelligent checkpoint retention policy, proactive artifact pruning, a portable containerized development environment, and robust fault tolerance mechanisms like metrics buffering and atomic checkpoint writes. The system will support a dual-agent architecture for criteria matching and evidence binding, with all experiments tracked in a local MLflow database.

**Sequential Dependency**: This feature (002) provides the foundational HPO infrastructure. Feature 001 (threshold tuning) will extend this by adding per-criterion decision thresholds to the search space after this implementation is complete.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: PyTorch 2.0+, transformers 4.35+, datasets 2.15+, Optuna 3.4+, MLflow 2.8+, Hydra 1.3+, TextAttack 0.3.8, torchcrf 1.1.0
**Storage**: Local filesystem for artifacts, SQLite for MLflow and Optuna databases
**Testing**: pytest (unit, integration, contract) with coverage ≥80%, Hypothesis (property-based)
**Target Platform**: Linux server with Docker and optional NVIDIA GPU (CUDA 11.8+)
**Project Type**: ML Experimentation Pipeline (Single Project Structure)
**Performance Goals**:
- Reduce checkpoint storage by ≥60% vs. keep-all
- Resume interrupted jobs in ≤2 minutes
- Container environment setup in ≤15 minutes on a new machine (first-time, moderate network)
- Checkpoint save overhead ≤30s per epoch
- Storage monitoring thread CPU usage <1%
 - Metrics logging latency p95 ≤ 100ms per log event (in-memory buffering only; disk I/O and network transmission excluded)
 - Pruning under storage pressure adds ≤10s per save on average (non-blocking metric logging)

**Constraints**:
- HPO trials must run sequentially (one at a time)
- Proactively prune artifacts when available disk space < 10%
- Minimum checkpoint interval of 1 epoch
- Sequential execution rationale: Storage constraints (parallel trials multiply checkpoint requirements), GPU memory limits (1-10GB models cannot fit multiple instances), simplified storage monitoring

**Scale/Scope**:
- HPO workload: Up to 1000 trials per study
- Model catalog: Start with 5 validated models (mental-bert, psychbert, clinicalbert, bert-base, roberta-base), expand to 30+ later
 - Dataset: RedSM5 mental health posts (<10GB), loaded from Hugging Face Datasets with explicit `train/validation/test` splits (dataset ID: `irlab-udc/redsm5`, optional revision: `main`); no local CSV fallback
- Search space: ~10^15 combinations (model × architecture × loss × augmentation × hyperparameters)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

This plan is in full compliance with the HPO Constitution v1.1.0 (amended 2025-10-10 for per-study test evaluation in large-scale HPO).

- **I. Reproducibility-First**: Satisfied via deterministic seeding, pinned dependencies in `poetry.lock`, and version-controlled configs.
- **II. Storage-Optimized Artifact Management**: Core feature of this plan. Implemented via the `CheckpointManager` and `RetentionPolicy`. The default policy is `keep_last_n=1`, `keep_best_k=1`, with a hard cap of `keep_best_k_max=2`, and `max_total_size=10GB`. Proactive pruning is triggered at <10% disk space.
- **III. Dual-Agent Architecture**: Satisfied by the `MultiTaskModel` design, which supports both criteria matching and evidence binding heads. The project structure reflects this. **v1.1.0 Amendment**: Per-study test evaluation for large-scale HPO (1000+ trials) prevents test set overfitting; each trial evaluates on validation set during training.
- **IV. MLflow-Centric Experiment Tracking**: Satisfied by using a local MLflow database and implementing a `MetricsBuffer` for fault tolerance.
- **V. Auto-Resume Capability**: Satisfied by atomic checkpoint writes, integrity validation (SHA256 hash), and robust resume logic in the `Trainer`.
- **VI. Portable Development Environment**: Satisfied by the use of a Docker container with all dependencies managed by Poetry.
- **VII. Makefile-Driven Operations**: Satisfied by providing a `Makefile` for common tasks like `train`, `test`, and `lint`.

## Project Structure

### Documentation (this feature)

```
specs/002-storage-optimized-training/
├── plan.md              # This file
├── research.md          # Research on HPO frameworks, architectures, etc.
├── data-model.md        # Detailed entity definitions and relationships
├── quickstart.md        # Guide for setup and execution
├── contracts/           # Schemas for configs and outputs
└── tasks.md             # Detailed implementation task list
```

### Source Code (repository root)

The source code is organized to reflect the distinct components of the ML pipeline, including the dual-agent model architecture.

```
src/
├── cli/                  # Command-line interface (train.py, evaluate.py, cleanup.py)
├── data/                 # Data loading, preprocessing, and augmentation
│   ├── __init__.py
│   ├── augmentation.py   # TextAttack integration for evidence-only augmentation
│   ├── dataset.py        # PyTorch Dataset classes (binary_pairs, multi_label formats)
│   └── preprocessing.py  # RedSM5 CSV loading, train/val/test splitting
├── hpo/                  # Hyperparameter optimization logic
│   ├── __init__.py
│   ├── metrics_buffer.py # Disk-backed buffer for MLflow outages (FR-017)
│   ├── search_space.py   # Optuna search space definition
│   └── trial_executor.py # Sequential trial execution with cleanup
├── models/               # Model definitions, including dual-agent components
│   ├── __init__.py
│   ├── encoders/
│   │   └── hf_encoder.py # Hugging Face model wrapper with retry logic
│   ├── heads/
│   │   ├── __init__.py
│   │   ├── criteria_matching.py  # 5 head types × 4 pooling strategies
│   │   └── evidence_binding.py   # 5 span extraction architectures
│   ├── losses.py         # BCE, weighted BCE, focal, adaptive focal, hybrid
│   └── multi_task.py     # Core dual-agent model with coupling support
├── training/             # Training loop, checkpointing, and evaluation
│   ├── __init__.py
│   ├── checkpoint_manager.py  # Retention policy, SHA256 validation, atomic writes
│   ├── evaluator.py      # Test set evaluation (per-trial; optional study summary separate)
│   └── trainer.py        # Epoch-based training with auto-resume
└── utils/                # Shared utilities (logging, retry, etc.)
    ├── __init__.py
    ├── config.py         # Hydra config loading and schema validation
    ├── logging.py        # Dual logging (JSON + stdout), error formatters
    ├── retry.py          # Exponential backoff decorator (FR-005, FR-017)
    └── storage_monitor.py # Background thread for disk space monitoring (FR-018)

tests/
├── contract/             # Validate data contracts and schemas
│   ├── test_config_schema.py
│   ├── test_output_formats.py
│   └── test_checkpoint_metadata.py
├── integration/          # End-to-end tests for user stories
│   ├── test_full_hpo_resume.py
│   ├── test_storage_exhaustion.py
│   ├── test_checkpoint_corruption.py
│   └── test_portable_environment.py
└── unit/                 # Unit tests for individual components
    ├── test_checkpoint_manager.py
    ├── test_augmentation.py
    ├── test_heads.py
    └── test_losses.py

configs/                  # Hydra configuration files
├── hpo_study.yaml        # HPO study configuration
├── model/                # Model architecture configs
├── data/                 # Dataset configs
└── retention_policy/     # Checkpoint retention configs

docker/                   # Container environment
├── Dockerfile            # Multi-stage build with Poetry
├── docker-compose.yml    # Volume mounts, GPU access
└── .dockerignore

Makefile                  # Common operations (train, resume, evaluate, cleanup, test, lint)
pyproject.toml            # Poetry dependency specification
poetry.lock               # Exact version pins
README.md                 # Project overview and quickstart
```

### Experiments Directory Conventions

```
experiments/
├── trial_<uuid>/
│   ├── logs/
│   │   ├── training.jsonl        # structured logs (rotated)
│   │   └── stdout.log            # human-readable logs (rotated)
│   ├── checkpoints/              # atomic, hashed checkpoints
│   ├── config.yaml               # resolved config snapshot
│   └── evaluation_report.json    # per-trial report (authoritative)
└── study_<uuid>/
    └── summary_report.json       # optional study-level summary (references best trial)
```

**Structure Decision**: A single project structure is chosen for simplicity and cohesion. The `src/models/` directory is further organized to explicitly separate the shared `encoders` from the task-specific `heads` (criteria_matching, evidence_binding), directly reflecting the dual-agent architecture mandated by the constitution.

## Complexity Tracking

*No constitutional violations identified. This section is not applicable.*

---

## Phase 0: Research

Research findings are documented in [research.md](./research.md). Key decisions:

- **HPO Framework**: Optuna (TPE sampler) for sequential trial execution with conditional search spaces
- **Multi-Task Architecture**: Flexible coupling (independent vs coupled) with HPO-searchable combination methods
- **Input Formatting**: HPO-searchable (binary post-criterion pairs vs multi-label)
- **Loss Functions**: Modular library (BCE, weighted BCE, focal, adaptive focal, hybrid)
- **Data Augmentation**: TextAttack on evidence sentences only (preserves non-evidence text)
- **Checkpoint Management**: Epoch-based retention with proactive pruning at 10% disk threshold
- **Model Catalog**: Start with 5 validated models, expand to 30+ after validation
- **Threshold Tuning**: Post-training calibration on validation set (Feature 001 dependency)

---

## Phase 1: Data Model & Contracts

### Unified Data Model

The data model is documented in [data-model.md](./data-model.md). Key entities:

**TrialConfig** (merged schema from features 001 and 002):
- Model architecture: `model_id`, `criteria_head_type`, `evidence_head_type`, `task_coupling`
- Loss function: `loss_function`, `focal_gamma`, `label_smoothing`, `class_weights`
- Augmentation: `augmentation_methods`, `augmentation_prob`
- Regularization: `layer_wise_lr_decay`, `differential_lr_ratio`, `warmup_ratio`, `adversarial_epsilon`
- Training: `learning_rate`, `batch_size`, `accumulation_steps`, `epochs`, `optimizer`, `weight_decay`
- Thresholds (Feature 001): `criteria.thresholds`, `criteria.threshold_strategy`, `evidence.null_threshold`, `evidence.min_span_score`
- Retention: `keep_last_n`, `keep_best_k`, `max_checkpoint_size_gb`
- Optimization: `optimization_metric`, `seed`

**Trial**: Execution state, status, directories, best checkpoint references, metrics history, storage tracking

**Checkpoint**: File metadata, metrics snapshot, retention flags (retained, co_best, is_last_n, is_best_k), SHA256 integrity hash

**RetentionPolicy**: Retention parameters, disk space monitoring, pruning strategy

**EvaluationReport**: Per-trial JSON report with test metrics, config snapshot, and checkpoint references (generated after each trial completes); optional study-level summary may reference the best trial's report

**MetricsBuffer**: Disk-backed buffer for MLflow outages with exponential backoff replay

**LogEvent**: Structured logging (JSON + stdout) with severity, component, trial context

### Contracts

Schemas are defined in `contracts/`:

- `config_schema.yaml`: TrialConfig validation schema (merged 001 + 002)
- `config_schema_extended.yaml`: Full HPO search space definition
- `trial_output_schema.json`: EvaluationReport JSON schema
- `checkpoint_metadata.json`: Checkpoint file metadata schema

---

## Phase 2: Implementation Phases

### Phase 2.1: Setup & Infrastructure (Tasks T001-T008)

**Deliverables**:
- Project structure initialized
- Poetry dependency management configured
- Makefile with common operations (train, resume, evaluate, cleanup, test, lint, format)
- Hydra configuration framework
- Exponential backoff retry utility
- Dual logging system (JSON + stdout)
- Storage monitoring background thread
- Contract validation test framework

**Key Components**:
- `Makefile`: Self-documenting targets using Docker Compose for containerized execution
- `src/utils/retry.py`: `@exponential_backoff` decorator (max_attempts=5, delays [1s, 2s, 4s, 8s, 16s])
- `src/utils/logging.py`: `LogEvent` dataclass, dual logger, `format_storage_exhaustion_error()` with detailed context
- `src/utils/storage_monitor.py`: Background thread checking disk usage every 60 seconds, sets `STORAGE_CRITICAL` flag at <10%

### Phase 2.2: Foundational Components (Tasks T009-T018)

**Deliverables**:
- Data loading and preprocessing (RedSM5 from Hugging Face Hub with local cache)
- TextAttack data augmentation (evidence-only)
- Hugging Face model encoder wrapper (with retry logic)
- Criteria matching heads (5 types × 4 pooling strategies)
- Evidence binding heads (5 span extraction architectures)
- Multi-task model architecture (independent and coupled modes)
- Loss functions (5 base + hybrid combinations)
- Checkpoint manager with SHA256 integrity validation
- MLflow metrics buffering with exponential backoff replay
- Optuna search space definition

-**Key Components**:
- `src/data/preprocessing.py`: Load from Hugging Face datasets with explicit `train/validation/test` splits; no local CSV fallback; deterministic loading aligned to constitution
- `src/training/checkpoint_manager.py`: Atomic writes (temp → rename), SHA256 hash validation, retention policy enforcement, proactive pruning
- `src/hpo/metrics_buffer.py`: JSONL disk buffer, automatic replay with backoff, warn at 100MB (no hard limit)

### Phase 2.3: User Story 1 - Storage-Optimized Training/HPO with Resume (Tasks T019-T030)

**Goal**: Enable ML engineers to run long-running HPO without exhausting storage, with automatic resume capability.

**Deliverables**:
- Training loop with epoch-based checkpointing
- Sequential trial executor (Optuna integration)
- Resume from latest checkpoint with integrity validation
- Proactive retention pruning at <10% disk space
- CLI entry point for training/HPO
- Integration tests (full HPO with resume, storage exhaustion, checkpoint corruption)
- Property-based tests (retention invariants)
- Performance tests (checkpoint save overhead, storage monitor CPU)

**Key Components**:
- `src/training/trainer.py`: Epoch-based training, deterministic seeding, checks `STORAGE_CRITICAL` flag before saves
- `src/hpo/trial_executor.py`: Sequential execution (`n_jobs=1`), trial directory creation, failure handling
- `src/cli/train.py`: Modes (single, hpo, dry-run), `--resume` flag, progress bars (tqdm)

### Phase 2.4: User Story 2 - Portable Environment (Tasks T031-T036)

**Goal**: Enable ML engineers to run training consistently across different machines using containerized environment.

**Deliverables**:
- Dockerfile with Poetry (multi-stage build)
- Docker Compose configuration (volume mounts, GPU access)
- Integration tests (fresh machine setup, cross-machine consistency)
- Container setup documentation
- Makefile targets for container operations

**Key Components**:
- `docker/Dockerfile`: Base `python:3.10-slim`, install Poetry 1.7.0, `poetry install --no-dev`, CUDA 11.8+ support
- `docker/docker-compose.yml`: Mount `Data/` (read-only), `experiments/`, `~/.cache/huggingface/`, GPU access (`--gpus all`)
- Makefile: `make build`, `make shell`, `make train-container`

### Phase 2.5: User Story 3 - Per-Study Test Evaluation & JSON Reports (Tasks T037-T042)

**Goal**: Enable researchers to evaluate the best model from the entire HPO study on the test set and generate a study-level machine-readable JSON report.

**Clarification**: Evaluation is **per-study** (hybrid approach) - each trial evaluates on validation set during training; test set evaluation runs once after all trials complete, evaluating only the best model from the entire study. Co-best checkpoints within the best trial are all evaluated and included in the study report.

**Deliverables**:
- Test set evaluator (loads best checkpoint(s) from best trial of study)
- JSON report generator (EvaluationReport schema, per-study)
- Integration into trial completion workflow
- Contract tests (schema validation)
- Integration test (per-trial evaluation)
- Report analysis documentation

**Key Components**:
- `src/training/evaluator.py`: Load best checkpoint(s) from best trial of study, evaluate on test set (HF Hub split="test"), compute criteria matching + evidence binding metrics
- `src/cli/evaluate.py`: CLI for manual evaluation : `python -m src.cli.evaluate --study-id <uuid>`
- EvaluationReport: Includes `study_id, best_trial_id`, `test_metrics`, `config`, `checkpoint_references` (co-best within the best trial)

### Phase 2.6: Polish & Integration (Tasks T043-T047)

**Deliverables**:
- End-to-end integration test (all user stories)
- Test coverage ≥80% for core modules
- Code quality (ruff, black, mypy)
- Complete README
- Final constitution compliance check

---

## Dependency Management Strategy

**Approach**: Poetry + exported requirements.txt for Docker

**Rationale**:
- Poetry provides superior dependency resolution and lock file management (constitution requirement)
- Exported requirements.txt enables faster Docker builds (layer caching)
- Supports both local development (Poetry) and containerized deployment (pip)

**Implementation**:
1. Maintain `pyproject.toml` and `poetry.lock` as source of truth
2. Export to `docker/requirements.txt` via `poetry export -f requirements.txt --output docker/requirements.txt --without-hashes`
3. Dockerfile uses `pip install -r requirements.txt` for faster builds
4. CI validates that requirements.txt is up-to-date with poetry.lock

**Conflict Handling**:
- On dependency resolution conflicts, builds MUST fail with actionable guidance. Resolution MUST occur by updating `pyproject.toml` and regenerating `poetry.lock`; ad-hoc version overrides are prohibited. Record conflict decisions in PR/CHANGELOG.

**Log Retention & Rotation**:
- JSONL and stdout logs are rotated daily or at 1GB, retaining the last 14 files. Rotation MUST not drop events (safe rename and continue). Paths are under `experiments/trial_<uuid>/logs/`.

**HPO Progress Observability**:
- Emit per-trial progress (trial_index, n_trials, completion_rate, best_so_far, ETA if available) to JSON logs and MLflow params/tags.

**Dockerfile Pattern**:
```dockerfile
FROM python:3.10-slim
WORKDIR /workspace
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "src.cli.train"]
```

---

## Storage Monitoring Integration

**Strategy**: Background thread with thread-safe communication

**Implementation**:
- `storage_monitor.py` runs background thread checking disk usage every 60 seconds via `shutil.disk_usage()`
- Monitors filesystem containing `experiments/` directory
- Sets global `STORAGE_CRITICAL` flag (threading.Event) when available space < 10%
- Logs: INFO every 10 minutes, WARNING at <10%, ERROR at <5%
- `checkpoint_manager.py` checks flag before each save; triggers aggressive pruning if set
- Thread lifecycle: started at training initialization, stopped at trial completion
- Metrics buffer: JSONL buffer file with fields {run_id, metric, step, value, timestamp}. Replay order is chronological by (timestamp, step). Dedupe key is (run_id, metric, step, timestamp). No hard cap, but WARNING at ≥100MB; on storage critical, attempt immediate flush before failing.

**Separation of Concerns**:
- `storage_monitor.py`: Monitoring only (no pruning logic)
- `checkpoint_manager.py`: Pruning logic (triggered by flag)

---

## Testing Strategy

**Coverage Target**: ≥80% for `src/training/`, `src/models/`, `src/data/`, `src/hpo/`

**Test Types**:

1. **Unit Tests**: Individual component logic
   - `test_checkpoint_manager.py`: Retention policy, pruning scenarios, SHA256 validation
   - `test_augmentation.py`: Evidence-only augmentation correctness
   - `test_heads.py`: Forward pass shapes, activation functions
   - `test_losses.py`: Loss function gradients, numerical stability

2. **Integration Tests**: End-to-end workflows
   - `test_full_hpo_resume.py`: Launch HPO, interrupt, resume, verify metrics/checkpoints
   - `test_storage_exhaustion.py`: Simulate <10% disk, verify pruning triggered
   - `test_checkpoint_corruption.py`: Corrupt checkpoint, verify fallback to previous
   - `test_portable_environment.py`: Fresh machine setup within 15 minutes
   - `test_hpo_progress_observability.py`: HPO progress signals (rate, ETA) emitted to logs and MLflow

3. **Contract Tests**: Schema validation
   - `test_config_schema.py`: Validate TrialConfig against `contracts/config_schema.yaml`
   - `test_output_formats.py`: Validate EvaluationReport against `contracts/trial_output_schema.json`
   - `test_checkpoint_metadata.py`: Validate checkpoint metadata

4. **Property-Based Tests** (Hypothesis): Invariants
   - Checkpoint retention: Always keep ≥1 checkpoint, never delete co-best
   - Data split disjointness: No post_id in multiple splits
   - Augmentation: Preserves non-evidence text

5. **Performance Tests**: Targets
   - Checkpoint save overhead ≤30s per epoch
   - Storage monitor CPU usage <1%

**Coverage Reporting**: HTML reports via pytest-cov, CI fails if <80%

---

## Makefile Targets (Documented)

The following Makefile targets will be implemented (documented now, implemented in task T003):

```makefile
.PHONY: help train resume evaluate cleanup test lint format build shell

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

train:  ## Start new HPO study
	docker-compose run --rm trainer python -m src.cli.train --config configs/hpo_study.yaml

resume:  ## Resume interrupted HPO study
	docker-compose run --rm trainer python -m src.cli.train --config configs/hpo_study.yaml --resume

evaluate:  ## Evaluate best model on test set (requires TRIAL_ID)
	docker-compose run --rm trainer python -m src.cli.evaluate --trial-id $(TRIAL_ID)

cleanup:  ## Remove old checkpoints (dry-run by default, use EXECUTE=true to run)
	docker-compose run --rm trainer python -m src.cli.cleanup --dry-run=$(if $(EXECUTE),false,true)

test:  ## Run all tests with coverage
	docker-compose run --rm trainer pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:  ## Run linters (ruff, mypy)
	docker-compose run --rm trainer ruff check src/ tests/
	docker-compose run --rm trainer mypy src/

format:  ## Format code (black, ruff --fix)
	docker-compose run --rm trainer black src/ tests/
	docker-compose run --rm trainer ruff check --fix src/ tests/

build:  ## Build Docker image
	docker-compose build

shell:  ## Start interactive container shell
	docker-compose run --rm trainer bash
```

---

## Model Catalog (Initial Subset)

**Phase 1 (Initial Implementation)**: 5 validated models
1. `mental/mental-bert-base-uncased` - Mental health domain-specific
2. `mnaylor/psychbert-cased` - Psychology domain
3. `medicalai/ClinicalBERT` - Clinical domain
4. `google-bert/bert-base-uncased` - General NLP baseline
5. `FacebookAI/roberta-base` - Robust baseline

**Phase 2 (Expansion)**: Add 25+ models after validation
- DeBERTa variants (microsoft/deberta-v3-base, nvidia/quality-classifier-deberta)
- SpanBERT variants (optimized for span tasks)
- Longformer/BigBird (long document support)
- BioBERT, additional clinical models
- Additional mental health models (mnaylor/psychbert-finetuned-*)

**Rationale**: Start small to validate infrastructure, expand after proving storage optimization and HPO pipeline work correctly.

---

## Integration with Feature 001 (Threshold Tuning)

**Sequential Dependency**: Feature 001 extends this feature after implementation.

**Integration Points**:
1. **TrialConfig Schema**: Already includes threshold fields from Feature 001 in merged data model
2. **Search Space**: Feature 001 will add threshold parameters to `src/hpo/search_space.py`
3. **Evaluation**: Feature 001 will add post-training threshold calibration to `src/training/evaluator.py`
4. **Metrics**: Feature 001 will add per-criterion PR AUC and confusion matrices to EvaluationReport

**Threshold Calibration Approach** (Feature 001):
- **Post-training calibration** on validation set (not part of HPO search space)
- After each trial completes training, calibrate thresholds to maximize macro-F1
- Store calibrated thresholds in TrialConfig and EvaluationReport
- Rationale: Reduces HPO search space complexity, separates model training from decision boundary tuning

---

## Next Steps

1. **Review and approve** this plan
2. **Proceed to task breakdown**: Use existing `tasks.md` (47 tasks already defined)
3. **Begin implementation**: Start with Phase 2.1 (Setup & Infrastructure)
4. **Deliver MVP**: User Story 1 (storage-optimized HPO with resume) in 4-5 weeks
5. **Iterate**: Add User Stories 2 and 3, then integrate Feature 001 (threshold tuning)

---

## References

- **Spec**: [spec.md](./spec.md) - Feature specification with clarifications
- **Research**: [research.md](./research.md) - Design decisions and alternatives
- **Data Model**: [data-model.md](./data-model.md) - Entity definitions and relationships
- **Contracts**: [contracts/](./contracts/) - Schemas for configs and outputs
- **Tasks**: [tasks.md](./tasks.md) - Detailed implementation task list (47 tasks)
- **Quickstart**: [quickstart.md](./quickstart.md) - Setup and usage guide
- **Constitution**: [.specify/memory/constitution.md](../../.specify/memory/constitution.md) - Project principles

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A       | N/A        | N/A                                 |
