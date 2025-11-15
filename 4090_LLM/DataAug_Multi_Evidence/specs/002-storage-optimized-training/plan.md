# Implementation Plan: Storage-Optimized Training & HPO Pipeline

**Branch**: `002-storage-optimized-training` | **Date**: 2025-10-10 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-storage-optimized-training/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement a storage-optimized training and hyperparameter optimization (HPO) pipeline that manages checkpoints/artifacts intelligently to prevent disk exhaustion during long-running experiments. The system will support up to 1000 trials with 1-10GB models, loading from Hugging Face, tracking with MLflow, auto-resuming after interruption, and running in portable Docker containers. Each trial evaluates on validation set; the best model from the entire study is evaluated on test set once per study with results saved as JSON.

## Technical Context

**Language/Version**: Python 3.10  
**Primary Dependencies**: PyTorch, Transformers (Hugging Face), MLflow, Hydra, Optuna (via hydra-optuna-sweeper), datasets (Hugging Face)  
**Storage**: Local filesystem for checkpoints/artifacts, MLflow SQLite database for experiment tracking, disk-based metric buffering during outages  
**Testing**: pytest with markers (unit, integration, slow), 80%+ coverage for core logic  
**Target Platform**: Linux (containerized via Docker, multi-machine support)  
**Project Type**: Single project (ML training pipeline)  
**Performance Goals**: Support 1000 HPO trials sequentially; checkpoint write/read <30s for 10GB models; metric logging latency <100ms per batch  
**Constraints**: Storage-constrained environments (proactive pruning at <10% disk space); 1-10GB models; <10GB datasets; sequential trial execution; auto-resume within 2 minutes; container startup <15 minutes on moderate network  
**Scale/Scope**: 1000 trials per HPO study; 5-30 model catalog; single dataset from Hugging Face hub; dual-agent architecture (criteria matching + evidence binding)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Reproducibility-First ✅
- All experiments recorded with deterministic seeding (Python, NumPy, PyTorch, DataLoader)
- Dependency versions pinned via Poetry lock file
- Container environments with exact version specifications
- Configuration versioning via Hydra with immutable resolved configs per trial
- **Status**: COMPLIANT

### Principle II: Storage-Optimized Artifact Management ✅
- Intelligent checkpoint retention (keep_last_n, keep_best_k, max_total_size)
- Proactive pruning at <10% disk space
- Aggressive pruning policy (single best checkpoint as last resort)
- Metrics preserved regardless of artifact pruning
- **Status**: COMPLIANT (core feature requirement)

### Principle III: Dual-Agent Architecture ✅
- Two specialized agents (criteria matching + evidence binding) with shared encoder
- Simultaneous training support
- Agent-specific evaluation metrics (standard ML + evidence-specific metrics)
- Per-trial validation evaluation, per-study test evaluation for 1000+ trials
- **Status**: COMPLIANT

### Principle IV: MLflow-Centric Experiment Tracking ✅
- Local MLflow database within project directory
- Continuous metric/parameter/artifact logging
- Disk-based buffering during outages with exponential backoff retry
- Metadata survives artifact pruning
- **Status**: COMPLIANT

### Principle V: Auto-Resume Capability ✅
- Resume from latest valid checkpoint
- No metric duplication
- Checkpoint integrity validation via hash
- Atomic checkpoint writes (temp file + rename)
- Graceful fallback to earlier checkpoints
- **Status**: COMPLIANT

### Principle VI: Portable Development Environment ✅
- Docker-based containerization
- Poetry dependency management
- Cross-machine consistency
- Accelerator access and data mount requirements
- **Status**: COMPLIANT

### Principle VII: Makefile-Driven Operations ⚠️
- Common operations accessible via Makefile commands
- Self-documenting commands
- **Status**: NEEDS IMPLEMENTATION (create Makefile with standard targets)

**Gate Result**: ✅ PASS with minor implementation note (Makefile to be created in Phase 1)

## Project Structure

### Documentation (this feature)

```
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```
src/dataaug_multi_both/
├── checkpoint/
│   ├── manager.py           # Checkpoint retention policy enforcement
│   ├── pruner.py            # Storage-aware artifact pruning
│   ├── integrity.py         # Checkpoint validation and atomic writes
│   └── __init__.py
├── training/
│   ├── trainer.py           # Core training loop with resume support
│   ├── dual_agent.py        # Dual-agent model (criteria + evidence)
│   ├── metrics.py           # Agent-specific evaluation metrics
│   └── __init__.py
├── hpo/
│   ├── study.py             # HPO study orchestration
│   ├── trial.py             # Per-trial execution and evaluation
│   └── __init__.py
├── tracking/
│   ├── mlflow_client.py     # MLflow integration with buffering
│   ├── buffer.py            # Disk-based metric buffering
│   └── __init__.py
├── data/
│   ├── loader.py            # Hugging Face dataset loading
│   └── __init__.py
├── models/
│   ├── hub_loader.py        # Hugging Face model loading with retry
│   └── __init__.py
├── config/
│   ├── schemas.py           # Hydra config schemas
│   └── __init__.py
├── utils/
│   ├── seeding.py           # Deterministic seeding utilities
│   ├── storage.py           # Disk space monitoring
│   ├── logging.py           # Dual logging (JSON + stdout)
│   └── __init__.py
└── __init__.py

configs/
├── train.yaml               # Base training configuration
├── hpo.yaml                 # HPO study configuration
├── model/
│   ├── mental-bert.yaml
│   ├── psychbert.yaml
│   └── ...                  # Model-specific configs
├── retention/
│   ├── default.yaml
│   └── aggressive.yaml
└── experiment/
    └── base.yaml

tests/
├── unit/
│   ├── test_checkpoint_manager.py
│   ├── test_pruner.py
│   ├── test_buffer.py
│   ├── test_seeding.py
│   └── ...
├── integration/
│   ├── test_training_resume.py
│   ├── test_hpo_study.py
│   ├── test_storage_pruning.py
│   └── ...
└── fixtures/
    └── ...

experiments/                 # Runtime artifacts (ignored by git)
├── study_<uuid>/
│   ├── evaluation.json      # Study-level test evaluation
│   └── trials/
│       └── trial_<uuid>/
│           ├── checkpoints/
│           ├── config.yaml
│           └── metrics.jsonl
└── mlruns/                  # MLflow tracking database

Makefile                     # Common operations (train, hpo, evaluate, cleanup)
Dockerfile                   # Training container definition
.dockerignore
pyproject.toml
poetry.lock
```

**Structure Decision**: Single project structure selected. The feature is a cohesive ML training pipeline without frontend/backend separation. All code lives under `src/dataaug_multi_both/` following the existing package structure. Modular organization by functional domain (checkpoint management, training, HPO, tracking, etc.) enables independent testing and future extensibility.

## Complexity Tracking

*Fill ONLY if Constitution Check has violations that must be justified*

**No violations requiring justification.** All constitutional principles are satisfied by the feature design.

---

## Phase Completion Summary

### Phase 0: Research ✅ COMPLETE

**Output**: `research.md` and `research_addendum.md`

**Key Decisions Resolved**:
1. ✅ Checkpoint retention strategies → Metadata-indexed with configurable policies
2. ✅ MLflow metric buffering → JSONL disk buffer with exponential backoff replay
3. ✅ Hugging Face model loading → Cache-first with retry wrapper
4. ✅ Deterministic seeding → Comprehensive (Python, NumPy, PyTorch, DataLoader)
5. ✅ Docker container environment → Multi-stage with CUDA base image
6. ✅ Proactive storage monitoring → `shutil.disk_usage()` with 10% threshold
7. ✅ Checkpoint size estimation → `state_dict()` byte size with 1.5x fallback
8. ✅ Hydra + Optuna integration → `hydra-optuna-sweeper` plugin
9. ✅ Log sanitization → Regex-based with custom pattern support
10. ✅ Atomic checkpoint writes → Temp file + SHA256 validation

**All NEEDS CLARIFICATION items from Technical Context resolved.**

### Phase 1: Design & Contracts ✅ COMPLETE

**Outputs**:
- ✅ `data-model.md` - 11 core entities with relationships, state transitions, validation rules
- ✅ `contracts/` - JSON/YAML schemas for configuration and evaluation reports
  - `config_schema.yaml` - Base training configuration schema
  - `config_schema_extended.yaml` - Full HPO configuration schema
  - `trial_output_schema.json` - Trial evaluation report schema
  - `study_output_schema.json` - Study evaluation report schema
  - `checkpoint_metadata.json` - Checkpoint metadata schema
- ✅ `quickstart.md` - Getting started guide for developers
- ✅ Agent context updated via `update-agent-context.sh copilot`

**Key Entities Defined**:
1. HPOStudy - Study orchestration and tracking
2. Trial - Individual hyperparameter configuration execution
3. Checkpoint - Model state with integrity validation
4. RetentionPolicy - Storage management rules
5. MetricsBuffer - Offline metric buffering
6. ModelSource - Hugging Face model tracking
7. DataSource - Hugging Face dataset tracking
8. TrialEvaluationReport - Per-trial test results
9. StudyEvaluationReport - Study-level test results
10. LogEvent - Structured observability
11. EnvironmentProfile - Containerized environment specs

**Technology Stack Added to Agent Context**:
- Language: Python 3.10
- Frameworks: PyTorch, Transformers, MLflow, Hydra, Optuna, datasets
- Storage: Filesystem checkpoints, MLflow SQLite, disk-based buffering
- Project Type: Single project (ML training pipeline)

### Constitution Re-Check ✅ PASS

All constitutional principles remain satisfied after Phase 1 design:
- ✅ Principle I: Reproducibility-First (deterministic seeding, version pinning, immutable configs)
- ✅ Principle II: Storage-Optimized Artifact Management (retention policies, proactive pruning)
- ✅ Principle III: Dual-Agent Architecture (criteria + evidence binding, agent-specific metrics)
- ✅ Principle IV: MLflow-Centric Experiment Tracking (local DB, metric buffering, metadata preservation)
- ✅ Principle V: Auto-Resume Capability (checkpoint validation, atomic writes, graceful fallback)
- ✅ Principle VI: Portable Development Environment (Docker containers, Poetry dependencies)
- ✅ Principle VII: Makefile-Driven Operations (to be created during implementation)

### Phase 2: Tasks Generation

**Status**: NOT STARTED (per workflow, Phase 2 is handled by separate `/speckit.tasks` command)

**Next Command**: `/speckit.tasks` to generate implementation tasks from this plan

---

## Implementation Readiness Checklist

- [x] Technical context fully specified (no NEEDS CLARIFICATION items)
- [x] Constitution check passed with no violations
- [x] Research phase complete (all technical unknowns resolved)
- [x] Data model defined (11 entities with validation rules)
- [x] API contracts specified (5 JSON/YAML schemas)
- [x] Quickstart guide created
- [x] Agent context updated with new technologies
- [x] Project structure defined (directory tree documented)
- [x] Storage directory structure specified
- [x] State transition diagrams documented
- [ ] Implementation tasks generated (requires `/speckit.tasks` command)

---

## Artifacts Manifest

| Artifact | Path | Status | Size |
|----------|------|--------|------|
| Feature Spec | `specs/002-storage-optimized-training/spec.md` | ✅ Complete | 29KB |
| Implementation Plan | `specs/002-storage-optimized-training/plan.md` | ✅ Complete | 8.4KB |
| Research Report | `specs/002-storage-optimized-training/research.md` | ✅ Complete | 22KB |
| Research Addendum | `specs/002-storage-optimized-training/research_addendum.md` | ✅ Complete | 22KB |
| Data Model | `specs/002-storage-optimized-training/data-model.md` | ✅ Complete | 24KB |
| Quickstart Guide | `specs/002-storage-optimized-training/quickstart.md` | ✅ Complete | 13KB |
| Config Schema | `specs/002-storage-optimized-training/contracts/config_schema.yaml` | ✅ Complete | 6.2KB |
| Extended Config Schema | `specs/002-storage-optimized-training/contracts/config_schema_extended.yaml` | ✅ Complete | 16KB |
| Trial Output Schema | `specs/002-storage-optimized-training/contracts/trial_output_schema.json` | ✅ Complete | 5.4KB |
| Study Output Schema | `specs/002-storage-optimized-training/contracts/study_output_schema.json` | ✅ Complete | 2.1KB |
| Checkpoint Metadata Schema | `specs/002-storage-optimized-training/contracts/checkpoint_metadata.json` | ✅ Complete | 3.8KB |
| Agent Context | `.github/copilot-instructions.md` | ✅ Updated | - |

**Total Documentation**: ~149KB across 12 artifacts

---

## Summary

**Branch**: `002-storage-optimized-training`  
**Status**: Design complete, ready for task generation  
**Generated Artifacts**: 12 files (spec, plan, research, data model, contracts, quickstart, agent context)

The implementation planning workflow has successfully completed Phases 0 and 1:
- Phase 0 resolved all technical unknowns through systematic research of 10 key areas
- Phase 1 designed 11 core entities with full validation rules and API contracts
- Agent context updated with Python 3.10, PyTorch, Transformers, MLflow, Hydra, Optuna stack
- Constitutional compliance verified (all 7 principles satisfied)

**Next Action**: Run `/speckit.tasks` to generate implementation tasks from this plan.
