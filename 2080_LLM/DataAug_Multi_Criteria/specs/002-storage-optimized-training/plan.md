# Implementation Plan: Storage-Optimized Training & HPO Pipeline

**Branch**: `002-storage-optimized-training` | **Date**: 2025-01-14 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-storage-optimized-training/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement a storage-optimized training and HPO pipeline for dual-agent architecture (criteria matching + evidence binding). The system will manage checkpoint retention intelligently, support resume capabilities, integrate with Hugging Face model hub and datasets, track experiments via remote MLflow with authentication, and provide a portable Docker-based development environment. Large-scale HPO studies (1000+ trials) will use per-study test evaluation to prevent overfitting while maintaining full reproducibility and HIPAA/GDPR compliance.

## Technical Context

**Language/Version**: Python 3.10  
**Primary Dependencies**: PyTorch 2.2.0, Transformers 4.33.0, Hydra 1.3.0, MLflow 2.8.0, Optuna 3.4.0, Datasets 2.14.0  
**Storage**: Local filesystem for checkpoints/artifacts, remote MLflow tracking server (authenticated, TLS), Hugging Face datasets (irlab-udc/redsm5)  
**Testing**: pytest with markers (unit, integration, slow), 80%+ coverage target  
**Target Platform**: Linux server with CUDA support, containerized via Docker  
**Project Type**: Single project (ML training pipeline)  
**Performance Goals**: Support 1000-trial HPO studies, handle 1-10GB models, <10GB datasets, resume within 2 minutes  
**Constraints**: <10% disk space threshold for aggressive pruning, 15-minute environment setup, HIPAA/GDPR compliance, sequential trial execution  
**Scale/Scope**: 1000+ HPO trials, 5-30 model variants, dual-agent architecture (shared encoder), per-study test evaluation

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Reproducibility-First ✅
- All experiments use deterministic seeding (Python, NumPy, PyTorch)
- Configuration managed via Hydra with version control
- Docker containers with Poetry lock files for dependency pinning
- Dataset revisions pinned and logged
- **STATUS**: COMPLIANT - FR-011, FR-015, FR-016, FR-025, FR-026, FR-027 address this

### Principle II: Storage-Optimized Artifact Management ✅
- Retention policy: keep_last_n=1, keep_best_k=1 (default), configurable
- Proactive pruning at <10% disk space threshold
- Aggressive pruning strategy for low-storage scenarios
- Metrics preserved regardless of artifact pruning
- **STATUS**: COMPLIANT - FR-001, FR-002, FR-009, FR-018, FR-028 address this

### Principle III: Dual-Agent Architecture ✅
- Criteria matching agent + evidence binding agent with shared encoder
- Agent-specific metrics: standard ML metrics + exact match/has-answer/char-level F1
- Per-trial validation evaluation, per-study test evaluation for large-scale HPO
- **STATUS**: COMPLIANT - FR-007, FR-008, Constitution Principle III align

### Principle IV: MLflow-Centric Experiment Tracking ✅
- Remote MLflow tracking server with authentication (TLS)
- Continuous metric logging with disk buffering during outages
- Automatic retry with exponential backoff
- Metadata survives artifact pruning
- **STATUS**: COMPLIANT - FR-003, FR-017, FR-037, FR-038 address this

### Principle V: Auto-Resume Capability ✅
- Resume from latest valid checkpoint
- Atomic checkpoint writes (temp file + rename)
- Integrity validation via checksum/hash
- No metric duplication on resume
- **STATUS**: COMPLIANT - FR-004, FR-024, FR-030, FR-031 address this

### Principle VI: Portable Development Environment ✅
- Docker-based containerization
- Poetry dependency management
- Accelerator access and data mount requirements
- 15-minute setup time target
- **STATUS**: COMPLIANT - FR-013, FR-016, A-004, SC-006 address this

### Principle VII: Makefile-Driven Operations ✅
- Self-documenting Makefile commands
- Automated environment setup
- Multi-step operation abstraction
- **STATUS**: COMPLIANT - Will be implemented as part of infrastructure

**GATE RESULT**: ✅ PASS - All constitutional principles are addressed by functional requirements

## Project Structure

### Documentation (this feature)

```
specs/002-storage-optimized-training/
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
├── models/
│   ├── encoders/        # Shared encoder implementations (HF transformers)
│   ├── criteria_matcher.py    # Criteria matching agent
│   ├── evidence_binder.py     # Evidence binding agent
│   └── dual_agent.py          # Combined dual-agent architecture
├── training/
│   ├── trainer.py       # Main training loop with checkpointing
│   ├── checkpoint_manager.py  # Retention policy + pruning logic
│   ├── resume_handler.py      # Resume from checkpoint with validation
│   └── metrics_buffer.py      # Disk buffering for MLflow outages
├── hpo/
│   ├── optuna_study.py  # Optuna integration for HPO
│   ├── trial_manager.py # Trial lifecycle management
│   └── study_evaluator.py     # Per-study test set evaluation
├── data/
│   ├── dataset_loader.py      # Hugging Face datasets integration
│   └── data_splits.py         # Train/val/test split management
├── tracking/
│   ├── mlflow_client.py       # Remote MLflow with authentication
│   ├── token_manager.py       # Token service integration (Vault)
│   └── metric_logger.py       # Continuous metric logging
├── utils/
│   ├── logging.py       # Structured JSON + stdout logging
│   ├── storage_monitor.py     # Disk space monitoring
│   ├── seeding.py       # Deterministic seeding utilities
│   └── sanitization.py  # Log sanitization for secrets
├── cli/
│   ├── train.py         # CLI for training jobs
│   ├── hpo.py           # CLI for HPO studies
│   ├── evaluate_study.py      # Study-level test evaluation
│   └── cleanup.py       # Artifact cleanup utilities
└── config/
    └── schemas.py       # Hydra config dataclasses

configs/                 # Hydra configuration files
├── train.yaml          # Base training config
├── hpo.yaml            # HPO study config
├── model/              # Model configurations
├── data/               # Dataset configurations
└── retention/          # Retention policy configurations

tests/
├── contract/           # Contract tests for APIs
├── integration/        # End-to-end HPO + resume tests
└── unit/              # Unit tests for core logic
    ├── test_checkpoint_manager.py
    ├── test_resume_handler.py
    ├── test_metrics_buffer.py
    └── test_storage_monitor.py

docker/
├── Dockerfile          # Multi-stage build with Poetry
├── docker-compose.yml  # Development environment
└── .devcontainer/      # VS Code dev container config

.specify/
├── scripts/bash/
│   ├── setup-plan.sh
│   └── update-agent-context.sh
└── memory/
    └── constitution.md

Makefile                # Common operations (train, hpo, clean, test)
```

**Structure Decision**: Single project structure. The codebase follows a modular organization with clear separation between models (dual-agent architecture), training infrastructure (checkpointing, resume), HPO orchestration (Optuna integration), data management (HF datasets), and experiment tracking (MLflow). The `src/dataaug_multi_both/` package root aligns with existing code and supports the specialized dual-agent use case while maintaining standard Python package conventions.

## Complexity Tracking

*Fill ONLY if Constitution Check has violations that must be justified*

No constitutional violations identified. All principles are addressed by the functional requirements.

---

## Phase 0: Research - ✅ COMPLETE

**Status**: All research tasks completed and documented in `research.md`.

**Key Decisions**:
1. Tiered retention policy with proactive disk monitoring
2. Atomic checkpoint writes with SHA256 validation
3. Remote MLflow with Vault-based token authentication
4. JSONL metric buffering with exponential backoff
5. Cache-first HF loading with retry logic
6. Multi-task learning with shared BERT encoder
7. Per-study test evaluation for large-scale HPO
8. Multi-stage Docker builds with Poetry
9. TLS + filesystem encryption + access controls for compliance
10. Token polling with pause/resume for HF token expiration

**Artifacts**:
- `/specs/002-storage-optimized-training/research.md` (20,967 chars)

---

## Phase 1: Design & Contracts - ✅ COMPLETE

**Status**: Data model, API contracts, and quickstart guide completed.

**Generated Artifacts**:
1. **data-model.md** (16,394 chars): 10 core entities with validation rules, state transitions, and relationships
   - Trial, Study, Checkpoint, RetentionPolicy, EvaluationReport
   - DatasetInfo, ModelSource, LogEvent, MetricsBuffer, DualAgentModel
   
2. **contracts/cli_contracts.md** (15,804 chars): CLI command contracts
   - 7 CLI commands: train, hpo, evaluate_study, resume, cleanup, list_studies, list_checkpoints
   - Configuration schemas: train.yaml, hpo.yaml
   - Makefile targets for common operations
   - MLflow tracking contracts (metrics, parameters, tags)

3. **quickstart.md** (11,563 chars): Getting started guide
   - Prerequisites and 5-minute quick start
   - 4 common workflows: train & resume, large-scale HPO, multi-model comparison, cleanup
   - Configuration customization examples
   - Monitoring, debugging, and troubleshooting guide
   - Best practices for reproducibility, storage, and experiment tracking

4. **Agent Context Updated**: GitHub Copilot instructions updated with project context

**Constitution Re-Check**: ✅ PASS
- All 7 constitutional principles remain compliant after design phase
- No new violations introduced
- Design artifacts align with specification requirements

---

## Phase 2: Task Breakdown - NOT STARTED

**Next Step**: Run `/speckit.tasks` command to generate `tasks.md` with implementation tasks.

**Expected Outcome**:
- Breakdown of implementation tasks aligned with project structure
- Task dependencies and execution order
- Estimated effort and priority assignments
- Integration points and testing requirements

**Command to execute**:
```bash
# Run the tasks generation workflow
# Note: This is a separate command, NOT part of /speckit.plan
```

---

## Implementation Plan Summary

**Branch**: `002-storage-optimized-training`  
**Feature**: Storage-Optimized Training & HPO Pipeline  
**Status**: Phase 0 & 1 Complete, Ready for Phase 2 (Task Breakdown)

**Deliverables**:
- ✅ Technical context defined
- ✅ Constitution check passed (all 7 principles compliant)
- ✅ Project structure defined
- ✅ Research completed (10 key technology decisions)
- ✅ Data model designed (10 core entities)
- ✅ API contracts specified (7 CLI commands + configs)
- ✅ Quickstart guide created (4 workflows + troubleshooting)
- ✅ Agent context updated (GitHub Copilot)
- ⏳ Task breakdown (pending `/speckit.tasks` command)

**Generated Files**:
1. `plan.md` (this file) - Implementation plan
2. `research.md` - Research findings and decisions
3. `data-model.md` - Entity definitions and relationships
4. `contracts/cli_contracts.md` - CLI and configuration contracts
5. `quickstart.md` - User-facing quickstart guide

**Next Actions**:
1. Review generated artifacts for completeness
2. Execute `/speckit.tasks` to generate task breakdown
3. Begin implementation following task priorities
4. Run tests and linters continuously during development

**Total Artifact Size**: ~65,000 characters across 5 documentation files
