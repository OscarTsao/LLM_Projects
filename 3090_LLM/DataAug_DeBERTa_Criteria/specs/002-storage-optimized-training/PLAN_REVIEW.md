# Plan Review: Cross-Reference Analysis

**Feature**: Storage-Optimized Training & HPO Pipeline  
**Review Date**: 2025-10-10  
**Reviewer**: AI Assistant  
**Documents Reviewed**:
- `plan.md` (implementation plan)
- `spec.md` (feature specification with clarifications)
- `data-model.md` (entity definitions)
- `research.md` (design decisions)
- `contracts/*.{yaml,json}` (schemas)
- `.specify/memory/constitution.md` (project constitution)

---

## Executive Summary

The plan.md is **substantially complete** but has **7 critical gaps** and **3 moderate issues** that must be addressed before implementation. The primary concerns are:

1. **Missing checkpoint integrity validation** (violates Constitution Principle V)
2. **Incomplete retry/backoff utilities** (violates Constitution Principle IV)
3. **Ambiguous storage monitoring integration** (violates Constitution Principle II)
4. **Poetry vs requirements.txt conflict** (violates Constitution Technical Standards)
5. **Missing Makefile** (violates Constitution Principle VII)
6. **Incomplete constitution check** (plan references placeholder instead of actual constitution)
7. **Missing test coverage targets** (violates Constitution Code Quality standards)

**Recommendation**: Address all critical issues before proceeding to task breakdown (`/speckit.tasks`).

---

## Critical Issues (Must Fix Before Implementation)

### 1. Missing Checkpoint Integrity Validation ❌

**Constitution Violation**: Principle V (Auto-Resume Capability)

**Evidence**:
- **spec.md FR-004** (line 94): "MUST validate checkpoint integrity (via checksum/hash) before loading"
- **spec.md FR-024** (line 114): "MUST use atomic checkpoint writes"
- **spec.md clarification** (line 27): "Combination of atomic writes to prevent corruption + validation on resume as safety net"
- **data-model.md Checkpoint** (line 119): Includes `integrity_hash` field
- **plan.md**: No mention of hash computation or validation in `src/training/checkpoint_manager.py` (line 105)

**Impact**: Corrupted checkpoints could be loaded, causing training failures or silent data corruption.

**Required Changes**:

```diff
src/training/
├── trainer.py
├── checkpoint_manager.py   # ADD: compute_checkpoint_hash(), validate_checkpoint_integrity()
└── evaluator.py

tests/unit/
└── test_checkpoint_manager.py  # ADD: test_checkpoint_integrity_validation, test_corrupted_checkpoint_fallback
```

**Specific Implementation Requirements**:
- Use SHA256 for checkpoint hashing
- Store hash in checkpoint metadata AND separate `.hash` file
- On resume: validate hash before `torch.load()`
- If validation fails: try previous checkpoint, log ERROR with details
- Atomic write pattern: `torch.save()` to temp file → compute hash → rename to final path

---

### 2. Exponential Backoff Retry Utilities Missing ❌

**Constitution Violation**: Principle IV (MLflow-Centric Experiment Tracking)

**Evidence**:
- **spec.md FR-005** (line 98): "retry download with exponential backoff (delays: 1s, 2s, 4s, 8s, 16s) up to 5 attempts"
- **spec.md FR-017** (line 109): "automatically replay buffered metrics with exponential backoff retry"
- **spec.md clarification** (line 29): "Automatically replay with exponential backoff retry; keep buffer file until successful upload confirmed"
- **plan.md**: Shows `metrics_buffer.py` (line 110) but no retry utility module

**Impact**: Transient network failures will cause hard failures instead of graceful retries.

**Required Changes**:

```diff
src/utils/
├── logging.py
├── storage_monitor.py
├── config.py
+└── retry.py  # NEW: exponential_backoff_retry decorator, RetryConfig class
```

**Specific Implementation Requirements**:
- Decorator: `@exponential_backoff(max_attempts=5, base_delay=1.0, max_delay=16.0)`
- Apply to: Hugging Face model downloads, MLflow metric uploads, dataset loading
- Log retry attempts at WARNING level
- Raise original exception after max attempts exhausted

---

### 3. Storage Monitoring Integration Ambiguous ❌

**Constitution Violation**: Principle II (Storage-Optimized Artifact Management)

**Evidence**:
- **spec.md FR-018**: "monitor available disk space and trigger proactive retention pruning when available space drops below 10%"
- **plan.md**: Shows `src/utils/storage_monitor.py` (line 113) ✅
- **data-model.md RetentionPolicy** (lines 448-449): Includes `current_disk_usage_percent`, `last_pruning_timestamp`
- **Clarification answer**: Background thread for continuous monitoring

**Ambiguity**: Plan doesn't specify:
- How often does background thread check disk usage?
- How does it communicate with `checkpoint_manager.py`?
- What happens if pruning is triggered mid-checkpoint-save?

**Required Changes**:

Add to `plan.md` Technical Context:

```markdown
**Storage Monitoring Strategy**:
- Background thread in `storage_monitor.py` checks disk usage every 60 seconds
- If available space < 10%, sets global `STORAGE_CRITICAL` flag and emits WARNING
- `checkpoint_manager.py` checks flag before each save; triggers aggressive pruning if set
- Thread-safe communication via `threading.Event` or shared `StorageStatus` object
- Monitoring thread started at training initialization, stopped at trial completion
```

**Specific Implementation Requirements**:
- Use `shutil.disk_usage()` for cross-platform disk space checks
- Monitor the filesystem containing `experiments/` directory
- Pruning triggered by checkpoint_manager, not storage_monitor (separation of concerns)
- Log disk usage at INFO level every 10 minutes, WARNING when <10%, ERROR when <5%

---

### 4. Poetry vs requirements.txt Conflict ❌

**Constitution Violation**: Technical Standards (Dependency Management)

**Evidence**:
- **Constitution**: "All Python dependencies MUST be managed through Poetry with exact version pinning in poetry.lock"
- **plan.md** (line 142): Shows `docker/requirements.txt`
- **spec.md FR-016**: "pinned to exact versions in a lock file (e.g., `requirements.txt`)"

**Clarification Answer**: Use Poetry

**Required Changes**:

```diff
docker/
├── Dockerfile  # UPDATE: Use poetry install instead of pip install -r requirements.txt
-├── requirements.txt
+├── pyproject.toml  # Poetry dependency specification
+└── poetry.lock     # Exact version pins
└── docker-compose.yml
```

**Dockerfile Pattern**:
```dockerfile
FROM python:3.10-slim
RUN pip install poetry==1.7.0
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && poetry install --no-dev --no-interaction
```

---

### 5. Makefile Missing ❌

**Constitution Violation**: Principle VII (Makefile-Driven Operations)

**Evidence**:
- **Constitution Principle VII**: "All common operations MUST be accessible through simple Makefile commands"
- **plan.md**: No Makefile in project structure (lines 70-144)

**Required Changes**:

```diff
# Repository root
+Makefile  # NEW: Common operations
docker/
src/
tests/
```

**Required Makefile Targets**:
```makefile
.PHONY: help train resume evaluate cleanup test lint format

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

train:  ## Start new HPO study
	docker-compose run --rm trainer python -m src.cli.train --config configs/hpo_study.yaml

resume:  ## Resume interrupted HPO study
	docker-compose run --rm trainer python -m src.cli.train --config configs/hpo_study.yaml --resume

evaluate:  ## Evaluate best model on test set
	docker-compose run --rm trainer python -m src.cli.evaluate --trial-id $(TRIAL_ID)

cleanup:  ## Remove old checkpoints (dry-run by default)
	docker-compose run --rm trainer python -m src.cli.cleanup --dry-run

test:  ## Run all tests
	docker-compose run --rm trainer pytest tests/ -v --cov=src --cov-report=html

lint:  ## Run linters
	docker-compose run --rm trainer ruff check src/ tests/
	docker-compose run --rm trainer mypy src/

format:  ## Format code
	docker-compose run --rm trainer black src/ tests/
	docker-compose run --rm trainer ruff check --fix src/ tests/
```

---

### 6. Constitution Check References Placeholder ❌

**Evidence**:
- **plan.md** (line 56): "The project constitution is a template placeholder"
- **Actual constitution**: Now exists at `.specify/memory/constitution.md` with 7 core principles

**Required Changes**:

Replace plan.md lines 52-68 with:

```markdown
## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Status**: ✅ All constitutional principles satisfied

**Reference**: `.specify/memory/constitution.md` (Version 1.0.0)

### Principle Compliance

✅ **I. Reproducibility-First**: Deterministic seeding (FR-011), Poetry lock files (Technical Standards), Docker containers (FR-013), config versioning (FR-008)

✅ **II. Storage-Optimized Artifact Management**: Retention policies (FR-001, FR-002), proactive pruning at 10% threshold (FR-018), metrics preserved regardless of pruning (FR-003)

✅ **III. Dual-Agent Architecture**: Criteria matching + evidence binding heads (src/models/heads/), shared encoder (src/models/multi_task.py), agent-specific metrics (data-model.md EvaluationReport)

✅ **IV. MLflow-Centric Experiment Tracking**: Local MLflow database (experiments/mlflow_db/), continuous logging (FR-003), metrics buffering with exponential backoff retry (FR-017)

✅ **V. Auto-Resume Capability**: Resume from latest checkpoint (FR-004), checkpoint integrity validation via hash (FR-024), atomic writes (data-model.md Checkpoint lifecycle)

✅ **VI. Portable Development Environment**: Docker-based (A-004), Poetry dependencies (Technical Standards), documented mounts (docker-compose.yml)

✅ **VII. Makefile-Driven Operations**: Makefile with train/resume/evaluate/cleanup targets (see Project Structure)

**No violations** - All constitutional requirements met by design.
```

---

### 7. Test Coverage Targets Missing ❌

**Constitution Violation**: Code Quality Standards

**Evidence**:
- **Constitution**: "Test coverage MUST exceed 80% for core training and evaluation logic"
- **plan.md** (lines 118-129): Shows test structure but no coverage targets

**Required Changes**:

Add to plan.md after line 129:

```markdown
**Testing Strategy**:
- **Unit Test Coverage**: ≥80% for `src/training/`, `src/models/`, `src/data/`, `src/hpo/`
- **Integration Tests**: 
  - Full trial execution (train → evaluate → report generation)
  - Checkpoint resume after interruption
  - Storage exhaustion scenario (trigger pruning at 10% threshold)
  - MLflow tracking backend outage (metrics buffering + replay)
- **Contract Tests**: Validate all schemas in `contracts/` against actual outputs
- **Property-Based Tests** (hypothesis): 
  - Checkpoint retention invariants (always keep ≥1 checkpoint)
  - Data split disjointness (no post_id in multiple splits)
  - Augmentation preserves non-evidence text
- **Coverage Reporting**: HTML reports generated via pytest-cov, fail CI if <80%
```

---

## Moderate Issues (Should Fix Before Implementation)

### 8. Detailed Error Message Utilities Not Specified ⚠️

**Evidence**:
- **spec.md clarification** (line 28): Error must include "current disk usage, space needed for next checkpoint, list of largest artifacts (with sizes and paths), and actionable commands"
- **spec.md FR-014** (line 105): Repeats requirement
- **plan.md** (line 112): Shows `logging.py` for "Dual logging" but no error formatting utilities

**Recommendation**:

Expand `src/utils/logging.py` or add `src/utils/error_formatters.py`:

```python
def format_storage_exhaustion_error(
    current_usage_gb: float,
    available_gb: float,
    next_checkpoint_size_gb: float,
    trial_dir: Path
) -> str:
    """Generate actionable error message for storage exhaustion."""
    # List largest artifacts with sizes
    # Suggest cleanup commands
    # Provide retention policy adjustment guidance
```

---

### 9. Sequential Execution Rationale Missing ⚠️

**Evidence**:
- **spec.md FR-021**: "HPO trials MUST execute sequentially"
- **plan.md research.md** (line 32): "Configure `n_jobs=1`"
- **Missing**: Why sequential?

**Recommendation**:

Add to plan.md Technical Context (after line 43):

```markdown
**Sequential Execution Rationale**:
- Storage constraints: Parallel trials would multiply checkpoint storage requirements
- GPU memory limits: 1-10GB models cannot fit multiple instances on single GPU
- Simplifies storage monitoring: Single active trial means predictable disk usage patterns
- Aligns with Constitution Principle II: Storage optimization takes precedence
```

---

### 10. Data Source Fallback Strategy Unclear ⚠️

**Evidence**:
- **plan.md Summary** (line 10): "first attempts to load data from Hugging Face Hub... falling back to local RedSM5 mental health dataset CSVs"
- **spec.md FR-019**: "load all data from a single Hugging Face dataset"
- **Conflict**: Spec says HF only, plan says HF with local fallback

**Recommendation**:

Clarify in spec.md Assumptions:

```markdown
- **A-009**: Primary data source is Hugging Face datasets hub (dataset ID: TBD). If hub is unavailable, system falls back to local `Data/redsm5/*.csv` files. Both sources must produce identical train/val/test splits for reproducibility.
```

---

## Minor Issues / Enhancements

### 11. Missing Hydra Configuration Management 📝

**Evidence**:
- **Constitution Development Workflow**: "All hyperparameters MUST be managed through Hydra configuration files"
- **plan.md**: No mention of Hydra or config files in project structure

**Recommendation**:

Add to project structure:

```
configs/
├── hpo_study.yaml          # HPO study configuration
├── model/                  # Model architecture configs
├── data/                   # Dataset configs
└── retention_policy/       # Checkpoint retention configs
```

Update `src/utils/config.py` description to: "Hydra config loading + schema validation"

---

### 12. Missing Contract Validation in Tests 📝

**Evidence**:
- **plan.md** (line 119): Shows `tests/contract/` directory
- **Constitution Code Quality**: "Integration tests MUST validate... Shared schemas"

**Recommendation**:

Add to `tests/contract/test_output_formats.py`:

```python
def test_evaluation_report_schema():
    """Validate EvaluationReport JSON against contract schema."""
    # Load contracts/trial_output_schema.json
    # Generate sample report
    # Validate with jsonschema library

def test_checkpoint_metadata_schema():
    """Validate checkpoint metadata against contract."""
    # Similar validation for checkpoint_metadata.json
```

---

## Alignment Summary

| Document | Alignment Status | Issues Found |
|----------|------------------|--------------|
| **spec.md** | 🟡 Mostly Aligned | Missing domain-specific requirements (mental health, RedSM5) in FR section |
| **data-model.md** | ✅ Fully Aligned | All entities reflected in plan structure |
| **research.md** | ✅ Fully Aligned | Design decisions match plan architecture |
| **contracts/*.yaml** | ✅ Fully Aligned | Schemas match data model |
| **constitution.md** | 🔴 Partially Aligned | 7 critical gaps (see above) |

---

## Recommended Action Plan

### Phase 1: Fix Critical Issues (Before `/speckit.tasks`)

1. ✅ Create formal constitution.md (DONE)
2. Update plan.md:
   - Add `src/utils/retry.py` to project structure
   - Add checkpoint integrity validation to `checkpoint_manager.py` description
   - Add storage monitoring integration details to Technical Context
   - Replace `docker/requirements.txt` with `pyproject.toml` + `poetry.lock`
   - Add Makefile to project structure
   - Replace Constitution Check section with actual principle compliance
   - Add Testing Strategy section with 80% coverage target

### Phase 2: Address Moderate Issues (During Implementation)

3. Expand `logging.py` or add `error_formatters.py` for detailed error messages
4. Add sequential execution rationale to Technical Context
5. Clarify data source fallback strategy in spec.md

### Phase 3: Enhancements (Optional)

6. Add `configs/` directory for Hydra configuration management
7. Add contract validation tests to `tests/contract/`

---

## Conclusion

The plan.md provides a **solid foundation** for the mental health dual-task HPO pipeline, but requires **7 critical updates** to align with the constitution and spec clarifications. The most significant gaps are:

1. Missing checkpoint integrity validation (safety-critical)
2. Missing retry utilities (reliability-critical)
3. Ambiguous storage monitoring (performance-critical)
4. Poetry vs requirements.txt conflict (reproducibility-critical)

**Recommendation**: Update plan.md to address all critical issues, then proceed to `/speckit.tasks` for task breakdown.

**Estimated Effort to Fix**: 2-3 hours to update plan.md with all required changes.

