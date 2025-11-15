# Required Updates to plan.md

**Date**: 2025-10-10  
**Status**: Action Items from Plan Review  
**Priority**: Critical - Must complete before `/speckit.tasks`

---

## Update 1: Add Retry Utility Module

**Location**: plan.md line 111 (after `metrics_buffer.py`)

**Add**:
```markdown
├── hpo/
│   ├── search_space.py         # Optuna search space definition
│   ├── trial_executor.py       # Sequential trial execution with cleanup
│   └── metrics_buffer.py       # MLflow metrics buffering for tracking backend outages
├── utils/
│   ├── logging.py              # Dual logging (JSON + stdout)
│   ├── storage_monitor.py      # Disk space monitoring + proactive pruning
│   ├── retry.py                # NEW: Exponential backoff retry decorator/utility
│   └── config.py               # Hydra config loading + schema validation
```

---

## Update 2: Add Checkpoint Integrity Validation

**Location**: plan.md line 105

**Change from**:
```markdown
├── training/
│   ├── trainer.py              # Training loop with epoch-based checkpointing
│   ├── checkpoint_manager.py   # Retention policy enforcement + pruning
│   └── evaluator.py            # Test set evaluation + JSON report generation
```

**Change to**:
```markdown
├── training/
│   ├── trainer.py              # Training loop with epoch-based checkpointing
│   ├── checkpoint_manager.py   # Retention policy enforcement + pruning + integrity validation (SHA256 hash)
│   └── evaluator.py            # Test set evaluation + JSON report generation
```

---

## Update 3: Replace requirements.txt with Poetry

**Location**: plan.md lines 140-143

**Change from**:
```markdown
docker/
├── Dockerfile
├── requirements.txt             # Pinned dependencies
└── docker-compose.yml           # Container orchestration
```

**Change to**:
```markdown
docker/
├── Dockerfile                   # Multi-stage build with Poetry
├── docker-compose.yml           # Container orchestration

# Repository root (Poetry files)
pyproject.toml                   # Poetry dependency specification
poetry.lock                      # Exact version pins (committed to git)
```

---

## Update 4: Add Makefile

**Location**: plan.md line 87 (before `src/`)

**Add**:
```markdown
# Repository root
Makefile                         # Common operations: train, resume, evaluate, cleanup, test, lint
pyproject.toml                   # Poetry dependency specification
poetry.lock                      # Exact version pins
src/
```

---

## Update 5: Add Storage Monitoring Details

**Location**: plan.md line 43 (after Constraints section)

**Add**:
```markdown
**Storage Monitoring Strategy**:
- Background thread in `storage_monitor.py` checks disk usage every 60 seconds
- Monitors filesystem containing `experiments/` directory using `shutil.disk_usage()`
- If available space < 10%, sets `STORAGE_CRITICAL` flag and emits WARNING
- `checkpoint_manager.py` checks flag before each save; triggers aggressive pruning if set
- Thread-safe communication via `threading.Event` or shared `StorageStatus` object
- Monitoring thread lifecycle: started at training init, stopped at trial completion
- Logging: INFO every 10 minutes, WARNING at <10%, ERROR at <5%
```

---

## Update 6: Replace Constitution Check Section

**Location**: plan.md lines 52-68

**Replace entire section with**:
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

✅ **V. Auto-Resume Capability**: Resume from latest checkpoint (FR-004), checkpoint integrity validation via SHA256 hash (FR-024), atomic writes (data-model.md Checkpoint lifecycle)

✅ **VI. Portable Development Environment**: Docker-based (A-004), Poetry dependencies (Technical Standards), documented mounts (docker-compose.yml)

✅ **VII. Makefile-Driven Operations**: Makefile with train/resume/evaluate/cleanup targets (see Project Structure)

**No violations** - All constitutional requirements met by design.
```

---

## Update 7: Add Testing Strategy

**Location**: plan.md line 129 (after test structure)

**Add**:
```markdown
**Testing Strategy**:
- **Unit Test Coverage**: ≥80% for `src/training/`, `src/models/`, `src/data/`, `src/hpo/`
  - Measured via pytest-cov, HTML reports generated
  - CI fails if coverage drops below 80%
- **Integration Tests**: 
  - Full trial execution (train → evaluate → report generation)
  - Checkpoint resume after interruption (validate no metric duplication)
  - Storage exhaustion scenario (trigger pruning at 10% threshold)
  - MLflow tracking backend outage (metrics buffering + exponential backoff replay)
  - Checkpoint corruption recovery (hash validation + fallback to previous checkpoint)
- **Contract Tests**: 
  - Validate `contracts/trial_output_schema.json` against actual EvaluationReport outputs
  - Validate `contracts/checkpoint_metadata.json` against saved checkpoints
  - Validate `contracts/config_schema.yaml` against TrialConfig instances
- **Property-Based Tests** (hypothesis): 
  - Checkpoint retention invariants (always keep ≥1 checkpoint, never delete co-best)
  - Data split disjointness (no post_id appears in multiple splits)
  - Augmentation preserves non-evidence text regions
  - Metrics are finite (no NaN/Inf from unstable training)
- **Performance Tests**:
  - Checkpoint save overhead ≤30s per epoch
  - Metric logging latency <1s per batch
  - Storage monitoring thread CPU usage <1%
```

---

## Update 8: Add Sequential Execution Rationale

**Location**: plan.md line 43 (after Constraints section, before Storage Monitoring Strategy)

**Add**:
```markdown
**Sequential Execution Rationale**:
- Storage constraints: Parallel trials would multiply checkpoint storage requirements (1-10GB per trial)
- GPU memory limits: Cannot fit multiple 1-10GB models on single GPU
- Simplifies storage monitoring: Single active trial means predictable disk usage patterns
- Aligns with Constitution Principle II: Storage optimization takes precedence over parallelism
- Optuna configuration: `n_jobs=1` enforces sequential execution
```

---

## Update 9: Add Hydra Configuration Directory

**Location**: plan.md line 87 (repository root structure)

**Add**:
```markdown
# Repository root
Makefile
pyproject.toml
poetry.lock

configs/                         # Hydra configuration files
├── hpo_study.yaml              # HPO study configuration (n_trials, sampler, optimization_metric)
├── model/                      # Model architecture configs (per model_id)
├── data/                       # Dataset configs (splits, augmentation)
└── retention_policy/           # Checkpoint retention configs (keep_last_n, keep_best_k)

src/
```

---

## Update 10: Expand Error Formatting Utilities

**Location**: plan.md line 112

**Change from**:
```markdown
├── utils/
│   ├── logging.py              # Dual logging (JSON + stdout)
```

**Change to**:
```markdown
├── utils/
│   ├── logging.py              # Dual logging (JSON + stdout) + detailed error formatters
│   │                           # Includes format_storage_exhaustion_error() with artifact enumeration
```

---

## Update 11: Add Contract Validation Tests

**Location**: plan.md line 120

**Change from**:
```markdown
tests/
├── contract/
│   ├── test_config_schema.py
│   └── test_output_formats.py
```

**Change to**:
```markdown
tests/
├── contract/
│   ├── test_config_schema.py       # Validate TrialConfig against contracts/config_schema.yaml
│   ├── test_output_formats.py      # Validate EvaluationReport against contracts/trial_output_schema.json
│   └── test_checkpoint_metadata.py # Validate checkpoint dicts against contracts/checkpoint_metadata.json
```

---

## Update 12: Update Technical Context Dependencies

**Location**: plan.md lines 15-23

**Change from**:
```markdown
**Primary Dependencies**:
- PyTorch 2.0+, transformers (Hugging Face), datasets (Hugging Face)
- MLflow for experiment tracking
- TextAttack for data augmentation
- torchcrf for CRF-based sequence labeling
- Optuna for HPO orchestration
- peft (Parameter-Efficient Fine-Tuning)
- lion-pytorch
- adabelief-pytorch
```

**Change to**:
```markdown
**Primary Dependencies** (managed via Poetry):
- PyTorch 2.0+, transformers (Hugging Face), datasets (Hugging Face)
- MLflow for experiment tracking
- TextAttack for data augmentation
- torchcrf for CRF-based sequence labeling
- Optuna for HPO orchestration
- peft (Parameter-Efficient Fine-Tuning)
- lion-pytorch, adabelief-pytorch (optimizer variants)
- hydra-core for configuration management
- pytest, pytest-cov, hypothesis for testing
- ruff, black, mypy for code quality
```

---

## Verification Checklist

After applying all updates, verify:

- [ ] All 7 constitution principles have compliance statements in Constitution Check
- [ ] `src/utils/retry.py` appears in project structure
- [ ] `checkpoint_manager.py` description mentions integrity validation
- [ ] `pyproject.toml` and `poetry.lock` replace `requirements.txt`
- [ ] Makefile appears in repository root structure
- [ ] Storage monitoring strategy section added to Technical Context
- [ ] Testing Strategy section added with 80% coverage target
- [ ] Sequential execution rationale documented
- [ ] Hydra `configs/` directory added to structure
- [ ] Error formatting utilities mentioned in `logging.py` description
- [ ] Contract validation tests expanded in test structure
- [ ] Dependencies list mentions Poetry and includes hydra-core

---

## Next Steps

1. Apply all 12 updates to `plan.md`
2. Review updated plan against constitution (should show ✅ for all principles)
3. Proceed to `/speckit.tasks` for task breakdown
4. During implementation, ensure:
   - `src/utils/retry.py` implements exponential backoff with delays [1s, 2s, 4s, 8s, 16s]
   - `checkpoint_manager.py` uses SHA256 for integrity hashing
   - `storage_monitor.py` runs as background thread with 60s check interval
   - Dockerfile uses Poetry for dependency installation
   - Makefile includes all required targets (train, resume, evaluate, cleanup, test, lint, format)

---

**Estimated Time to Apply Updates**: 1-2 hours  
**Blocking**: Yes - must complete before task breakdown  
**Reviewer**: User approval required after updates applied

