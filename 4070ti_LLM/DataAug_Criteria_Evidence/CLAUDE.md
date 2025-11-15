# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PSY Agents NO-AUG** is a baseline implementation for clinical text analysis (Criteria and Evidence extraction from DSM-5 diagnostic posts) **without data augmentation**. This is a control repository for comparing against augmentation-enhanced models.

**Critical Principles:**

1. **STRICT Field Separation** (enforced by assertions):
   - **Criteria Task**: Uses ONLY `status` field from annotations
   - **Evidence Task**: Uses ONLY `cases` field from annotations
   - Any code violating this separation will fail with `AssertionError`

2. **NO Augmentation** (✅ CLEANED):
   - This is a baseline NO-AUG control repository
   - ✅ REMOVED: Unused augmentation code has been cleaned up
   - Dependencies nlpaug/textattack are listed but not used in training

3. **Architecture Implementation** (✅ CONSOLIDATED):
   - ✅ SINGLE implementation in `src/Project/` (484KB) - Used by standalone scripts
   - ✅ REMOVED: Duplicate `src/psy_agents_noaug/architectures/` implementation (420KB) has been cleaned up
   - This consolidation eliminates code divergence risks

**Development Environment:** Configure locally with Poetry, PyTorch, and CUDA as needed (Dev Container setup retired).

## Essential Commands

### Development Environment
```bash
# Local setup workflow
make setup              # Full setup: dependencies + pre-commit + sanity checks
make install            # Install dependencies only
poetry install          # Direct Poetry install if preferred
```

### Data Preparation
```bash
# Generate ground truth from HuggingFace (recommended)
make groundtruth
# Or: python -m psy_agents_noaug.cli make_groundtruth data=hf_redsm5

# Generate from local CSV
make groundtruth-local
```

### Training
```bash
# Train with defaults (criteria, roberta_base)
make train

# Train specific task/model
make train TASK=criteria MODEL=roberta_base

# Or use CLI directly with Hydra overrides
python -m psy_agents_noaug.cli train \
    task=criteria \
    model=roberta_base \
    training.num_epochs=20 \
    training.batch_size=32

# Train Criteria architecture standalone
python scripts/train_criteria.py

# Evaluate Criteria model
python scripts/eval_criteria.py checkpoint=outputs/checkpoints/best_checkpoint.pt
```

### Hyperparameter Optimization (HPO) 🚀 PRODUCTION READY

**TWO HPO SYSTEMS** (both use real redsm5 data):

**1. Multi-Stage HPO** (Progressive refinement):
```bash
# Single architecture (criteria, evidence, share, or joint)
make hpo-s0 HPO_TASK=criteria    # Stage 0: Sanity (8 trials)
make hpo-s1 HPO_TASK=criteria    # Stage 1: Coarse (20 trials)
make hpo-s2 HPO_TASK=criteria    # Stage 2: Fine (50 trials)
make refit HPO_TASK=criteria     # Stage 3: Refit on train+val

# Or run all stages for one architecture
make full-hpo HPO_TASK=criteria

# Run ALL architectures sequentially
make full-hpo-all
```

**2. Maximal HPO** (Single large run, 600-1200 trials):
```bash
# Single architecture
make tune-criteria-max    # 800 trials
make tune-evidence-max    # 1200 trials
make tune-share-max       # 600 trials
make tune-joint-max       # 600 trials

# Run ALL architectures sequentially
make maximal-hpo-all

# Or use wrapper script for custom settings
python scripts/run_all_hpo.py --mode maximal --parallel 4
```

See `docs/HPO_GUIDE.md` for comprehensive documentation.

### Testing and Quality
```bash
make test               # Run all tests
make test-cov           # With coverage report
make test-groundtruth   # Test strict validation rules only

make lint               # Run ruff + black --check
make format             # Format with ruff + black

make pre-commit-run     # Run all pre-commit hooks

# Single test file
poetry run pytest tests/test_groundtruth.py -v
```

### MLflow and Evaluation
```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Evaluate model
make eval CHECKPOINT=outputs/checkpoints/best_checkpoint.pt

# Export metrics to CSV
make export
```

## Architecture Overview

### Four Architectures (⚠️ DUPLICATE IMPLEMENTATIONS)

**Current State:**
- `src/Project/` (376KB) - **ACTIVE**: Used by `train_criteria.py`, `eval_criteria.py`
- `src/psy_agents_noaug/architectures/` (528KB) - **INACTIVE**: Has train/eval engines but not used by CLI

**Architectures:**

1. **Criteria** (`criteria/`): Binary classification for criterion presence
   - Model: Transformer encoder → pooled output → binary classification head
   - Dataset: `CriteriaDataset` (uses ONLY `status` field)
   - Scripts: `scripts/train_criteria.py`, `scripts/eval_criteria.py` ✅ Production-ready

2. **Evidence** (`evidence/`): Span extraction for supporting text
   - Model: Transformer encoder → span prediction head (start/end positions)
   - Dataset: `EvidenceDataset` (uses ONLY `cases` field)

3. **Share** (`share/`): Shared encoder with dual heads
   - Single transformer encoder for both tasks
   - Separate classification head and span prediction head

4. **Joint** (`joint/`): Dual encoders with fusion
   - Separate encoders for criteria and evidence tasks
   - Fusion layer before evidence head

**Interface Parity (October 2025):**
- All architectures accept `head_cfg` and `task_cfg` parameters
- Standardized output keys (`"logits"` for all)
- Backward compatible with direct parameter passing

### Data Pipeline Architecture

**STRICT Validation Flow:**
```
Raw Data → Field Map Validation → Groundtruth Generation
                                           ↓
                     +--------------------+--------------------+
                     ↓                                         ↓
            Criteria Groundtruth                     Evidence Groundtruth
            (status field ONLY)                      (cases field ONLY)
                     ↓                                         ↓
              CriteriaDataset                          EvidenceDataset
```

**Key Files:**
- `configs/data/field_map.yaml`: Defines field mappings and validation rules
- `src/psy_agents_noaug/data/groundtruth.py`: Groundtruth generation with assertions
- `src/psy_agents_noaug/data/loaders.py`: Data loading with strict validation
- `src/psy_agents_noaug/data/splits.py`: Train/val/test splitting

**Validation Guarantees:**
- `_assert_field_usage()` function fails if wrong field is accessed
- Status values normalized to binary (0/1)
- Cases parsed from JSON/list format and validated
- Tests in `tests/test_groundtruth.py` verify separation

### Training Infrastructure

**Reproducibility (NEW - 2025):**
- Enhanced seed management in `src/psy_agents_noaug/utils/reproducibility.py`
- Full determinism with `torch.use_deterministic_algorithms()`
- Hardware-optimized DataLoader settings (num_workers, pin_memory, persistent_workers)
- Mixed precision support (Float16/BFloat16 with automatic GPU detection)

**Training Scripts:**
- `scripts/train_criteria.py`: Standalone Criteria training ✅ PRODUCTION-READY
- `scripts/eval_criteria.py`: Standalone Criteria evaluation ✅ PRODUCTION-READY
- `scripts/train_best.py`: HPO integration router (routes to architecture-specific scripts)
- `scripts/run_hpo_stage.py`: Multi-stage HPO runner ✅ REAL DATA
- `scripts/tune_max.py`: Maximal HPO runner ✅ REAL DATA
- `scripts/run_all_hpo.py`: Sequential HPO wrapper for all architectures ✅ NEW

**Training Configs:**
- `configs/training/default.yaml`: Standard settings with hardware optimizations
- `configs/training/optimized.yaml`: Comprehensive annotated config for max performance

**Core Training Loop:**
- `src/psy_agents_noaug/training/train_loop.py`: `Trainer` class with:
  - Mixed precision (AMP)
  - Gradient accumulation and clipping
  - Early stopping on validation metrics
  - MLflow logging
  - Checkpoint management

**Key Optimizations (2025 Best Practices):**
```yaml
# Mixed Precision
amp:
  enabled: true
  dtype: "float16"  # Use "bfloat16" for Ampere+ GPUs

# DataLoader (2-5x faster)
num_workers: 8          # Start with 2× CPU cores per GPU
pin_memory: true        # Always true for GPU training
persistent_workers: true
prefetch_factor: 2

# Reproducibility vs Speed
deterministic: true     # Full reproducibility (slower)
cudnn_benchmark: false  # Deterministic algorithms
```

### Configuration System (Hydra)

**Composition:**
```yaml
# configs/config.yaml
defaults:
  - data: hf_redsm5        # Data source
  - model: roberta_base     # Model architecture
  - training: default       # Training config
  - task: criteria          # Task definition
  - hpo: stage0_sanity      # HPO config
```

**Override Examples:**
```bash
# Single override
python -m psy_agents_noaug.cli train task=evidence

# Nested override
python -m psy_agents_noaug.cli train training.batch_size=32 training.optimizer.lr=3e-5

# Multiple models (multirun)
python -m psy_agents_noaug.cli train -m model=bert_base,roberta_base,deberta_v3_base
```

**Config Groups:**
- `data/`: Data sources (hf_redsm5, local_csv) + field_map.yaml
- `model/`: Model architectures (bert_base, roberta_base, deberta_v3_base)
- `training/`: Training hyperparameters
- `task/`: Task definitions (criteria, evidence)
- `hpo/`: HPO stages (stage0_sanity, stage1_coarse, stage2_fine, stage3_refit)

### CLI Architecture

**Entry Point:** `src/psy_agents_noaug/cli.py`

**Commands:**
1. `make_groundtruth`: Generate ground truth with strict validation
2. `train`: Train model with specified config
3. `hpo`: Run HPO stage (calls Optuna)
4. `refit`: Retrain best model on train+val
5. `evaluate_best`: Evaluate checkpoint on test set
6. `export_metrics`: Export MLflow metrics to CSV

**All commands use Hydra for configuration management.**

## Critical Data Rules

### Field Separation (ENFORCED)

```python
# In src/psy_agents_noaug/data/groundtruth.py
def _assert_field_usage(field_name: str, expected_field: str, operation: str):
    """Raises AssertionError if wrong field is used."""
    assert field_name == expected_field, (
        f"STRICT VALIDATION: {operation} must use '{expected_field}' field, "
        f"but '{field_name}' was provided"
    )

# Criteria uses ONLY status
_assert_field_usage(field_name, "status", "Criteria groundtruth generation")

# Evidence uses ONLY cases
_assert_field_usage(field_name, "cases", "Evidence groundtruth generation")
```

**Never mix these fields. Tests will fail if violated.**

### Field Mapping Format

```yaml
# configs/data/field_map.yaml
annotations:
  columns:
    status:
      required: true
      used_for: ["criteria"]
      type: "string"
      normalization:
        positive_values: ["positive", "present", "true", "1", 1, true]
        negative_values: ["negative", "absent", "false", "0", 0, false]

    cases:
      required: true
      used_for: ["evidence"]
      type: "json"
      structure:
        - text: "string"
        - start_char: "int"
        - end_char: "int"
```

## Key Testing Files

- `tests/test_groundtruth.py`: **Validates STRICT field separation** (highest priority)
- `tests/test_loaders.py`: Data loading validation
- `tests/test_training_smoke.py`: Training pipeline smoke tests
- `tests/test_hpo_config.py`: HPO config validation
- `tests/test_integration.py`: End-to-end workflow tests

## MLflow Tracking

**Backend:** SQLite (`mlflow.db`)
**Artifacts:** File system (`mlruns/`)

```python
# Tracking URI resolution
tracking_uri = "sqlite:///mlflow.db"
artifact_location = "./mlruns"

# Logged metrics
- Training: loss, accuracy (per step)
- Validation: loss, accuracy, F1 (macro/micro), precision, recall (per epoch)
- System: learning rate, epoch time
- Hyperparameters: all training config
```

**View UI:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Project Structure Highlights

**Note:** Recent cleanup details live in commit history and CODEBASE_STRUCTURE_ANALYSIS.md (legacy summary docs were removed).

```
NoAug_Criteria_Evidence/
├── configs/                    # Hydra configs (YAML)
│   ├── config.yaml            # Main composition
│   ├── data/field_map.yaml    # STRICT field mappings
│   ├── model/                 # Model configs
│   ├── training/              # Training configs
│   │   ├── default.yaml       # Standard settings
│   │   └── optimized.yaml     # Max performance settings
│   ├── task/                  # Task definitions
│   └── hpo/                   # HPO stages
│
├── src/psy_agents_noaug/
│   ├── architectures/         # Four architectures
│   │   ├── criteria/          # Binary classification
│   │   ├── evidence/          # Span extraction
│   │   ├── share/             # Shared encoder
│   │   └── joint/             # Dual encoders
│   ├── data/
│   │   ├── groundtruth.py     # Groundtruth with strict validation
│   │   ├── loaders.py         # Data loading
│   │   └── splits.py          # Train/val/test splits
│   ├── training/
│   │   ├── train_loop.py      # Trainer class (AMP, early stopping)
│   │   └── evaluate.py        # Evaluator class
│   ├── utils/
│   │   ├── reproducibility.py # Enhanced seed + hardware utils
│   │   └── mlflow_utils.py    # MLflow helpers
│   └── cli.py                 # Unified CLI
│
├── src/Project/               # Architecture implementations (800KB)
│   │                          # Used by: train_criteria.py, eval_criteria.py
│   ├── Criteria/              # Simpler, standalone implementation
│   ├── Evidence/              # Binary/multi-class classification
│   ├── Joint/                 # Multi-task model
│   └── Share/                 # Shared encoder
│
├── scripts/
│   ├── train_criteria.py      # ✅ Production-ready Criteria training
│   ├── eval_criteria.py       # ✅ Production-ready Criteria evaluation
│   ├── train_best.py          # HPO integration router
│   ├── run_hpo_stage.py       # HPO runner
│   └── make_groundtruth.py    # Ground truth generation
│
├── tests/
│   ├── test_groundtruth.py    # ⚠️ CRITICAL: Tests field separation
│   └── ...
│
├── docs/
│   ├── TRAINING_GUIDE.md      # ✅ NEW: Comprehensive training guide
│   ├── TRAINING_SETUP_COMPLETE.md  # ✅ NEW: Setup summary
│   ├── DATA_PIPELINE_IMPLEMENTATION.md
│   ├── CLI_AND_MAKEFILE_GUIDE.md
│   └── ...
│
├── Makefile                   # Convenient command shortcuts
└── pyproject.toml            # Poetry dependencies
```

## Known Technical Debt

**✅ 1. Duplicate Architecture Implementations - RESOLVED**
- ✅ REMOVED: `src/psy_agents_noaug/architectures/` (420KB) - Cleaned up Oct 2025
- ✅ KEPT: `src/Project/` (484KB) - ACTIVE for standalone scripts
- **Result**: Eliminated 420KB of duplicate code, eliminated divergence risk

**✅ 2. Unused Augmentation Code - RESOLVED**
- ✅ REMOVED: `src/psy_agents_noaug/augmentation/` directory - Cleaned up Oct 2025
- ✅ REMOVED: `tests/test_augmentation_*.py` - Cleaned up Oct 2025
- Dependencies: nlpaug, textattack (still listed but not used in training)
- **Result**: Eliminated 80KB of dead code, clarified NO-AUG mission

**3. Hidden Production Scripts**
- `scripts/train_criteria.py` (416 lines) - ✅ Production-ready but no Makefile target
- `scripts/eval_criteria.py` (306 lines) - ✅ Production-ready but no Makefile target
- **Impact**: Reduced discoverability
- **Solution**: Add to Makefile or document explicitly

## Common Pitfalls

1. **Field Mixing**: Never use `status` for evidence or `cases` for criteria. Tests will fail.

2. **Config Paths**: Hydra configs are relative to `configs/`, not full paths:
   ```bash
   # ✓ Correct
   python -m psy_agents_noaug.cli train task=criteria

   # ✗ Wrong
   python -m psy_agents_noaug.cli train task=configs/task/criteria.yaml
   ```

3. **Poetry Environment**: Always use `poetry run` or activate poetry shell:
   ```bash
   poetry run python -m psy_agents_noaug.cli train task=criteria
   # Or
   poetry shell
   python -m psy_agents_noaug.cli train task=criteria
   ```

4. **Reproducibility Trade-off**: `deterministic=true` ensures exact reproducibility but is 20% slower. Set to `false` for production/inference.

5. **Checkpoint Paths**: Use absolute paths or paths relative to project root:
   ```bash
   # ✓ Correct
   make eval CHECKPOINT=outputs/checkpoints/best_checkpoint.pt

   # ✗ May fail if run from wrong directory
   make eval CHECKPOINT=best_checkpoint.pt
   ```

6. **HPO Requirements**: Stage 3 (refit) requires best config from stage 2:
   ```bash
   make hpo-s2  # Must complete first
   make refit   # Uses outputs/hpo_stage2/best_config.yaml
   ```

## Development Workflow

1. **Make Changes**: Edit code in `src/` or `configs/`
2. **Format**: `make format`
3. **Lint**: `make lint`
4. **Test**: `make test`
5. **Pre-commit**: `make pre-commit-run`
6. **Commit**: Standard git workflow

## Quick Reference

```bash
# Complete workflow from scratch
make setup                      # 1. Setup
make groundtruth                # 2. Generate data
make hpo-s0                     # 3. Sanity check
make full-hpo                   # 4. Full HPO (stages 1-3)
make eval                       # 5. Evaluate
make export                     # 6. Export results

# Development cycle
make format && make lint && make test

# View documentation
cat docs/TRAINING_GUIDE.md      # Comprehensive training guide
cat docs/CLI_AND_MAKEFILE_GUIDE.md  # CLI reference
cat docs/QUICK_START.md         # Quick start guide
```

## Recent Updates (2025)

**October 2025:**
- 🚨 **BREAKING CHANGE**: Input format changed from `[CLS] post [SEP] criterion [SEP]` to `[CLS] criterion [SEP] post [SEP]`
  - All existing checkpoints incompatible - full retraining required
  - See `MIGRATION_GUIDE_INPUT_FORMAT.md` for migration steps
  - See `INPUT_FORMAT_RATIONALE.md` for theoretical justification
  - Expected performance improvement: +1.5-3.0% AUC/F1
- ✅ Production-ready HPO system (multi-stage + maximal modes)
- ✅ Interface parity: all architectures accept `head_cfg`/`task_cfg`
- ✅ Optuna 4.5.0 compatibility (NSGAIISampler)
- ✅ PyTorch 2.x AMP API migration
- ✅ 67/69 tests passing (97.1%), 31% coverage
- ✅ Comprehensive codebase structure analysis added

**Training Infrastructure (January 2025):**
- Enhanced reproducibility with full determinism (`torch.use_deterministic_algorithms()`)
- Hardware-optimized DataLoader (2-5x faster with persistent workers)
- Mixed precision (AMP) with Float16/BFloat16 auto-detection
- Production-ready standalone scripts: `train_criteria.py`, `eval_criteria.py`

**Project Optimization:**
- Removed 3 redundant docs (-985 lines)
- Retired Docker/devcontainer configuration
- Cleaned all cache files
- **NEW**: Added `CODEBASE_STRUCTURE_ANALYSIS.md` and `CODEBASE_QUICK_SUMMARY.txt`

**Key Documentation:**
- `MIGRATION_GUIDE_INPUT_FORMAT.md` - 🚨 Input format breaking change migration
- `INPUT_FORMAT_RATIONALE.md` - Theoretical justification for criterion-first format
- `docs/HPO_GUIDE.md` - HPO system comprehensive guide
- `docs/TRAINING_GUIDE.md` - Training best practices
- `docs/CLI_AND_MAKEFILE_GUIDE.md` - CLI reference
- `CODEBASE_STRUCTURE_ANALYSIS.md` - ⭐ Detailed code inventory and consolidation roadmap
