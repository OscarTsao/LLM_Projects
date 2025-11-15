# COMPREHENSIVE CODEBASE STRUCTURE ANALYSIS
# PSY Agents NO-AUG - Clinical Text Analysis (Criteria and Evidence Extraction)
# Generated: 2025-10-25

## EXECUTIVE SUMMARY

The codebase contains SIGNIFICANT DUPLICATE IMPLEMENTATIONS with overlapping architectures and utilities:
- Two complete, parallel architecture implementations (376KB vs 528KB)
- Multiple augmentation-related modules despite being a "NO-AUG" (no augmentation) baseline
- Extensive test suite (23 test files, 400+ test functions)
- Well-documented project with 23+ markdown documentation files
- Clear split between Hydra CLI-based training and standalone script-based training

## 1. DIRECTORY STRUCTURE OVERVIEW

### Root Level (8 markdown files + configs)
```
├── CLAUDE.md                    - Development guide (primary)
├── OPTIMIZATION_SUMMARY.md      - Cleanup documentation
├── CLEANUP_SUMMARY.md           - Previous cleanup work
├── BENCHMARK_GUIDE.md           - Benchmarking documentation
├── AGENTS.md                    - Agent architecture docs
├── README.md                    - Main readme
├── README_DOCKER.md             - Docker setup
├── THIRD_PARTY_LICENSES.md      - License compliance
├── configs/                     - Hydra configuration (23 YAML files)
├── .devcontainer/               - VS Code dev container
├── .github/workflows/           - CI/CD pipelines
└── .pre-commit-config.yaml      - Pre-commit hooks
```

### Source Code (src/)
```
src/
├── Project/                          [DUPLICATE COPY 1: 376KB]
│   ├── Criteria/                     - Binary classification (present/absent)
│   ├── Evidence/                     - Span extraction
│   ├── Joint/                        - Dual encoder with fusion
│   ├── Share/                        - Shared encoder architecture
│   └── utils/checkpoint.py           - Checkpoint utilities
│
└── psy_agents_noaug/                 [MAIN PACKAGE: 1.2MB]
    ├── architectures/                [DUPLICATE COPY 2: 528KB]
    │   ├── criteria/                 - Identical to Project/Criteria
    │   ├── evidence/                 - Identical to Project/Evidence
    │   ├── joint/                    - Identical to Project/Joint
    │   ├── share/                    - Identical to Project/Share
    │   └── utils/                    - Shared utilities
    │
    ├── augmentation/                 [THEORETICAL: 32KB, UNUSED]
    │   ├── pipeline.py               - AugmenterPipeline (11KB)
    │   ├── registry.py               - Augmentation registry (6.6KB)
    │   ├── tfidf_cache.py            - TF-IDF caching (2.4KB)
    │   └── __init__.py
    │
    ├── data/                         [ACTIVE: 1.8KB total]
    │   ├── groundtruth.py            - Ground truth generation (475 lines)
    │   ├── datasets.py               - Dataset classes (456 lines)
    │   ├── loaders.py                - DataLoader creation (375 lines)
    │   ├── splits.py                 - Train/val/test splitting
    │   ├── classification_loader.py  - Classification-specific loaders
    │   ├── augmentation_utils.py     - Augmentation helpers (IMPORTED but NO-AUG)
    │   └── __init__.py
    │
    ├── models/                       [UTILITY: 500+ lines]
    │   ├── criteria_head.py          - Classification head
    │   ├── evidence_head.py          - Span prediction head
    │   ├── encoders.py               - Transformer encoders (257 lines)
    │   └── __init__.py
    │
    ├── training/                     [ACTIVE: 1.5KB total]
    │   ├── train_loop.py             - Core Trainer class (534 lines)
    │   ├── evaluate.py               - Evaluator class (449 lines)
    │   ├── setup.py                  - Training setup helpers
    │   └── __init__.py
    │
    ├── hpo/                          [OPTUNA HPO: 400+ lines]
    │   ├── optuna_runner.py          - HPO optimization (352 lines)
    │   └── __init__.py
    │
    ├── utils/                        [UTILITIES: 700+ lines]
    │   ├── reproducibility.py        - Seed & hardware utils (198 lines)
    │   ├── mlflow_utils.py           - MLflow helpers (341 lines)
    │   ├── logging.py                - Logging utilities
    │   ├── logging_config.py         - Logging config
    │   ├── type_aliases.py           - Type hints
    │   └── __init__.py
    │
    ├── cli.py                        - Typer-based CLI (201 lines) [MOSTLY STUBS]
    └── __init__.py
```

### Scripts (scripts/) - 16 files
```
ACTIVELY USED (referenced in Makefile):
├── audit_security.py              - Security vulnerability scanning
├── bench_dataloader.py            - DataLoader performance benchmarking
├── generate_licenses.py           - License report generation
├── generate_sbom.py               - SBOM generation
├── run_all_hpo.py                 - Sequential HPO wrapper [NEW]
├── tune_max.py                    - Maximal HPO (800-1200 trials)
└── verify_determinism.py          - Determinism verification

STANDALONE/PARTIALLY USED:
├── train_criteria.py              - Criteria training (PRODUCTION-READY, NOT IN MAKEFILE)
├── eval_criteria.py               - Criteria evaluation (PRODUCTION-READY, NOT IN MAKEFILE)
├── train_best.py                  - HPO integration router (expects train_*.py scripts)
├── run_hpo_stage.py               - Multi-stage HPO runner
├── make_groundtruth.py            - Ground truth generation

UTILITY/SUPPORT:
├── export_metrics.py              - MLflow metrics export
├── validate_installation.py       - Installation validation
├── gpu_utilization.py             - GPU memory profiling
└── profile_augmentation.py        - Augmentation profiling (UNUSED for NO-AUG)
```

### Tests (tests/) - 23 files, 400+ test functions
```
CRITICAL:
├── test_groundtruth.py            - Field separation validation (HIGHEST PRIORITY)

ARCHITECTURE TESTS:
├── test_arch_shapes.py            - Architecture output shapes

HPO/TRAINING INTEGRATION:
├── test_hpo_integration.py        - HPO workflow validation
├── test_hpo_config.py             - HPO configuration validation
├── test_training_smoke.py         - Training pipeline smoke tests

PIPELINE TESTS:
├── test_pipeline_comprehensive.py - End-to-end pipeline tests
├── test_pipeline_extended.py      - Extended pipeline tests
├── test_pipeline_integration.py   - Integration tests
├── test_pipeline_scope.py         - Pipeline scope validation

DATA/LOADERS:
├── test_loaders.py                - DataLoader tests
├── test_integration.py            - Integration tests

AUGMENTATION TESTS (Ironic for NO-AUG):
├── test_augmentation_registry.py  - Augmentation registry (12KB)
├── test_augmentation_utils.py     - Augmentation utils (19KB)
├── test_tfidf_cache.py            - TF-IDF caching (771B)
├── test_tfidf_cache_extended.py   - Extended TF-IDF tests (11KB)

MISC:
├── test_smoke.py                  - Basic smoke tests
├── test_cli_flags.py              - CLI flag validation
├── test_head_space.py             - Head implementation space
├── test_mlflow_artifacts.py       - MLflow artifact logging
├── test_perf_contract.py          - Performance contract tests
├── test_qa_null_policy.py         - Null handling policy (840B)
├── test_seed_determinism.py       - Deterministic training
└── test_train_smoke.py            - Minimal CLI smoke (208B) [NEAR-EMPTY]
```

### Configs (configs/) - 23 YAML files
```
Task-specific:
├── task/criteria.yaml             - Criteria task config
├── task/evidence.yaml             - Evidence task config

Architecture-specific:
├── criteria/train.yaml            - Criteria training config
├── criteria/hpo.yaml              - Criteria HPO config
├── evidence/train.yaml            - Evidence training config
├── evidence/hpo.yaml              - Evidence HPO config
├── share/train.yaml               - Share architecture config
├── share/hpo.yaml
├── joint/train.yaml               - Joint architecture config
├── joint/hpo.yaml

Model-specific:
├── model/bert_base.yaml
├── model/roberta_base.yaml
├── model/deberta_v3_base.yaml

Data sources:
├── data/hf_redsm5.yaml            - HuggingFace RedSM5 dataset
├── data/local_csv.yaml            - Local CSV data
├── data/field_map.yaml            - CRITICAL: Field separation mapping

HPO stages:
├── hpo/stage0_sanity.yaml         - Sanity check (8 trials)
├── hpo/stage1_coarse.yaml         - Coarse search (20 trials)
├── hpo/stage2_fine.yaml           - Fine search (50 trials)
├── hpo/stage3_refit.yaml          - Refit on train+val

Training configs:
├── training/default.yaml          - Standard settings
├── training/optimized.yaml        - Max performance settings

Main:
└── config.yaml                    - Hydra composition file
```

### Documentation (docs/) - 13 files
```
COMPREHENSIVE:
├── TRAINING_GUIDE.md              - Complete training guide
├── TRAINING_SETUP_COMPLETE.md     - Setup summary
├── HPO_GUIDE.md                   - Hyperparameter optimization guide
├── CLI_AND_MAKEFILE_GUIDE.md      - CLI and build system reference

ARCHITECTURE/IMPLEMENTATION:
├── DATA_PIPELINE_IMPLEMENTATION.md - Data pipeline details
├── hpo/HPO_INTEGRATION_SUMMARY.md - HPO integration status
├── hpo/SUPERMAX_HPO_IMPLEMENTATION.md - Super-max HPO (5K+ trials)

SECURITY/COMPLIANCE:
├── AUGMENTATION_AUDIT.md          - Audit of augmentation code (NO-AUG verification)
├── CI_CD_SETUP.md                 - CI/CD configuration
├── TESTING.md                     - Testing strategy

OPERATIONAL:
├── QUICK_START.md                 - Quick start guide
├── TEST_REPORT_AUGMENTATION.md    - Test report for augmentation
└── README.md                      - Overview
```

---

## 2. CRITICAL FINDINGS: DUPLICATE IMPLEMENTATIONS

### Finding 1: TWO COMPLETE ARCHITECTURE DUPLICATES

**Scope:** All 4 architectures (Criteria, Evidence, Joint, Share)

**src/Project/** (376KB) vs **src/psy_agents_noaug/architectures/** (528KB):

```
CRITERIA ARCHITECTURE:
  src/Project/Criteria/models/model.py               123 lines
  src/psy_agents_noaug/architectures/criteria/models/model.py  96 lines

  src/Project/Criteria/engine/train_engine.py        499 lines
  src/psy_agents_noaug/architectures/criteria/engine/train_engine.py  499 lines (IDENTICAL)

  src/Project/Criteria/data/dataset.py
  src/psy_agents_noaug/architectures/criteria/data/dataset.py

IDENTICAL PATTERNS FOR: Evidence, Joint, Share
```

**Key Differences:**
- `src/Project/` is SIMPLER, used by standalone scripts (train_criteria.py, eval_criteria.py, etc.)
- `src/psy_agents_noaug/architectures/` has train/eval ENGINES, not used by current CLI
- Both implement identical functionality with ~30-50 line differences
- Utilities module (`checkpoint.py`, `log.py`, etc.) DUPLICATED across both

**Current Usage:**
- ✅ src/Project/ → Used by scripts/train_criteria.py, scripts/eval_criteria.py, scripts/tune_max.py
- ❌ src/psy_agents_noaug/architectures/ → NOT USED by Makefile or main CLI

**Consolidation Impact:**
- Remove duplication: -400KB
- Merge into single src/psy_agents_noaug/architectures/
- Update scripts to use psy_agents_noaug imports
- Estimated effort: 2-4 hours

---

### Finding 2: AUGMENTATION MODULES (IRONIC FOR "NO-AUG")

**Problem:** This is a "NO-AUG" (no augmentation) baseline, yet extensive augmentation code exists.

**Location & Size:**
```
src/psy_agents_noaug/augmentation/              32KB (not used for training)
├── pipeline.py                                  11KB - AugmenterPipeline class
├── registry.py                                  6.6KB - Augmentation registry
├── tfidf_cache.py                              2.4KB - TF-IDF resource caching
└── __init__.py

src/psy_agents_noaug/data/augmentation_utils.py  - Imported but not used

Dependencies (active in pyproject.toml):
├── nlpaug>=1.1.11,<2.0
├── textattack>=0.3.10,<1.0
```

**Test Coverage (Augmentation-focused despite NO-AUG):**
- test_augmentation_registry.py (12KB)
- test_augmentation_utils.py (19KB)
- test_tfidf_cache.py (771B)
- test_tfidf_cache_extended.py (11KB)

**Current Usage:**
- Imported in: train_loop.py, classification_loader.py, datasets.py
- **BUT:** Never actually used in training (augment config defaults to "none")
- CLI has augmentation flags (--aug-lib, --aug-methods) but CLI train command doesn't route

**Consolidation Impact:**
- REMOVE: augmentation/ module (-32KB)
- REMOVE: augmentation_utils.py
- REMOVE: test_augmentation_*.py files
- REMOVE: nlpaug, textattack from dependencies
- UPDATE: Remove augmentation imports from train_loop.py, classification_loader.py
- IMPACT: -100KB code + reduced dependency footprint
- Estimated effort: 1-2 hours (simple removals)

---

### Finding 3: UNUSED/UNDERUTILIZED SCRIPTS (7 of 16)

**Not Referenced in Makefile or Main CLI:**

1. **train_criteria.py** (416 lines)
   - Status: PRODUCTION-READY but NOT IN MAKEFILE
   - Used by: Could be used directly or via train_best.py router
   - Issue: train_best.py expects this but it's not integrated into workflow

2. **eval_criteria.py** (306 lines)
   - Status: PRODUCTION-READY but NOT IN MAKEFILE
   - Issue: No make eval-criteria target
   - Note: `make eval` points to psy_agents_noaug.cli evaluate_best (different path)

3. **run_hpo_stage.py** (300+ lines)
   - Status: Semi-used; train_loop.py may use it
   - Issue: Direct invocation unclear
   - Integration: Complex HPO routing

4. **make_groundtruth.py** (unclear, likely duplicates CLI command)
   - Status: Probably duplicates psy_agents_noaug.cli make_groundtruth
   - Impact: Confusing alternatives

5. **gpu_utilization.py**
   - Status: Utility script, no Makefile target
   - Usage: Unclear

6. **profile_augmentation.py**
   - Status: Profiling augmentation (NO-AUG project uses it?)
   - Usage: Unclear

7. **export_metrics.py** (tied to deprecated CLI command)
   - Status: Has Makefile target but unclear if used
   - Integration: Make export -> psy_agents_noaug.cli export_metrics

**Consolidation Impact:**
- Clarify: Which train_* scripts are production-ready?
- Deprecate: make_groundtruth.py if CLI command works
- Document: Clear execution paths for each script
- Estimated effort: 1-2 hours (documentation + cleanup)

---

### Finding 4: DUPLICATE TEST FILES (Minor but Notable)

**Near-Identical Tests:**

1. **test_train_smoke.py** (208 bytes) vs **test_training_smoke.py** (4.7KB)
   - First: Minimal CLI smoke test (just checks if module imports)
   - Second: Comprehensive training smoke tests
   - Overlap: Both test CLI/training loading, but different scopes

2. **test_tfidf_cache.py** (771B) vs **test_tfidf_cache_extended.py** (11KB)
   - First: Minimal TF-IDF cache test
   - Second: Comprehensive extended tests
   - Status: Acceptable (basic vs. extended)

**Issue:** test_train_smoke.py seems redundant

**Consolidation Impact:**
- Remove: test_train_smoke.py
- Keep: test_training_smoke.py
- Impact: Minimal (-200 bytes)
- Estimated effort: <1 hour

---

### Finding 5: OVERSIZED DOCUMENTATION (13 markdown files)

**Total Documentation:** 23 root + docs markdown files

**Potential Redundancy:**
- CLAUDE.md (development guide - primary)
- OPTIMIZATION_SUMMARY.md (cleanup summary)
- CLEANUP_SUMMARY.md (previous cleanup)
- AGENTS.md (agent architecture)
- docs/README.md (overview)
- docs/QUICK_START.md (quick start)
- docs/TRAINING_GUIDE.md (comprehensive guide)
- docs/TRAINING_SETUP_COMPLETE.md (setup summary)

**Assessment:**
- Most docs serve distinct purposes (good!)
- OPTIMIZATION_SUMMARY.md + CLEANUP_SUMMARY.md could be merged
- AGENTS.md purpose unclear (seems historical)
- docs/README.md might duplicate root README.md

**Consolidation Impact:**
- Merge: OPTIMIZATION_SUMMARY.md + CLEANUP_SUMMARY.md
- Review: AGENTS.md for relevance
- Review: docs/README.md vs root README.md
- Impact: Minimal (-500 lines docs)
- Estimated effort: <1 hour

---

## 3. UNUSED IMPORTS & DEAD CODE PATTERNS

### High-Confidence Dead Code:

1. **CLI train command** (src/psy_agents_noaug/cli.py line 25+)
   - Accepts augmentation flags: aug_lib, aug_methods, aug_p_apply, etc.
   - Implementation: Prints config but doesn't actually route training
   - Status: STUB - no actual training happens

2. **Augmentation in training** (src/psy_agents_noaug/training/train_loop.py)
   - Imports: AugmenterPipeline from augmentation
   - Status: Imported but configuration check shows aug_config is never used in practice

3. **classification_loader.py**
   - Heavy augmentation integration
   - Status: Imported by datasets.py but datasets.py is not used by main training path

4. **augmentation_utils.py**
   - Status: Imported by classification_loader.py but pure augmentation
   - Usage: Only in augmentation test files

---

## 4. CONFIGURATION ORGANIZATION

**Status:** Well-organized with Hydra

**Issues:**
- None major; 23 YAML configs are organized by category
- Good: Field mapping in data/field_map.yaml
- Good: Multiple training configs (default, optimized)
- Good: HPO stages clearly defined

---

## 5. TEST COVERAGE ANALYSIS

**Total Test Functions:** 400+

**Test Distribution:**
- Architecture/Shape tests: ~40
- HPO/Training integration: ~80
- Pipeline/Integration tests: ~100
- Data/Loader tests: ~50
- Augmentation tests: ~50 (IRONIC for NO-AUG)
- Misc/Utility: ~80

**Assessment:**
- GOOD: Comprehensive test coverage
- ISSUE: Augmentation tests (50+) irrelevant for NO-AUG
- ISSUE: test_augmentation_*.py files should be removed/skipped

**Consolidation Impact:**
- Remove augmentation test files: ~55KB
- Update: Skip/mark augmentation tests as xfail
- Estimated effort: <1 hour

---

## 6. CONFIGURATION FILE USAGE

**Usage Patterns:**
- ✅ Actively used: task/, model/, training/, data/, hpo/
- ⚠️ Architecture-specific: criteria/train.yaml, evidence/train.yaml, etc.
  - Issue: These duplicate general task/training configs in some ways
  - These allow architecture-specific overrides

**Assessment:** No major issues; good separation

---

## 7. SCRIPT INVOCATION ANALYSIS

**Via Makefile (Primary):**
```
make setup          → install, pre-commit-install, sanity-check
make groundtruth    → psy_agents_noaug.cli make_groundtruth
make train          → psy_agents_noaug.cli train
make train-evidence → psy_agents_noaug.cli train
make hpo-s*         → psy_agents_noaug.cli hpo
make refit          → psy_agents_noaug.cli refit
make eval           → psy_agents_noaug.cli evaluate_best
make export         → psy_agents_noaug.cli export_metrics
make bench          → scripts/bench_dataloader.py
make verify-determinism → scripts/verify_determinism.py
make audit          → scripts/audit_security.py
make sbom           → scripts/generate_sbom.py
make licenses       → scripts/generate_licenses.py
make tune-*-supermax → scripts/tune_max.py
make full-hpo-all   → scripts/run_all_hpo.py
```

**Direct Script Usage (Not in Makefile):**
- scripts/train_criteria.py (production-ready, not exposed)
- scripts/eval_criteria.py (production-ready, not exposed)

**Issue:** Production-ready training/eval scripts are hidden

---

## 8. DEPENDENCY ANALYSIS

**Augmentation-specific dependencies (UNUSED in NO-AUG):**
- nlpaug>=1.1.11,<2.0
- textattack>=0.3.10,<1.0

**Impact:** ~200MB disk space, slow pip install

**Recommendation:** Move to optional dependency group or remove entirely

---

## 9. KEY METRICS

| Metric | Value | Status |
|--------|-------|--------|
| Total Python files | 169 | OK |
| Duplicate code (Project/ + architectures/) | ~900KB total | HIGH ISSUE |
| Augmentation code (NO-AUG baseline) | 32KB modules + 55KB tests | MODERATE ISSUE |
| Documentation files | 23 | ACCEPTABLE |
| Test functions | 400+ | EXCELLENT |
| Config files | 23 | GOOD |
| Scripts | 16 (7 unused/unclear) | NEEDS CLEANUP |
| Source code lines | ~15,000 | OK |
| Test code lines | ~2,500 | GOOD COVERAGE |

---

## 10. CONSOLIDATION ROADMAP

### Phase 1: Remove Augmentation (1-2 hours)
1. Remove src/psy_agents_noaug/augmentation/
2. Remove src/psy_agents_noaug/data/augmentation_utils.py
3. Remove src/psy_agents_noaug/data/classification_loader.py (augmentation-heavy)
4. Remove augmentation imports from train_loop.py, datasets.py
5. Remove test_augmentation_*.py files
6. Remove nlpaug, textattack from dependencies
7. Update CLAUDE.md: clarify NO-AUG status

### Phase 2: Consolidate Architectures (2-4 hours)
1. Remove src/Project/ directory entirely
2. Move src/psy_agents_noaug/architectures/ to canonical location
3. Update scripts to import from psy_agents_noaug.architectures
4. Consolidate duplicate checkpoint.py, log.py, etc. utilities
5. Add integration tests for consolidated structure

### Phase 3: Clarify Script Exposure (1-2 hours)
1. Document: Which train_* scripts are production-ready
2. Add Makefile targets for train_criteria.py, eval_criteria.py if desired
3. Remove or properly document: make_groundtruth.py, train_best.py
4. Clarify: run_hpo_stage.py vs tune_max.py difference

### Phase 4: Clean Tests (1 hour)
1. Remove test_train_smoke.py
2. Mark augmentation tests as skip/xfail
3. Consolidate test_*_cache.py duplicates if possible

### Phase 5: Documentation Review (<1 hour)
1. Merge OPTIMIZATION_SUMMARY.md + CLEANUP_SUMMARY.md
2. Review AGENTS.md relevance
3. Consider consolidating docs/README.md concepts

---

## 11. RECOMMENDATIONS (Priority Order)

### CRITICAL (Do First)
1. **Remove src/Project/ duplication** → -376KB, eliminates confusion
2. **Remove augmentation module** → -100KB, aligns with "NO-AUG" mission
3. **Update imports in active scripts** → Consolidate to single import path

### HIGH (Do Second)
4. **Document/Fix script exposure** → Clarify which scripts are production-ready
5. **Remove test duplication** → test_train_smoke.py, consolidate cache tests

### MEDIUM (Nice to Have)
6. **Move augmentation deps to optional** → Reduce install size
7. **Consolidate similar documentation** → Merge optimization/cleanup docs

### LOW (Polish)
8. **Review AGENTS.md** → Ensure still relevant
9. **Audit CLI stubs** → Fix augmentation flags in CLI that don't route

---

## 12. UNUSED FILES SUMMARY

```
HIGH CONFIDENCE UNUSED:
- src/Project/              (entire directory, use architectures/ instead)
- src/psy_agents_noaug/augmentation/
- src/psy_agents_noaug/data/augmentation_utils.py
- src/psy_agents_noaug/data/classification_loader.py
- tests/test_augmentation_*.py
- tests/test_tfidf_cache*.py
- tests/test_train_smoke.py
- scripts/gpu_utilization.py (no clear use)
- scripts/profile_augmentation.py (no clear use)
- scripts/make_groundtruth.py (duplicates CLI command)

PARTIALLY UNCLEAR:
- scripts/train_criteria.py (production-ready but not exposed)
- scripts/eval_criteria.py (production-ready but not exposed)
- scripts/train_best.py (router script, conditional use)
- scripts/run_hpo_stage.py (used by HPO pipeline)

DOCUMENTATION CONSOLIDATION CANDIDATES:
- OPTIMIZATION_SUMMARY.md + CLEANUP_SUMMARY.md (can merge)
- AGENTS.md (verify relevance)
- docs/README.md vs root README.md (consolidate?)
```

---

## CONCLUSION

This codebase has **EXCELLENT FUNCTIONALITY** but suffers from:

1. **Architectural Debt:** Two parallel implementations (376KB + 528KB duplication)
2. **Mission Creep:** Extensive augmentation code in a "NO-AUG" baseline
3. **Script Exposure:** Production-ready scripts hidden from users
4. **Test Bloat:** 50+ augmentation tests in a NO-AUG project

**Recommended approach:**
- Execute Phase 1-2 consolidation (3-6 hours)
- Estimated space savings: -500KB code + reduced dependencies
- Improved code maintainability and user clarity

The codebase is **PRODUCTION-READY** for the stated NO-AUG baseline, but would benefit significantly from removing the duplication and augmentation artifacts.
