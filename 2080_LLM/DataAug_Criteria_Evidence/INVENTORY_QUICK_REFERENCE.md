# Quick Reference - Repository Structure & Key Files

## File Locations at a Glance

### Entry Points
| Command | File | Purpose |
|---------|------|---------|
| `python -m psy_agents_noaug.cli` | `src/psy_agents_noaug/cli.py` | CLI (train/tune/show-best) |
| `make train` | `scripts/train_criteria.py` | Standalone Criteria training ✓ |
| `make eval` | `scripts/eval_criteria.py` | Standalone evaluation ✓ |
| `make hpo-s0/s1/s2` | `scripts/run_hpo_stage.py` | Multi-stage HPO |
| `make tune-criteria-max` | `scripts/tune_max.py` | Maximal HPO (800 trials) |
| `make full-hpo-all` | `scripts/run_all_hpo.py` | All 4 architectures |

### Core Data Pipeline
| Component | File | Key Functions |
|-----------|------|----------------|
| **STRICT Validation** | `src/psy_agents_noaug/data/groundtruth.py` | `_assert_field_usage()`, `create_criteria_groundtruth()`, `create_evidence_groundtruth()` |
| **Data Loading** | `src/psy_agents_noaug/data/loaders.py` | `ReDSM5DataLoader` (HF + local CSV) |
| **Dataset Classes** | `src/psy_agents_noaug/data/datasets.py` | `ClassificationDataset`, `SpanDataset` |
| **Train/Val/Test Split** | `src/psy_agents_noaug/data/splits.py` | `stratified_split()`, `create_splits()` |

### Training Infrastructure
| Component | File | Classes |
|-----------|------|---------|
| **Training Loop** | `src/psy_agents_noaug/training/train_loop.py` | `Trainer` (AMP, early stopping, MLflow) |
| **Evaluation** | `src/psy_agents_noaug/training/evaluate.py` | `Evaluator` (metrics computation) |
| **Setup** | `src/psy_agents_noaug/training/setup.py` | Device setup, reproducibility |

### HPO System
| Component | File | Key Classes |
|-----------|------|-------------|
| **Optuna Runner** | `src/psy_agents_noaug/hpo/optuna_runner.py` | `OptunaRunner`, `create_search_space_from_config()` |
| **Maximal HPO** | `scripts/tune_max.py` | `suggest_common()`, `suggest_criteria_head()`, `suggest_evidence_head()` |
| **Multi-Stage** | `scripts/run_hpo_stage.py` | Hydra integration, stage management |

### Augmentation (CURRENTLY UNUSED)
| Component | File | Key Classes |
|-----------|------|-------------|
| **Pipeline** | `src/psy_agents_noaug/augmentation/pipeline.py` | `AugmenterPipeline`, `AugConfig` |
| **Registry** | `src/psy_agents_noaug/augmentation/registry.py` | `REGISTRY` (28+ augmenters) |
| **TF-IDF Cache** | `src/psy_agents_noaug/augmentation/tfidf_cache.py` | TF-IDF model caching |
| **Integration** | `src/psy_agents_noaug/data/augmentation_utils.py` | Collate function integration |

### Architectures (4 Identical Pattern)
| Architecture | Models | Data | Engine |
|--------------|--------|------|--------|
| **Criteria** | `src/psy_agents_noaug/architectures/criteria/models/model.py` | `data/dataset.py` | `engine/{train,eval}_engine.py` |
| **Evidence** | `src/psy_agents_noaug/architectures/evidence/models/model.py` | `data/dataset.py` | `engine/{train,eval}_engine.py` |
| **Share** | `src/psy_agents_noaug/architectures/share/models/model.py` | `data/dataset.py` | `engine/{train,eval}_engine.py` |
| **Joint** | `src/psy_agents_noaug/architectures/joint/models/model.py` | `data/dataset.py` | `engine/{train,eval}_engine.py` |

### Configuration Files
| Config Type | Location | Key Files |
|-------------|----------|-----------|
| **Field Mapping** | `configs/data/field_map.yaml` | CRITICAL - status vs cases separation |
| **Data Sources** | `configs/data/{hf_redsm5,local_csv}.yaml` | Data source configuration |
| **Tasks** | `configs/task/{criteria,evidence}.yaml` | Task definitions |
| **Models** | `configs/model/{bert_base,roberta_base,deberta_v3_base}.yaml` | Model architectures |
| **Training** | `configs/training/{default,optimized,supermax_optimized}.yaml` | Training hyperparameters |
| **HPO** | `configs/hpo/stage{0,1,2,3}*.yaml` | HPO search space definitions |

### Test Files (24 total)
| Category | Test Files |
|----------|-----------|
| **Core** | `test_groundtruth.py`, `test_loaders.py`, `test_integration.py` |
| **Training** | `test_training_smoke.py`, `test_train_smoke.py`, `test_arch_shapes.py` |
| **Augmentation** | `test_augmentation_registry.py`, `test_pipeline_*.py` (4 files), `test_tfidf_cache*.py` (2 files) |
| **HPO** | `test_hpo_config.py`, `test_hpo_integration.py` |
| **Other** | `test_cli_flags.py`, `test_qa_null_policy.py`, `test_seed_determinism.py`, `test_smoke.py`, `test_perf_contract.py`, `test_mlflow_artifacts.py` |

## Key Metrics at a Glance

| Metric | Value |
|--------|-------|
| Total Python Files | 100+ |
| Total Lines of Code | ~15K (psy_agents_noaug) + ~5K (scripts) + ~2K (tests) |
| Test Coverage | 67/69 passing (97.1%), 31% code coverage |
| Augmentation Methods | 28+ (16 nlpaug + 12 textattack) |
| HPO Modes | 3 (multi-stage, maximal, super-max) |
| Architectures | 4 (criteria, evidence, share, joint) |
| Configuration Groups | 12 (augmentation, data, model, task, training, hpo, criteria, evidence, share, joint) |

## Critical Points for Production

### MUST KNOW
1. **Field Separation** (`groundtruth.py`):
   - Criteria ONLY uses `status` field
   - Evidence ONLY uses `cases` field
   - Assertion fails if violated

2. **Two Implementations** (`src/Project/` vs `src/psy_agents_noaug/architectures/`):
   - Duplicate code (904 KB)
   - Train_criteria.py uses src/Project/
   - CLI uses src/psy_agents_noaug/
   - Must consolidate before production

3. **Augmentation Status**:
   - Code exists but is UNUSED in training paths
   - 28+ methods available but not called
   - Dataset classes accept augmenter but rarely use it
   - Dependencies installed but not utilized

### Key Files to Understand First
1. `configs/data/field_map.yaml` - Field validation rules
2. `src/psy_agents_noaug/data/groundtruth.py` - Core data validation
3. `src/psy_agents_noaug/training/train_loop.py` - Training orchestration
4. `scripts/tune_max.py` - HPO implementation
5. `src/psy_agents_noaug/augmentation/pipeline.py` - Augmentation hooks (if enabling)

## Quick Navigation

### To modify...
| Task | File(s) |
|------|---------|
| Field mappings | `configs/data/field_map.yaml` |
| Training hyperparameters | `configs/training/*.yaml` |
| HPO search space | `scripts/tune_max.py` + `configs/hpo/*.yaml` |
| Model architecture | `src/psy_agents_noaug/architectures/*/models/model.py` |
| Data loading | `src/psy_agents_noaug/data/loaders.py` |
| Augmentation methods | `src/psy_agents_noaug/augmentation/registry.py` |
| Training loop | `src/psy_agents_noaug/training/train_loop.py` |
| Evaluation metrics | `src/psy_agents_noaug/training/evaluate.py` |
| CLI commands | `src/psy_agents_noaug/cli.py` |

## Immediate Action Items

1. Decide on consolidation strategy (src/Project vs src/psy_agents_noaug)
2. Decide on augmentation (enable or remove)
3. Rename package from "noaug" to "aug" if needed
4. Update documentation to reflect current status
5. Run full test suite to baseline before changes
6. Plan production deployment

---

**For Complete Details**: See `INVENTORY.md` (1139 lines)
**Repository**: `/media/cvrlab308/cvrlab308_4090/YuNing/DataAug_Criteria_Evidence`
**Last Updated**: 2025-10-26
