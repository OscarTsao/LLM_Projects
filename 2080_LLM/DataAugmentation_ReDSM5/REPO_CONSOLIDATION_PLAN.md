# Psy Agents Consolidation Plan

## Objectives
- Refactor existing codebases into two reproducible repositories with aligned interfaces:
  1. `psy_agents_noaug`: baseline criteria + evidence matching without augmentation.
  2. `psy_agents_aug`: augmentation-enabled counterpart with identical CLI and evaluation stack.
- Standardize configs, CLI, packaging, and experiment tracking to enable drop-in comparisons.
- Preserve proven components from legacy repos while enforcing strict data provenance rules.

## Source Material Mapping
- `OscarTsao/NoAug_Criteria_Evidence`
  - Reuse: MLflow/Optuna wiring, no-augmentation defaults, dataset handling.
  - Target: seed `psy_agents_noaug` scaffolding, ground-truth scripts, baseline configs.
- `OscarTsao/DataAug_Multi_Both`
  - Reuse: Poetry structure, augmentation-aware configs, README language for data/eval.
  - Target: seed `psy_agents_aug` scaffolding, Hydra composition layout.
- `OscarTsao/DataAugmentation_ReDSM5`
  - Reuse: augmentation operators, seeds/combos logic, deterministic caching concepts.
  - Target: implement `src/psy_agents_aug/augment/` pipelines with unified interfaces.
- `OscarTsao/Criteria_Evidence_Agent`
  - Reuse: multi-task trainer, encoder abstractions, Optuna/MLflow callbacks, Makefile targets.
  - Target: shared core inside both repositories (`models`, `training`, `hpo`, `utils`).
- `OscarTsao/Psy_RAG`
  - Reuse: DSM-5 criteria JSON location/format, textual resources.
  - Target: populate `data/raw/redsm5/dsm_criteria.json` and ground-truth pipeline references.

## Unified Repository Architecture
```
.
├── README.md
├── LICENSE
├── pyproject.toml          # Poetry + lockfile
├── .pre-commit-config.yaml
├── .devcontainer/
├── configs/
│   ├── config.yaml
│   ├── data/
│   │   ├── hf_redsm5.yaml
│   │   ├── local_csv.yaml
│   │   └── field_map.yaml
│   ├── model/
│   │   ├── bert_base.yaml
│   │   ├── roberta_base.yaml
│   │   └── deberta_v3_base.yaml
│   ├── training/
│   │   └── default.yaml
│   ├── hpo/
│   │   ├── stage0_sanity.yaml
│   │   ├── stage1_coarse.yaml
│   │   ├── stage2_fine.yaml
│   │   └── stage3_refit.yaml
│   └── task/
│       ├── criteria.yaml
│       └── evidence.yaml
├── data/
│   ├── raw/redsm5/{posts.csv, annotations.csv, dsm_criteria.json}
│   └── processed/{splits.json, criteria_groundtruth.csv, evidence_groundtruth.csv}
├── mlruns/
├── outputs/
├── scripts/
│   ├── make_groundtruth.py
│   ├── run_hpo_stage.py
│   ├── train_best.py
│   └── export_metrics.py
├── src/
│   ├── psy_agents_(noaug|aug)/
│   │   ├── data/{loaders.py, splits.py, groundtruth.py}
│   │   ├── models/{encoders.py, criteria_head.py, evidence_head.py}
│   │   ├── training/{train_loop.py, evaluate.py}
│   │   ├── hpo/optuna_runner.py
│   │   ├── utils/{logging.py, reproducibility.py, mlflow_utils.py}
│   │   ├── cli.py
│   │   └── __init__.py
│   └── psy_agents_aug/augment/ (augmentation repo only)
├── tests/
│   ├── test_groundtruth.py
│   ├── test_loaders.py
│   ├── test_training_smoke.py
│   ├── test_hpo_config.py
│   └── test_augment_contract.py (augmentation repo only)
├── Makefile
└── .gitignore
```

## Immediate Workstreams
1. **Data Core**
   - Implement `groundtruth.py` with strict `status`/`cases` only logic and raise on other usage.
   - `scripts/make_groundtruth.py` to orchestrate dataset ingestion and CSV emission.
   - Build deterministic `splits.py` with HF + local CSV support; persist `splits.json`.
2. **Model/Training Core**
   - Import encoder + head implementations from `Criteria_Evidence_Agent`, update to shared API.
   - Normalize training loop and evaluation modules, integrate MLflow logging.
3. **Hydra & CLI**
   - Create Hydra entrypoint with commands: `make_groundtruth`, `train`, `hpo`, `refit`, `evaluate_best`, `export_metrics`.
   - Compose consistent config groups for data/model/training/task/hpo.
4. **Augmentation Module (AUG repo)**
   - Port pipelines from `DataAugmentation_ReDSM5`, expose `apply(batch, config)` contract.
   - Update data loaders to inject augmentations on train split when enabled.
5. **Tooling & CI**
   - Generate `.pre-commit-config.yaml` with ruff/black/isort/trailing whitespace.
   - Author GitHub Actions workflow: lint + tests + build.
   - Set up Makefile targets aligning with spec.
6. **Documentation**
   - Draft READMEs referencing strict label provenance, CLI usage, and reproducibility steps.
   - Insert "Migrated-from" annotations where code copied verbatim.

## Open Questions / Items to Clarify
- Confirm availability of GPU images for devcontainer (CUDA version, base image requirements).
- Determine whether augmentation repo must include DVC artifacts or if simple config toggles suffice.
- Decide on MLflow tracking server defaults (local file vs remote) for CI compatibility.

## Next Steps
1. Scaffold directory structure for both repositories within the workspace.
2. Create shared config/templates and placeholders.
3. Incrementally migrate code into the new layout, validating with unit tests at each step.
4. Decommission redundant directories from source repositories once functionality parity is confirmed.
