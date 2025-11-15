# Module Map

This document summarizes the major modules and how they fit together.

## Core Package: `src/psy_agents_noaug/`

- `cli.py` — Typer CLI: thin wrappers for train, tune (Optuna), and show-best.
- `data/`
  - `loaders.py` — strict HF/local loaders and post_id grouping split tools.
  - `groundtruth.py` — STRICT rules (criteria from `status`, evidence from `cases`).
  - `datasets.py` — Classification dataset (lazy/eager tokenisation) + collate.
  - `classification_loader.py` — DataLoaders + worker seeding + optional augment.
  - `augmentation_utils.py` — Build augmentation pipeline/resources for evidence.
- `models/`
  - `encoders.py` — HF encoder wrappers (pooling, LoRA, grad checkpointing).
  - `criteria_head.py`, `evidence_head.py` — simple MLP heads and combined models.
- `training/`
  - `train_loop.py` — AMP, ES, grad accumulation, MLflow logging, checkpoints.
  - `evaluate.py` — metrics, per-criterion F1, AUROC guards, reporting.
  - `setup.py` — optimizer/scheduler helpers.
- `augmentation/`
  - `registry.py` —  unified augmenter registry (nlpaug, TextAttack).
  - `pipeline.py` — deterministic pipeline + stats + worker seeding helpers.
  - `tfidf_cache.py` — fit/cache TF‑IDF for TfIdfAug.
- `hpo/`
  - `optuna_runner.py` — TPE/Hyperband + MLflow integration (generic runner).
- `utils/`
  - `reproducibility.py` — seeds/devices/dataloader tuning.
  - `mlflow_utils.py`, `logging.py`, `logging_config.py`, `type_aliases.py` — infra.

## Architectures: `src/psy_agents_noaug/architectures/`

Four variants with similar structure: `criteria/`, `evidence/`, `share/`, `joint/`.

- `data/dataset.py` — dataset builders per-architecture.
- `models/model.py` — encoder + head assembly with HPO-friendly cfg.
- `engine/train_engine.py` — compact training loop for HPO/integration.
- `engine/eval_engine.py` — evaluation/prediction conveniences.
- `utils/` — logging, seeding, MLflow, checkpoint and Optuna adapters.
- `utils/heads.py`, `utils/outputs.py`, `utils/dsm_criteria.py` — shared blocks.

## Scripts: `scripts/`

- `make_groundtruth.py` — generate GT (STRICT), save splits, validate.
- `tune_max.py` — maximal Optuna search across models/schedulers/heads.
- `train_best.py`, `run_hpo_stage.py`, `run_all_hpo.py` — orchestration helpers.

## Configs: `configs/`

- `config.yaml` — Hydra composition root.
- `data/*.yaml` — HF/local configs and field map.
- `task/*.yaml` — criteria/evidence task metadata.
- `model/*.yaml` — encoder + optional LoRA config.
- `training/*.yaml` — epochs/batch/scheduler/optimizer/AMP.
- `hpo/*.yaml` — staged search definitions.

## Tests: `tests/`

- Smoke tests for training/eval, strict groundtruth validation, augmentation infra,
  HPO config checks, CLI flags, perf contracts.

## Typical Flow

1. Generate strict groundtruth and splits using `scripts/make_groundtruth.py`.
2. Build DataLoaders via `data/classification_loader.py` (optional augmentation).
3. Choose model (encoder + head), train with `training/train_loop.py`.
4. Track to MLflow; export metrics via `scripts/export_metrics.py`.
5. Run staged HPO with `scripts/tune_max.py` or Hydra configs.
