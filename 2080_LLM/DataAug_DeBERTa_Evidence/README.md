DataAug DeBERTa Evidence
========================

Cross-task training and Optuna-based HPO pipeline for evidence-aware classification with DeBERTa-v3. The system now supports a production-grade, two-stage Optuna HPO driver with TextAttack/simple augmentation bitmasks (evidence sentence only), SQLite-backed MLflow with resilient buffering, and study summaries aligned with the provided JSON schema.

Installation
------------

```
poetry install
```

The project pins Poetry-managed dependencies and assumes Python 3.10+. For GPU training set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` if you observe CUDA OOM fragmentation.

Quick Start
-----------

Single run with the default configs:

```
poetry run dataaug train
```

Two-stage HPO with Optuna + MLflow (defaults: Stage A = 380 trials × 100 epochs, Stage B = 120 trials × 100 epochs):

```
poetry run dataaug tune
```

Equivalent Make targets (overridable via `TRIALS_A=`, `TRIALS_B=`, etc.):

```
make hpo-best
make retrain-best RETRAIN_SEEDS=3
```

Key Artifacts
-------------

- MLflow tracking database `sqlite:///mlflow.db` with disk-backed buffer `artifacts/mlflow_buffer/` to avoid run failures when the backend is temporarily unavailable.
- Checkpoints: `experiments/checkpoint.pt` (per job) plus per-run directories `experiments/trial_<run_name>/` containing `evaluation_report.json`.
- HPO artifacts: `artifacts/hpo/stage_a/` and `artifacts/hpo/stage_b/` hold frozen configs, Optuna visualisations, and metric snapshots (populated by the driver).
- Study summary: `poetry run dataaug summarize-study --study-name <name>` validates against `specs/002-storage-optimized-training/contracts/study_output_schema.json` before writing the summary JSON.

Configuration
-------------

Configurations live in `configs/` and are merged via OmegaConf:

- `train.yaml` – training, optimization, scheduler, and head defaults.
- `data.yaml` – dataset identifier, revision, splits, cache directories, and field mapping (set `fields.evidence` to control augmentation scope).
- `augmentation.yaml` – augmentation bitmasks (`simple.enabled_mask`, `ta.enabled_mask`) with per-method strength ranges and cross-family composition policy.
- `hpo.yaml` – Optuna study definition, sampler, and pruner defaults.
- `mlflow.yaml` – tracking URI and experiment name.

Override any value from the CLI, e.g. `poetry run dataaug train --override train.num_epochs=5`.

Reproducibility Notes
---------------------

- The two-stage driver deterministically seeds trials from `(study_name, trial_number, global_seed)` and persists the full resolved config per trial for retraining.
- Tokenized datasets are cached under `Data/token_cache/` keyed by dataset id, tokenizer, max length, and augmentation signature to avoid redundant preprocessing across trials.
- The tokenizer revision and dataset revision should be pinned in `configs/train.yaml` and `configs/data.yaml` respectively.

CI
--

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs lint (`ruff`), formatting (`black --check`), type checks (`mypy`), unit + integration pytest suites, and JSON schema validation.

Runbook
-------

1. **Two-stage HPO**
   ```bash
   make hpo-best TRIALS_A=380 EPOCHS_A=100 TRIALS_B=120 EPOCHS_B=100 K_TOP=5 SEED=42 TIMEOUT=604800
   ```
   Results: Optuna studies `deberta_v3_evidence_stage_a`/`stage_b`, artifacts under `artifacts/hpo/`, MLflow runs in `mlflow.db`.
2. **Inspect MLflow UI (optional)**
   ```bash
   poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
   ```
3. **Retrain best config with multiple seeds**
   ```bash
   make retrain-best RETRAIN_SEEDS=3
   ```
   Summary written to `artifacts/hpo/retrain/retrain_summary.json` (with mean ± std objective and best checkpoint path).
