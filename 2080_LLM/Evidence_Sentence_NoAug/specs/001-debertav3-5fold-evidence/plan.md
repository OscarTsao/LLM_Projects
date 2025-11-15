# Implementation Plan: 5-Fold DeBERTaV3-base Evidence Binding (NSP-Style)

**Branch**: `001-debertav3-5fold-evidence` | **Date**: 2025-11-12 | **Spec**: spec.md
**Input**: Feature specification from `/specs/001-debertav3-5fold-evidence/spec.md`

## Summary

Train and evaluate a binary classifier using `microsoft/deberta-v3-base` with a
Hugging Face sequence classification head on NSP-style criterion–sentence pairs
(`[CLS] <criterion> [SEP] <sentence> [SEP]`). Use 5-fold CV with
GroupStratifiedKFold by `post_id` (fallback GroupKFold) to avoid leakage.
Parameters are managed via Hydra; experiments and artifacts tracked in MLflow
(`sqlite:///mlflow.db`, `./mlruns`). HPO: not in scope for this feature.

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: PyTorch, Transformers, Datasets/Pandas, scikit-learn,
iterative stratification (or equivalent for group‑stratified CV), Hydra,
MLflow, Optuna  
**Storage**: Local files for data; MLflow SQLite DB `mlflow.db`, artifacts under `mlruns/`; optional
Optuna SQLite `optuna.db`  
**Testing**: pytest (unit tests for data parsing, metric computation, splitting)  
**Target Platform**: Linux/macOS dev; GPU optional for speed  
**Project Type**: single (library + CLI)  
**Performance Goals**: Reasonable epoch time on DeBERTaV3‑base; CV completes without OOM  
**Constraints**: Enforce reproducibility (seed, deterministic where possible); honor NSP input format  
**Scale/Scope**: Dataset size TBD; CV=5; batch size tuned to memory

## Constitution Check

Gates (all PASS):
- P1 BERT-based classifier (HF): using DeBERTaV3‑base ✓
- P2 NSP input format: criterion–sentence pairs ✓
- P3 Hydra config: configs/ with overrides ✓
- P4 MLflow local: sqlite:///mlflow.db + ./mlruns ✓
- P5 Optuna if HPO: N/A (not used in this feature) ✓
- P6 Reproducibility: seeds, env snapshot, Hydra config logging ✓

## Project Structure

### Documentation (this feature)

```text
specs/001-debertav3-5fold-evidence/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── contracts/
```

### Source Code (repository root)

```text
src/Project/SubProject/
├── models/              # HF model wrappers (extend if needed)
├── utils/               # logging, seeding, mlflow helpers
└── engine/              # NEW: training/inference entrypoints (to add)

scripts/
└── train_cv.py          # NEW: CLI for 5-fold training (to add)

tests/
├── unit/
└── integration/
```

**Structure Decision**: Single-project Python package aligned to existing
`src/Project/SubProject/` layout; add `engine/` and `scripts/train_cv.py` for
Trainer-based CV loop and Hydra integration. Tests mirror under `tests/`.

## Phase 0: Research (resolve unknowns)

Topics and decisions captured in research.md:
- GroupStratifiedKFold implementation: prefer scikit‑learn's
  StratifiedGroupKFold if available; otherwise use a third‑party iterative
  stratification library or approximate via GroupKFold + per‑fold class
  balancing checks.
- Fused AdamW availability: use HF Trainer with `optim=adamw_torch_fused` and
  auto‑fallback to `adamw_torch` if unsupported.
- Loss implementation in Trainer: override `compute_loss` to support weighted CE
  and optional Focal Loss (γ=2.0, α from frequencies) with Hydra switch.
- Metrics: compute accuracy, macro‑F1, positive‑class F1, ROC‑AUC, PR‑AUC via
  scikit‑learn; log per‑fold and aggregate.

## Phase 1: Design & Contracts

Artifacts to generate:
- data-model.md: Sample, FoldSplit, ModelArtifact entities and fields
- quickstart.md: end‑to‑end commands (MLflow UI + training CLI + inference)
- contracts/: N/A (no external API; CLI contract described in quickstart)

Agent context updated via `.specify/scripts/bash/update-agent-context.sh codex`.

## Phase 2: Implementation Strategy (high level)

1) Data loader: build criterion–sentence pairs; construct stratified random
   negatives to target 1:3 pos:neg ratio (`data.neg_ratio=3`), then persist
   split manifests per fold (grouped by post_id)
2) Trainer (HF) setup: DeBERTaV3‑base, tokenizer (max_length=512; truncation=longest_first),
   full fine‑tune (unfreeze all layers), compute_metrics
3) Optimizer & schedule: `adamw_torch_fused` or fallback to `adamw_torch`;
   LR scheduler `linear` with `warmup_ratio=0.06`
4) Precision: prefer BF16; else FP16; else FP32 (log chosen mode)
5) Loss: weighted CE by default; focal optional
6) Metrics & selection: compute accuracy, macro/pos F1, ROC/PR AUC. Set
   `metric_for_best_model=f1_macro` and `greater_is_better=true` in Trainer.
7) CV orchestration: parent MLflow run with 5 child runs
8) Aggregation: compute mean/std metrics; save summary JSON + plots
9) Inference: simple function/CLI for single pair
10) Reproducibility: seed, env snapshot (`pip freeze`), config logging

### Component Blueprint (refined)

**Data pipeline — `src/Project/SubProject/data/dataset.py`**
- Normalize DSM-5 criteria from `data/DSM5/*.json` and posts from `data/redsm5/*.csv`.
- Emit a canonical `Sample` record per `(post_id, sentence_id, criterion_id)` with `source_label`, `neg_sampling_strategy`, and checksum metadata to simplify manifest validation.
- Persist NSP-ready parquet files plus `outputs/manifests/folds.json` (fold assignment + seed + strategy) for reproducibility.

**Hydra configuration — `configs/`**
- Root `config.yaml` wires groups: `data/`, `model/`, `trainer/`, `loss/`, `cv/`, `logger/`, `runtime/`.
- `configs/data/evidence_pairs.yaml` documents file locations, neg sampling knobs, and manifest paths.
- `configs/trainer/deberta_cv.yaml` captures HF `TrainingArguments`, optimizer fallback, precision policy, and MLflow run metadata.

**Trainer & engines — `src/Project/SubProject/engine/`**
- `train_engine.py` hosts dataset-to-Trainer adapters, loss override, and CV orchestration (parent MLflow run + child folds).
- `eval_engine.py` provides single-pair inference helper plus batched evaluation hooks shared by the quickstart CLI.
- Loss handling (weighted CE vs focal) lives close to Trainer subclass to keep Hydra toggles localized.

**Utilities**
- `src/Project/SubProject/utils/metrics.py` centralizes accuracy/F1/ROC/PR computation so both Trainer callbacks and aggregation step reuse identical logic.
- `mlflow_utils.py` gains helpers for nested runs, dataset manifest logging, and automatic `pip freeze` capture.
- `seed.py` already exists; expose Hydra flag to wire deterministic controls from `configs/runtime/default.yaml`.

**Scripts**
- `scripts/train_cv.py` is the Hydra entrypoint for full CV, invoking `train_engine.run_cv`.
- `scripts/infer_pair.py` (or module entrypoint) loads model artifacts for CLI inference.
- Optional `scripts/aggregate_cv.py` can materialize the `cv_summary.json` artifact outside the training job if needed for reruns without retraining.

### Detailed component workstreams

#### 2.1 Data ingestion & manifest pipeline — `src/Project/SubProject/data/dataset.py`
- Build a composable loader that:
  1. Reads DSM-5 criteria JSON and normalizes criterion IDs/text (strip whitespace, collapse unicode quotes).
  2. Loads post CSVs plus NSP annotations; enforces presence of `post_id`, `sentence_id`, `label`.
  3. Applies stratified random negative sampling to reach `data.neg_ratio` while tagging each generated pair with `source_label=neg_sampled`.
  4. Materializes canonical `Sample` records sorted by composite key and writes `outputs/datasets/evidence_pairs.parquet`.
  5. Generates five deterministic fold manifests using preferred splitter + fallback; stores `fold_index`, `pos_count`, `neg_count`, `seed`, and strategy metadata in `outputs/manifests/folds.json`.
- Acceptance: rerunning with the same seed yields byte-identical manifests; each manifest logs counts that match `cv_summary`.

#### 2.2 Hydra configuration set — `configs/`
- `configs/config.yaml` defines defaults + optional runtime overrides (precision, logging).
- `configs/data/evidence_pairs.yaml` enumerates file paths, neg sampling knobs, manifest destinations.
- `configs/model/deberta_v3.yaml` and `configs/model/debug.yaml` cover full-finetune vs frozen modes.
- `configs/trainer/cv.yaml` mirrors HF `TrainingArguments` (max epochs, LR, scheduler, BF16/FP16 flags) and toggles MLflow tags.
- `configs/loss/*.yaml` toggles `weighted_ce` vs `focal`.
- Acceptance: `python scripts/train_cv.py +experiment=dryrun` prints resolved config, and overriding any group via CLI updates the run metadata in MLflow.

#### 2.3 Trainer + CV orchestration — `src/Project/SubProject/engine/train_engine.py`
- Implement:
  - Dataset adapters that convert manifest rows into `datasets.Dataset` objects with tokenizer encodings (max_length=512, `truncation="longest_first"`).
  - Custom Trainer subclass overriding `compute_loss` for weighted CE/focal and injecting class weights per fold.
  - CV driver that spins a parent MLflow run, loops through folds, and reuses the same tokenizer/model config while reinitializing weights per fold.
  - Precision management (BF16→FP16→FP32) plus optimizer fallback detection logged per fold.
- Acceptance: Each fold logs metrics + artifacts, and parent run metadata references all child run IDs.

#### 2.4 Metrics aggregation + MLflow artifacts — `src/Project/SubProject/engine/aggregation.py`
- Aggregate per-fold metrics into mean/std, plus best-fold metadata.
- Generate confusion matrices + ROC/PR plots using shared metrics utilities.
- Write `outputs/metrics/cv_summary.json` and log to parent run along with plot images and CSV exports of per-fold metrics.
- Acceptance: `cv_summary.json` contains deterministic statistics; MLflow UI shows artifacts under parent run only.

#### 2.5 Inference surfaces & CLI — `src/Project/SubProject/engine/eval_engine.py`, `scripts/infer_pair.py`
- Provide `score_pair(criterion_text, sentence_text, model_uri)` that loads tokenizer + model from MLflow artifact path, applies the same NSP tokenization, and returns label + probability + provenance (model run ID, precision mode).
- CLI accepts raw text or file path arguments, supports Hydra overrides for tokenizer/model URIs, and logs inference metadata to MLflow (optional).
- Acceptance: CLI works offline given artifacts, prints structured output, and can be scripted inside quickstart smoke tests.

#### 2.6 Testing & QA harness — `tests/`
- Unit tests:
  - `tests/unit/test_dataset_builder.py`: ratio enforcement, group integrity, manifest checksum.
  - `tests/unit/test_metrics.py`: accuracy/F1/ROC/PR determinism.
  - `tests/unit/test_loss.py`: weighted CE vs focal numeric parity on toy tensors.
- Integration/smoke:
  - `tests/integration/test_train_cv_smoke.py`: run 1 fold / 1 epoch on synthetic dataset fixture to verify logging.
  - `tests/integration/test_infer_cli.py`: execute CLI on tiny saved model (mock) and assert output JSON.
- Acceptance: suite runnable via `pytest -m "not gpu"` using CPU-safe fixtures; GPU-specific tests guarded by markers.

## Complexity Tracking

N/A — No constitution violations.
