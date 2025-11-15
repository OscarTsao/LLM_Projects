# Tasks: 5-Fold DeBERTaV3-base Evidence Binding (NSP-Style)

**Input**: Design documents from `/specs/001-debertav3-5fold-evidence/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md  
**Tests**: Required where spec/plan call out verification (dataset builder, metrics, CV wiring, inference).  
**Organization**: Phased by setup → foundation → user stories → polish to keep increments independently deliverable.

## Format: `[ID] [P?] [Story] Description`

- `[P]` means the task can proceed in parallel (different files, no blocking dependency).
- `[US#]` labels story-specific tasks (omit for setup/foundation/polish).
- Include concrete file paths in every description.

## Path Conventions

- Core code: `src/Project/SubProject/`
- Configs: `configs/`
- Scripts/CLIs: `scripts/`
- Tests mirror under `tests/`
- Generated artifacts: `outputs/`, `mlruns/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish Hydra scaffolding + logging defaults used by all later phases.

- [ ] T001 Create Hydra root defaults in `configs/config.yaml` wiring `data/`, `model/`, `trainer/`, `loss/`, `cv/`, `logger/`, and `runtime` groups plus global metadata (seed, tracking URI, artifact root).
- [ ] T002 [P] Add MLflow logger profile `configs/logger/mlflow.yaml` capturing `sqlite:///mlflow.db`, `./mlruns`, experiment name, and default tags consumed by every CLI.
- [ ] T003 [P] Define runtime controls in `configs/runtime/default.yaml` (deterministic seeding, precision toggles, environment overrides) that `scripts/train_cv.py` reads before Hydra instantiation.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build shared data, metrics, and logging utilities that all stories rely on. No story work may start before this phase completes.

- [ ] T004 Define evidence data config `configs/data/evidence_pairs.yaml` with DSM-5/RedSM5 paths, neg sampling knobs, manifest destinations, and validation expectations (required columns).
- [ ] T005 [P] Implement Sample builder in `src/Project/SubProject/data/dataset.py` to load criteria/posts, normalize text, compute canonical `(post_id,sentence_id,criterion_id)` IDs + SHA1s, and emit NSP-ready rows to `outputs/datasets/evidence_pairs.parquet`.
- [ ] T006 [P] Add negative sampling + fold split generation in `src/Project/SubProject/data/dataset.py` enforcing 1:3 pos:neg ratio, recording seed/strategy/pos-neg counts per fold, and persisting metadata-rich manifests to `outputs/manifests/folds.json`.
- [ ] T007 [P] Add dataset builder tests in `tests/unit/test_dataset_builder.py` using fixtures under `tests/fixtures/data/` to assert canonical IDs, ratio, grouping, and deterministic manifests (byte-identical given same seed).
- [ ] T008 [P] Create metrics helper `src/Project/SubProject/utils/metrics.py` computing accuracy, macro-F1, positive-class F1, ROC-AUC, and PR-AUC for reuse by Trainer callbacks, aggregation, and inference.
- [ ] T009 [P] Extend `src/Project/SubProject/utils/mlflow_utils.py` with helpers for nested CV runs, dataset manifest/`pip freeze` logging, and parent-run aggregation artifact registration.
- [ ] T010 [P] Cover metrics helper logic with deterministic unit tests in `tests/unit/test_metrics.py` (logit→prob conversion, tolerance for FP16/BF16).

---

## Phase 3: User Story 1 – Train 5-Fold DeBERTaV3 Evidence Classifier (Priority: P1) 🎯 MVP

**Goal**: Deliver a Hydra-driven CLI that trains 5-fold CV on NSP-style data, logging each fold to MLflow.  
**Independent Test**: `python scripts/train_cv.py cv.folds=5` completes all folds, logs per-fold metrics/artifacts, and writes `outputs/metrics/cv_summary.json` without errors.

### Tests for User Story 1

- [ ] T011 [P] [US1] Build synthetic evidence dataset fixture `tests/fixtures/data/small_evidence.csv` (plus criteria JSON) to exercise CV wiring without GPU requirements.
- [ ] T012 [P] [US1] Write Hydra/Trainer config unit test `tests/unit/test_train_cv_config.py` to assert optimizer fallback, precision flags, manifest seeding, and dry-run mode.

### Implementation for User Story 1

- [ ] T013 [US1] Implement tokenizer + dataset adapter in `src/Project/SubProject/engine/train_engine.py` that converts Samples into Hugging Face `Dataset` objects with `[CLS] criterion [SEP] sentence [SEP]`, `max_length=512`, and truncation warning telemetry.
- [ ] T014 [P] [US1] Implement weighted CE / focal loss handling via a custom Trainer subclass in `src/Project/SubProject/engine/train_engine.py`, injecting per-fold class weights derived from manifest stats and honoring Hydra `loss.*` params.
- [ ] T015 [US1] Implement the 5-fold CV loop in `src/Project/SubProject/engine/train_engine.py` that wraps parent/child MLflow runs, selects `adamw_torch_fused` with fallback, applies BF16→FP16→FP32 policy, and fails fast if any fold errors (OOM/CUDA/checkpoint).
- [ ] T016 [P] [US1] Create Hydra CLI `scripts/train_cv.py` that seeds via `Project.SubProject.utils.set_seed`, performs optional manifest-only dry runs, and invokes `train_engine.run_cv()` with overrides shown in quickstart.md.

**Checkpoint**: MVP achieved—training CLI produces per-fold runs + checkpoints.

---

## Phase 4: User Story 2 – Aggregate and Report CV Metrics (Priority: P2)

**Goal**: Surface aggregated metrics/artifacts (mean/std, ROC/PR, confusion matrices) logged to the parent MLflow run.  
**Independent Test**: After a CV run, parent MLflow run contains `cv_summary.json`, plots, and tags referencing the best fold model ID.

### Implementation for User Story 2

- [ ] T017 [US2] Implement aggregation utilities in `src/Project/SubProject/engine/aggregation.py` to compute mean/std metrics, tie-break best fold deterministically, and render confusion/ROC/PR artifacts from per-fold outputs.
- [ ] T018 [US2] Wire aggregation into `src/Project/SubProject/engine/train_engine.py` (or dedicated CLI) so parent runs log `outputs/metrics/cv_summary.json`, upload plots, and set MLflow tags referencing best fold metrics/run IDs.

### Tests for User Story 2

- [ ] T019 [P] [US2] Add unit tests in `tests/unit/test_cv_aggregation.py` using synthetic per-fold metrics to verify statistics, artifact paths, and MLflow logging hooks (mock client).

**Checkpoint**: Research stakeholders can compare experiments via aggregated artifacts.

---

## Phase 5: User Story 3 – Inference API for Criterion–Sentence (Priority: P3)

**Goal**: Provide a callable + CLI inference surface that scores a single pair using the best fold model.  
**Independent Test**: CLI call with criterion + sentence returns label/probability and logs provenance (model URI, fold index).

### Implementation for User Story 3

- [ ] T020 [US3] Implement inference helper in `src/Project/SubProject/engine/eval_engine.py` that loads tokenizer/model artifacts, encodes `[CLS] criterion [SEP] sentence [SEP]`, returns label + probability, and logs selected fold/precision metadata.
- [ ] T021 [P] [US3] Create CLI entrypoint `scripts/infer_pair.py` (or module `python -m src.Project.SubProject.engine.eval_engine`) parsing CLI args/Hydra overrides and emitting structured JSON for downstream tooling.

### Tests for User Story 3

- [ ] T022 [P] [US3] Add integration test `tests/integration/test_infer_pair.py` using a tiny mocked model artifact under `tests/artifacts/` to assert deterministic outputs and logged metadata.

**Checkpoint**: Downstream consumers can score pairs without rerunning CV.

---

## Final Phase: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, reporting, and validation automation shared across stories.

- [ ] T023 Update documentation in `README.md` and `specs/001-debertav3-5fold-evidence/quickstart.md` with finalized Hydra commands, manifest-only dry-run workflow, aggregation reruns, and inference usage (include validation run IDs).
- [ ] T024 [P] Author `docs/mlflow_report.md` summarizing artifact locations, key metrics, precision modes, and how to launch MLflow UI for stakeholders.
- [ ] T025 [P] Add smoke-test script `scripts/validate_quickstart.sh` that runs manifest build, a 1-epoch/1-fold training dry run, aggregation, and inference CLI to keep quickstart instructions executable.

---

## Dependencies & Execution Order

- **Setup → Foundational**: Setup tasks unblock foundational utilities; both must finish before story work.
- **US1** depends on Foundational for data/metrics/logging; delivers MVP training CLI.
- **US2** depends on US1 outputs (per-fold metrics/artifacts) but can start once fold outputs are available.
- **US3** depends on US1 checkpoints; can run in parallel with US2 after best model selection is defined.
- **Polish** runs after desired user stories complete.

---

## Parallel Opportunities

- `[P]` tasks in Setup/Foundational can run concurrently (different files).
- Within US1, fixture/tests (T011–T012), loss (T014), and CLI wiring (T016) can proceed in parallel once dataset adapter interface (T013) is defined.
- US2 aggregation work may start while later US1 fold training finishes (using stubbed outputs).
- US3 inference work can progress in parallel with US2 once the best model artifact schema is defined.

---

## Parallel Example: User Story 1

```bash
# Test scaffolding
Task T011: Build synthetic dataset fixture (tests/fixtures/data/small_evidence.csv)
Task T012: Add Hydra/Trainer config tests (tests/unit/test_train_cv_config.py)

# Implementation
Task T014: Implement Trainer loss subclass (src/Project/SubProject/engine/train_engine.py)
Task T016: Build Hydra CLI (scripts/train_cv.py)
```

---

## Implementation Strategy

### MVP First (User Story 1)

1. Finish Setup + Foundational (T001–T010) to guarantee configs, data builders, and logging exist.
2. Ship US1 (T011–T016) to deliver the 5-fold training CLI and baseline metrics.
3. Validate via quickstart command + MLflow UI before touching US2/US3.

### Incremental Delivery

1. MVP (US1) → training CLI + per-fold metrics.
2. Layer US2 aggregation (T017–T019) for reporting artifacts.
3. Add US3 inference (T020–T022) for downstream consumers.
4. Polish (T023–T025) to finalize docs and validation scripts.

### Parallel Team Strategy

1. Team completes Setup/Foundational collectively.
2. Split US1 tasks by responsibility (data adapter vs loss vs CLI).
3. In parallel, another contributor prototypes aggregation while US1 training wraps.
4. Assign inference CLI to a separate teammate once artifact schema stabilizes.
