# Tasks: four-agent-pipeline

**Input**: Design documents from `/specs/001-four-agent-pipeline/`
**Prerequisites**: plan.md (required), spec.md (required), research.md (optional), data-model.md (optional), contracts/ (optional)

## Format: `[ID] [P?] [Story] Description`
- `[P]` = parallelizable
- `[US#]` = user story label (story phases only)
- Include exact file paths

---

## Phase 1: Setup (Shared Infrastructure)

- [ ] T001 Create repository structure per plan in project root
- [ ] T002 [P] Add `schemas/predictions.schema.json` and `schemas/criteria.schema.json` references to README
- [ ] T003 [P] Initialize pre-commit, linting, formatting (ruff/black or equivalent)
- [ ] T004 [P] Create `scripts/run_pipeline.sh` and `scripts/run_hpo_evidence.sh`

---

## Phase 2: Foundational (Blocking Prerequisites)

- [ ] T005 Create `src/schema/types.py` with `EvidenceUnit`, `CriteriaResult`, `Suggestion`
- [ ] T006 [P] Implement `src/utils/seed.py` and `src/utils/io.py` (splits/artifacts I/O)
- [ ] T007 [P] Implement `src/utils/hydra_mlflow.py` (Hydra init + mlflow URI `file:./mlruns`)
- [ ] T008 Create Hydra configs under `configs/` per plan
- [ ] T009 [P] Author `configs/data/redsm5.yaml` (dataset paths, split files)
- [ ] T010 [P] Author `configs/evidence/pairclf.yaml` (model/optimizer/loss/trainer)
- [ ] T011 [P] Author `configs/criteria/aggregator.yaml` (symptoms, thresholds path, temperature)
- [ ] T012 [P] Author `configs/suggest/voi.yaml` (top_k, uncertain band)
- [ ] T013 [P] Author `configs/eval/default.yaml` (metrics, gates)
- [ ] T014 [P] Author `configs/pipeline/default.yaml` (wiring, output paths, mlflow URI)
- [ ] T015 [P] Author `configs/hpo/evidence_pairclf.yaml` (search space + study)
- [ ] T016 Generate deterministic GroupKFold splits and persist to `configs/data/splits/{train,dev,test}.jsonl`
- [ ] T017 Validate config composition via Hydra dry-run

**Checkpoint**: Foundation ready — proceed to user stories

---

## Phase 3: User Story 1 — Evidence Agent (Priority: P1) 🎯 MVP

**Independent Test**: Can train/infer and write `outputs/evaluation/{run}/predictions.jsonl` matching schema

- [ ] T018 [P] [US1] Implement `src/evidence/train_pairclf.py` (Hydra entry, seed, mlflow)
- [ ] T019 [P] [US1] Implement `src/evidence/infer_pairclf.py` (batch predict writer)
- [ ] T020 [US1] Implement `src/agents/evidence_agent.py` facade using configs and train/infer modules
- [ ] T021 [US1] Validate JSONL against `schemas/predictions.schema.json`
- [ ] T022 [US1] Write training artifacts to `outputs/training/{run}/model.ckpt`, `config.yaml`, `val_metrics.json`
- [ ] T023 [US1] Validate presence of required training artifacts under `outputs/training/{run}/`

---

## Phase 4: User Story 2 — Criteria Agent (Priority: P2)

**Independent Test**: Produces `criteria.jsonl` with `p_dx`, decision, cited EvidenceUnits

- [ ] T024 [P] [US2] Implement `src/criteria/aggregate.py` (feature derivation, rule counts)
- [ ] T025 [P] [US2] Add temperature scaling + thresholds read from `artifacts/calibration.json`
- [ ] T026 [US2] Implement `src/agents/criteria_agent.py` to write `criteria.jsonl`
- [ ] T027 [US2] Validate JSONL against `schemas/criteria.schema.json`
- [ ] T028 [US2] Ensure deterministic unique EU IDs and use them for citation
- [ ] T029 [US2] Validate that all "likely" decisions cite ≥1 present EvidenceUnit

---

## Phase 5: User Story 3 — Suggestion Agent (Priority: P3)

**Independent Test**: Emits Top‑K suggestions with |Δp| reasons for uncertain symptoms

- [ ] T030 [P] [US3] Implement `src/suggestion/voi.py` (value‑of‑evidence)
- [ ] T031 [US3] Implement `src/agents/suggestion_agent.py` (Top‑K emission)

---

## Phase 6: User Story 4 — Evaluation Agent (Priority: P3)

**Independent Test**: Writes `val_metrics.json`, `test_metrics.json`, `artifacts/calibration.json`

- [ ] T032 [P] [US4] Implement `src/eval/metrics.py` (per‑symptom P/R/F1, negation precision)
- [ ] T033 [P] [US4] Implement `src/eval/calibration.py` (temperature + thresholds fit on dev)
- [ ] T034 [US4] Implement `src/eval/report.py` (writers for evaluation artifacts)
- [ ] T035 [US4] Implement `src/agents/evaluation_agent.py`
- [ ] T036 Add `scripts/check_gates.sh` to enforce acceptance thresholds (fail on breach)
- [ ] T037 Integrate gating script into README and pipeline runbook

---

## Phase 7: HPO (Evidence)

- [ ] T038 [P] Implement Optuna study per `configs/hpo/evidence_pairclf.yaml`
- [ ] T039 [P] Export best artifacts to `outputs/hpo/{study}/` (ckpt, config, val/test metrics)

---

## Phase 8: Pipeline Runner

- [ ] T040 Implement `src/pipeline/run_pipeline.py` (wire all agents, per‑post RunState)
- [ ] T041 Add CLI or Hydra main to run full split end‑to‑end

---

## Phase 9: Docs & README

- [ ] T042 [P] Document commands, outputs, and acceptance gates in `README.md`
- [ ] T043 [P] Add quickstart with sample dataset in `specs/001-four-agent-pipeline/quickstart.md`

---

## Dependencies & Execution Order

- Setup (Phase 1) → Foundational (Phase 2) → US1 → US2 → US3/US4 (parallel) → HPO → Runner → Docs
- US2 depends on Evidence predictions; US3 depends on Criteria features; US4 can run after predictions exist

## Parallel Opportunities

- Config authoring tasks (T009–T015)
- Evidence train/infer separation (T017–T018)
- Evaluation metrics vs calibration (T027–T028)
