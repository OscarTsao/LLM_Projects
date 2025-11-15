# Spec Kit Prompt Script (Paste into your coding agent)

## 1) Constitution
/speckit.constitution
# Project Constitution — Four-Agent Psychiatric Evidence Pipeline
_Last updated: 2025-10-17 09:15:49 UTC_

## Purpose & Scope
This repository implements a **non‑conversational four‑agent pipeline** for psychiatric evidence processing over sentence‑labeled texts (e.g., ReDSM5). The four agents are:
1. **Evidence Agent** — classifies each (sentence, symptom) as present/absent with confidence and provenance.
2. **Criteria Agent** — aggregates evidence into a calibrated provisional probability and decision.
3. **Suggestion Agent** — recommends next most informative symptoms to verify (value‑of‑evidence).
4. **Evaluation Agent** — evaluates metrics, fits calibration/thresholds, checks faithfulness, and emits feedback.

**Research‑only**; not a clinical diagnostic tool. No DSM verbatim content is embedded. Criteria logic is implemented as our own JSON policies and simple classifiers.

## Non‑Negotiable Principles
- **Safety & Ethics:** research use only; no crisis routing in this MVP. No PHI; only public/de‑identified data.
- **Transparency:** every criteria decision must cite supporting EvidenceUnits (sentence‑level quotes). If not supported, mark the decision as *uncertain*.
- **Reproducibility:** fixed random seeds; GroupKFold by `post_id`; Hydra‑managed configurations; MLflow (file backend) logging.
- **Outputs (mandatory layout):**
  - HPO: `outputs/hpo/{study_name}/`
    - `best.ckpt`, `best_config.yaml`, `val_metrics.json`, `test_metrics.json`
  - Training: `outputs/training/{run_id}/`
    - `model.ckpt`, `config.yaml`, `val_metrics.json`
  - Evaluation: `outputs/evaluation/{run_id}/`
    - `predictions.jsonl`, `criteria.jsonl`, `test_metrics.json`
  - Calibration artifacts: `artifacts/calibration.json`
- **Tracking:** `mlflow.set_tracking_uri("file:./mlruns")` is required.
- **Quality Gates:**
  - Evidence macro‑F1 (present) >= target baseline + 10 points.
  - Negation precision ≥ 0.90.
  - Criteria AUROC ≥ 0.80; ECE ≤ 0.05 after temperature scaling.
  - 100% of "likely" decisions have at least one supporting present EvidenceUnit.
- **Acceptance:** changes must pass analysis & checklist before merging to `main`.

## Out of Scope (MVP)
Conversational interviewing, risk triage, multimodal inputs, DSM verbatim text, and deployment.



## 2) Functional Spec
/speckit.specify
# Functional Specification — Four-Agent Pipeline
_Last updated: 2025-10-17 09:15:49 UTC_

## Feature
**Name:** `four-agent-pipeline` (non‑conversational MVP)

## Objective
Given sentence‑labeled posts, produce:
1) sentence‑level **EvidenceUnits** (present/absent + confidence + provenance),
2) a calibrated **CriteriaResult** with decision and cited support,
3) **Top‑K suggestions** for which symptoms to verify next,
4) **Evaluation** metrics, calibration artifacts, and structured prediction outputs.

## Inputs
- Dataset in JSONL or CSV resolved to in‑memory objects with:
  - `post_id: str`
  - `sentences: List[{"sentence_id": str, "text": str}]`
  - `labels: List[{"sentence_id": str, "symptom": str, "status": int}]`  // 1=present, 0=explicit-absent
- **Symptoms (initial set)**: DEPRESSED_MOOD, ANHEDONIA, APPETITE_CHANGE, SLEEP_ISSUES, PSYCHOMOTOR, FATIGUE, WORTHLESSNESS, COGNITIVE_ISSUES, SUICIDAL_THOUGHTS.
- **Splits**: GroupKFold 80/10/10 by `post_id`, persisted to disk.

## Outputs (Files)
- **HPO:** `outputs/hpo/{study_name}/best.ckpt`, `best_config.yaml`, `val_metrics.json`, `test_metrics.json`
- **Training:** `outputs/training/{run_id}/model.ckpt`, `config.yaml`, `val_metrics.json`
- **Evaluation:** `outputs/evaluation/{run_id}/predictions.jsonl`, `criteria.jsonl`, `test_metrics.json`
- **Calibration:** `artifacts/calibration.json` (temperature + per‑symptom thresholds)

## Output Schemas
- **predictions.jsonl** (per (post, sentence, symptom)):
  ```json
  {"post_id":"P1","sentence_id":"S3","symptom":"SLEEP_ISSUES","assertion":"present","score":0.82,"gold":1}
  ```
- **criteria.jsonl** (per post):
  ```json
  {"post_id":"P1","p_dx":0.67,"decision":"likely",
    "supporting":{"DEPRESSED_MOOD":["EU_12"],"ANHEDONIA":["EU_19"]},
    "conflicts":["APPETITE_CHANGE"],"missing":["SLEEP_ISSUES"]}
  ```

## Functional Requirements
### Evidence Agent
- Pairwise cross‑encoder over `(sentence, symptom)` → `present/absent` probability.
- Writes `predictions.jsonl` with required schema.
- Hydra‑driven training/inference; Optuna HPO on model family and hyperparameters.

### Criteria Agent
- Aggregates EvidenceUnits into features (max score per symptom, any_present/absent, conflicts).
- JSON rule counts + small classifier → provisional probability `p_dx`.
- Temperature scaling and per‑symptom thresholds from `artifacts/calibration.json` (if present).
- Writes `criteria.jsonl` with supporting EvidenceUnit ids.

### Suggestion Agent
- Value‑of‑evidence (|Δp|) counterfactuals for uncertain symptoms (configurable band).
- Emits Top‑K suggestions per post with reason strings.

### Evaluation Agent
- Evidence metrics: per‑symptom P/R/F1 (present), negation precision.
- Criteria metrics: AUROC/F1; ECE; contradiction rate; faithfulness %.
- Fits calibration (temperature + thresholds) on dev; writes `artifacts/calibration.json`.
- Writes `val_metrics.json` and `test_metrics.json` to the appropriate output directories.

## Non‑Functional Requirements
- Reproducibility: seeds, deterministic splits persisted.
- Configurability: Hydra across all components.
- Observability: MLflow logs params, metrics, and artifacts to `./mlruns`.

## Success Criteria
- Evidence macro‑F1 (present) improved ≥10 points over rules baseline.
- Negation precision ≥ 0.90.
- Criteria AUROC ≥ 0.80; ECE ≤ 0.05 after calibration.
- 100% "likely" decisions are grounded in ≥1 present EvidenceUnit.

## Risks & Mitigations
- **Data scarcity:** start with public sample; later expand via gated dataset access.
- **Calibration drift:** run Evaluation after any Evidence retrain; refresh `artifacts/calibration.json`.
- **Label imbalance:** focal loss / class weights; per‑symptom thresholds.



## 3) Clarify
/speckit.clarify
# Clarifications
_Last updated: 2025-10-17 09:15:49 UTC_

1. **Negative class policy:** Only use annotated zeros as explicit absence; unmentioned (sentence, symptom) pairs are unknown and NOT treated as negatives.
2. **Splitting:** GroupKFold 80/10/10 by `post_id`; splits persisted to disk for reproducibility.
3. **Evidence granularity:** Sentence‑level (the sentence text is the quote/provenance). EvidenceUnit ids must be unique and traceable.
4. **Calibration:** Post‑hoc temperature scaling on the criteria probability, plus per‑symptom decision thresholds optimizing macro‑F1 or J‑statistic on dev.
5. **HPO objective:** `macro_F1_present + 0.2*neg_precision − 0.5*ECE` on the dev split.
6. **HPO search space:** `model_name ∈ {deberta‑v3‑base, deberta‑v3‑large, PubMedBERT, ClinicalBERT}`, `lr ∈ [1e‑5, 6e‑5]`, `dropout ∈ [0.0, 0.3]`, `max_len ∈ {192,256,384}`, `loss ∈ {CE, Focal}`, `pos_weight ∈ [0.5, 3.0]`.
7. **Tracking:** `mlflow.set_tracking_uri("file:./mlruns")` required; each run logs params, metrics, prediction/criteria artifacts.
8. **Output folders:** strictly enforced:
   - `outputs/hpo/{study}/` — best ckpt/config/metrics
   - `outputs/training/{run}/` — training ckpt/config/val metrics
   - `outputs/evaluation/{run}/` — predictions/criteria/test metrics
9. **Acceptance tests:** see checklist; failing any gate blocks merge.


## 4) Technical Plan
/speckit.plan
# Technical Plan
_Last updated: 2025-10-17 09:15:49 UTC_

## Stack
Python 3.10+, PyTorch, HuggingFace Transformers, Hydra, Optuna, MLflow (file backend), numpy/pandas, scikit‑learn.

## Architecture
Linear pipeline with typed contracts:
- Evidence → Criteria → Suggestion → Evaluation
- Shared artifacts: `artifacts/calibration.json`

## Data Model (Dataclasses)
- **EvidenceUnit**: `eu_id, post_id, sentence_id, sentence, symptom, assertion, score`
- **CriteriaResult**: `post_id, p_dx, decision, supporting{symptom→[eu_id]}, conflicts[], missing[]`
- **Suggestion**: `post_id, ranked:[{symptom, delta_p, reason}]`

## Directory Layout
```
psy-msa/
  .specify/
  configs/
    data/redsm5.yaml
    evidence/pairclf.yaml
    criteria/aggregator.yaml
    suggest/voi.yaml
    eval/default.yaml
    pipeline/default.yaml
    hpo/evidence_pairclf.yaml
  src/
    schema/types.py
    agents/{evidence_agent.py,criteria_agent.py,suggestion_agent.py,evaluation_agent.py}
    evidence/{train_pairclf.py,infer_pairclf.py}
    criteria/aggregate.py
    suggestion/voi.py
    eval/{metrics.py,calibration.py,report.py}
    pipeline/run_pipeline.py
    utils/{hydra_mlflow.py,seed.py,io.py}
  scripts/{run_pipeline.sh,run_hpo_evidence.sh}
  outputs/{hpo,training,evaluation}/
  mlruns/
  artifacts/
```

## Hydra Configuration Tree
- `data/redsm5.yaml`: dataset paths, split files.
- `evidence/pairclf.yaml`: model, optimizer, loss, trainer knobs.
- `criteria/aggregator.yaml`: symptoms, key symptoms, thresholds path, temperature.
- `suggest/voi.yaml`: top_k, uncertain band.
- `eval/default.yaml`: primary metrics, gates.
- `pipeline/default.yaml`: orchestrator wiring, output paths, mlflow URI.
- `hpo/evidence_pairclf.yaml`: search space & study params.

## HPO
- Optuna TPE + ASHA pruner; study per component (start with Evidence).
- Objective: `macro_F1_present + 0.2*neg_precision − 0.5*ECE`.
- Export best: `outputs/hpo/{study}/best.ckpt`, `best_config.yaml`, `val_metrics.json`, `test_metrics.json`.

## Evaluation
- Evidence: per‑symptom P/R/F1 (present), negation precision.
- Criteria: AUROC/F1, ECE, contradictions, faithfulness.
- Calibration fitting on dev → write `artifacts/calibration.json`.
- Writers produce evaluation JSONL and metrics JSON files.

## Runbook
- **HPO:** `python -m src.evidence.train_pairclf +hpo.enable=true hpo.study_name=evidence_hpo`
- **Pipeline:** `python -m src.pipeline.run_pipeline evidence.ckpt_path=outputs/hpo/evidence_hpo/best.ckpt`
- **Evaluation:** `python -m src.eval.report`

## Future (not in MVP)
- Conversation loop (Doctor/Patient) and LangGraph orchestration; POMDP/GNN Suggestion Agent.


## 5) Tasks
/speckit.tasks
# Tasks
_Last updated: 2025-10-17 09:15:49 UTC_

1. **Scaffold repository**
   - Create directories, `pyproject.toml`, pre‑commit hooks.
   - _Done when_ tree matches plan; pre‑commit runs.

2. **Create Hydra configs**
   - Files under `configs/` with sensible defaults and output paths.
   - _Done when_ configs validate and load; defaults compose.

3. **Data loader & splits**
   - Parse dataset to (post, sentences, labels). Implement GroupKFold by `post_id` and persist splits.
   - _Done when_ unit tests confirm reproducibility.

4. **Evidence Agent (train & infer)**
   - Pairwise model; CE/Focal toggle; AMP; early stopping.
   - Inference writes `outputs/evaluation/{run}/predictions.jsonl` (dev/test).
   - _Done when_ MLflow logs runs; schema tests pass.

5. **Criteria Agent**
   - Aggregation + JSON rule counts + logistic classifier; temperature scaling.
   - Writes `criteria.jsonl`; consumes `artifacts/calibration.json` if present.
   - _Done when_ decisions cite supporting evidence; calibration applied.

6. **Suggestion Agent (VOE)**
   - Δp counterfactuals for uncertain symptoms; Top‑K reasons.
   - _Done when_ Top‑K sorted by delta; unit tests for consistency.

7. **Evaluation Agent**
   - Metrics, faithfulness, contradictions; calibration fitting.
   - Writes `val_metrics.json`, `test_metrics.json`; saves `artifacts/calibration.json`.
   - _Done when_ gates computed; artifacts present.

8. **HPO**
   - Optuna study; MLflow logging; export best artifacts to `outputs/hpo/{study}/`.
   - _Done when_ the four required files exist and match MLflow run params.

9. **Pipeline Runner**
   - Wire agents; per‑post RunState; end‑to‑end execution.
   - _Done when_ single command processes a full split and writes outputs.

10. **Docs & README**
   - Usage, commands, output structure, limitations.
   - _Done when_ reviewer can reproduce end‑to‑end with sample data.


## 6) Analyze & Verify
/speckit.analyze
Please verify that every requirement in spec.md maps to a plan section, a task, and a concrete artifact. Confirm output directories and filenames match across documents. Confirm MLflow file backend.

## 7) Checklist
/speckit.checklist
# Acceptance Checklist
_Last updated: 2025-10-17 09:15:49 UTC_

- [ ] Reproducible GroupKFold splits persisted with seed.
- [ ] Evidence macro‑F1 (present) ≥ baseline+10 and negation precision ≥ 0.90.
- [ ] Criteria AUROC ≥ 0.80; ECE ≤ 0.05 after calibration.
- [ ] All "likely" decisions cite ≥1 present EvidenceUnit.
- [ ] HPO artifacts exist under `outputs/hpo/{study}/`:
      `best.ckpt`, `best_config.yaml`, `val_metrics.json`, `test_metrics.json`.
- [ ] Training artifacts exist under `outputs/training/{run}/`:
      `model.ckpt`, `config.yaml`, `val_metrics.json`.
- [ ] Evaluation artifacts exist under `outputs/evaluation/{run}/`:
      `predictions.jsonl`, `criteria.jsonl`, `test_metrics.json`.
- [ ] MLflow file store contains matching runs and artifacts (`./mlruns`).
- [ ] README documents commands, outputs, and limitations.


## 8) Implement
/speckit.implement
Implement according to the plan and tasks. Use Hydra for configs, Optuna for HPO, and MLflow file DB (`file:./mlruns`) for tracking. Ensure the following artifacts are written exactly:
- HPO: outputs/hpo/{study_name}/best.ckpt, best_config.yaml, val_metrics.json, test_metrics.json
- Training: outputs/training/{run_id}/model.ckpt, config.yaml, val_metrics.json
- Evaluation: outputs/evaluation/{run_id}/predictions.jsonl, criteria.jsonl, test_metrics.json
- Calibration: artifacts/calibration.json
