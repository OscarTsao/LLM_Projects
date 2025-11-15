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
- Negative class policy: only annotated zeros are explicit absences; unmentioned (sentence, symptom) pairs are unknown (not negatives).
- **Symptoms (initial set)**: DEPRESSED_MOOD, ANHEDONIA, APPETITE_CHANGE, SLEEP_ISSUES, PSYCHOMOTOR, FATIGUE, WORTHLESSNESS, COGNITIVE_ISSUES, SUICIDAL_THOUGHTS.
- **Splits**: GroupKFold 80/10/10 by `post_id`, persisted to disk under `configs/data/splits/{train,dev,test}.jsonl` (deterministic with fixed seed).

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
- Compute deterministic EvidenceUnit IDs (EU IDs) from `(post_id, sentence_id, symptom)` and ensure uniqueness for downstream citation.

### Criteria Agent
- Aggregates EvidenceUnits into features (max score per symptom, any_present/absent, conflicts).
- JSON rule counts + small classifier → provisional probability `p_dx`.
- Temperature scaling and per‑symptom thresholds from `artifacts/calibration.json` (if present).
- Writes `criteria.jsonl` with supporting EvidenceUnit ids.
- Validate that every "likely" decision cites ≥1 present EvidenceUnit; fail validation otherwise.

### Suggestion Agent
- Value‑of‑evidence (|Δp|) counterfactuals for uncertain symptoms (configurable band).
- Emits Top‑K suggestions per post with reason strings.

### Evaluation Agent
- Evidence metrics: per‑symptom P/R/F1 (present), negation precision.
- Criteria metrics: AUROC/F1; ECE; contradiction rate; faithfulness %.
- Fits calibration (temperature + thresholds) on dev; writes `artifacts/calibration.json`.
- Writes `val_metrics.json` and `test_metrics.json` to the appropriate output directories.
- Enforce acceptance gates via `scripts/check_gates.sh` (fails pipeline/CI if thresholds unmet).

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

## Clarifications
### Session 2025-10-17
1. Negative class policy: Only annotated zeros are explicit absence; unmentioned pairs are unknown and not treated as negatives.
2. Splitting: GroupKFold 80/10/10 by post_id; splits persisted to disk for reproducibility.
3. Evidence granularity: Sentence‑level with sentence text as provenance; EvidenceUnit ids must be unique and traceable.
4. Calibration: Post‑hoc temperature scaling on criteria probability, plus per‑symptom decision thresholds optimized on dev.
5. HPO objective: macro_F1_present + 0.2*neg_precision − 0.5*ECE (dev split).
6. HPO search space (Evidence): model_name ∈ {deberta‑v3‑base, deberta‑v3‑large, PubMedBERT, ClinicalBERT}; lr ∈ [1e‑5, 6e‑5]; dropout ∈ [0.0, 0.3]; max_len ∈ {192,256,384}; loss ∈ {CE, Focal}; pos_weight ∈ [0.5, 3.0].
7. Tracking: mlflow.set_tracking_uri("file:./mlruns"); log params, metrics, and artifacts for each run.
8. Output folders: strictly enforced as specified in Outputs (Files).
9. Acceptance tests: Use acceptance checklist; failing any gate blocks merge.
