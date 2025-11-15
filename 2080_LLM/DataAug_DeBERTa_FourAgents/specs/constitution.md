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

