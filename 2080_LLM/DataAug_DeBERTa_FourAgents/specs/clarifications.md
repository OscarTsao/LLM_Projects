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
