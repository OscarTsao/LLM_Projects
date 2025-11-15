# Data Model: Configs and Reports

## TrialConfig (Hydra-resolved)

- `model.encoder`: str (Hugging Face model id)
- `model.criteria_head`: {type, hidden_dim, dropout}
- `model.evidence_head`: {type, params}
- `criteria.threshold_strategy`: `global` | `per_class` (default `per_class`)
- `criteria.thresholds`: list[float], len=C, each in [0.0, 1.0] (default all 0.5)
- `evidence.null_threshold`: float in [0.0, 1.0] (default 0.5)
- `evidence.min_span_score`: float in [0.0, 1.0] (default 0.0)
- `hpo.tune_thresholds`: bool (default true)
- `hpo.metric`: str (e.g., `macro_f1`)
- `hpo.directions`: list[str]
 - `ui.progress`: bool (default true)
 - `ui.stdout_level`: `INFO|DEBUG|WARNING` (default `INFO`)

## HPO Search Space (Optuna)

- If `criteria.threshold_strategy==global`:
  - `global_threshold ~ Uniform(0.30, 0.90)` → broadcast to all classes
- If `criteria.threshold_strategy==per_class`:
  - For c in 1..C: `criteria.thresholds[c] ~ Uniform(0.30, 0.90)`
- Evidence (if applicable):
  - `evidence.null_threshold ~ Uniform(0.30, 0.95)`
  - `evidence.min_span_score ~ Uniform(0.00, 0.50)`

## EvaluationReport (per-trial JSON)

- `trial_id`: str
- `generated_at`: ISO8601 timestamp
- `config`: TrialConfig snapshot
- `optimization_metric_name`: str
- `best_validation_score`: float
- `decision_thresholds`:
  - `criteria`: list[float] (len=C)
  - `evidence`: { `null_threshold`: float, `min_span_score`: float }
- `test_metrics`:
  - `criteria_matching`:
    - `macro_f1`: float
    - `macro_pr_auc`: float
    - `per_criterion`: array of objects:
      - `id`: string
      - `f1`: float
      - `precision`: float
      - `recall`: float
      - `pr_auc`: float
      - `confusion_matrix`: [[tn, fp], [fn, tp]]
  - `evidence_binding`: token/sentence-level F1/precision/recall, exact match
