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
