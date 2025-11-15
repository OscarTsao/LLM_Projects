# Analysis & Verification
_Last updated: 2025-10-17 09:15:49 UTC_

## Traceability Matrix
| Requirement | Plan Section | Task(s) | Artifact |
|---|---|---|---|
| Sentence‑level evidence with provenance | Architecture, Data Model | 3,4 | `predictions.jsonl` schema |
| Calibrated criteria probability & decision | Evaluation, HPO | 5,7 | `criteria.jsonl`, `artifacts/calibration.json`, `test_metrics.json` |
| Top‑K suggestions by |Δp| | Suggestion Agent | 6 | `suggestions` (embedded in criteria or separate summary) |
| HPO with Hydra+MLflow | HPO, Runbook | 8 | `outputs/hpo/{study}/*`, MLflow run |
| Faithfulness & contradictions | Evaluation | 7 | Metrics JSON; failure gate logs |

## Consistency Checks
- Output directories and filenames are consistent across Spec, Plan, Tasks.
- Hydra and MLflow usage are uniform; tracking URI fixed to file backend.
- Calibration artifacts are produced in Evaluation and consumed in Criteria.

## Optimizations
- Use ASHA pruning to cut long trials.
- Persist splits and best config to enable cheap re‑use.
- JSONL streaming writers to reduce memory footprint on large corpora.

## Open Items
- None for MVP. Conversation orchestration is planned for a later milestone.
