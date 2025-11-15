# Four-Agent Psychiatric Evidence Pipeline — Spec Pack

This package contains **Spec Kit‑style documentation** and **config templates** for a non‑conversational four‑agent pipeline:

- Evidence → Criteria → Suggestion → Evaluation
- Hydra for configuration, Optuna for HPO, MLflow (file DB) for tracking
- Strict output layout under `outputs/` and `artifacts/`

## What’s inside

- `.specify/` — Spec Kit artifacts
  - `constitution.md` — project guardrails and non‑negotiables
  - `spec.md` — functional specification
  - `clarifications.md` — resolved Q&A
  - `plan.md` — technical plan
  - `tasks.md` — implementation tasks
  - `analysis.md` — traceability & verification
  - `checklist.md` — acceptance gates
- `schemas/` — JSON Schemas for output files
  - `predictions.schema.json`
  - `criteria.schema.json`
- `configs_templates/` — Hydra config templates
  - `pipeline/default.yaml`
  - `evidence/pairclf.yaml`
  - `criteria/aggregator.yaml`
  - `suggest/voi.yaml`
  - `eval/default.yaml`
  - `hpo/evidence_pairclf.yaml`
- `spec_kit_prompt.md` — copy‑paste script for Spec Kit commands

## How to use

1. Open `spec_kit_prompt.md` and paste the blocks into your coding agent (Spec Kit).
2. Let the agent scaffold the project and implement per the tasks and plan.
3. Ensure MLflow tracking URI is `file:./mlruns` and outputs follow these directories:
   - HPO → `outputs/hpo/{study_name}/`
   - Training → `outputs/training/{run_id}/`
   - Evaluation → `outputs/evaluation/{run_id}/`
   - Calibration → `artifacts/calibration.json`

## Quickstart

```bash
python -m src.pipeline.run_pipeline --seed 42
```

Or use the Makefile helpers:

```bash
make splits   # regenerate GroupKFold-inspired splits
make run      # execute the full four-agent pipeline
make test     # run unit/integration tests (pytest)
make lint     # basic syntax compilation check
```

The command above will:

- Read the sample dataset in `data/redsm5_sample.jsonl`
- Train an evidence classifier (memorised baseline) and persist artifacts under `outputs/training/<run_id>/`
- Generate `predictions.jsonl`, `criteria.jsonl`, and metrics under `outputs/evaluation/<run_id>/`
- Produce `artifacts/calibration.json`
- Enforce constitution gates via `scripts/check_gates.py`
- Log a run to the local MLflow file store (`mlruns/`)

Key outputs:

- `outputs/evaluation/<run_id>/predictions.jsonl`
- `outputs/evaluation/<run_id>/criteria.jsonl`
- `outputs/evaluation/<run_id>/val_metrics.json`
- `artifacts/calibration.json`

To rerun gate validation manually:

```bash
scripts/check_gates.sh --metrics outputs/evaluation/<run_id>/test_metrics.json \
  --neg-precision-min 0.90 --criteria-auroc-min 0.80 --ece-max 0.05
```

## Notes

- This is a **research** codebase template; not medical advice or a diagnostic tool.
- Avoid DSM verbatim content; use your own rule descriptions and public terminology mappings.
- Replace the memorised classifier with a real model before using on external data.
- `make clean` removes generated outputs (`outputs/`, `artifacts/`, `mlruns/`).
