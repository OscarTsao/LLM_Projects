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
make hpo      # run lightweight HPO for evidence model
make hpo-best # run robust two-stage Optuna HPO (Stage A + B)
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

HPO quickstart:

```bash
make hpo
# Best artifacts exported to outputs/hpo/<study_name>/{best.ckpt,best_config.yaml,val_metrics.json,test_metrics.json}

# Two-stage Optuna HPO (coarse + fine)
make hpo-best TRIALS_A=200 EPOCHS_A=5 TRIALS_B=60 EPOCHS_B=12 K_TOP=5 MODEL=microsoft/deberta-v3-base SEED=42
python -m src.hpo.hpo_driver --retrain-best --retrain-seeds 3  # optional multi-seed retrain
```

GPU training (Hugging Face backend):

```bash
# In the dev container (requires host NVIDIA + Docker w/ --gpus=all)
# Devcontainer is configured to request GPU and install CUDA-enabled PyTorch (CUDA 12.6 wheels) when available.

# Enable HF training backend for HPO runs
USE_HF_TRAIN=1 python -m src.hpo.hpo_driver --trials-a 5 --epochs-a 2 --trials-b 2 --epochs-b 2 --k-top 1

# Or set environment once in your shell
export USE_HF_TRAIN=1
make hpo-best-hf TRIALS_A=20 EPOCHS_A=2 TRIALS_B=10 EPOCHS_B=4 K_TOP=2
```


## Notes

- This is a **research** codebase template; not medical advice or a diagnostic tool.
- Avoid DSM verbatim content; use your own rule descriptions and public terminology mappings.
- Replace the memorised classifier with a real model before using on external data.
- `make clean` removes generated outputs (`outputs/`, `artifacts/`, `mlruns/`).
