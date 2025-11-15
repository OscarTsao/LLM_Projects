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
- `data/redsm5.yaml`: dataset paths, split files (persisted at `configs/data/splits/{train,dev,test}.jsonl`).
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
 - Add `scripts/check_gates.sh` to enforce acceptance thresholds from the constitution; fail CI/pipeline when unmet.

## Runbook
- **HPO:** `python -m src.evidence.train_pairclf +hpo.enable=true hpo.study_name=evidence_hpo`
- **Pipeline:** `python -m src.pipeline.run_pipeline evidence.ckpt_path=outputs/hpo/evidence_hpo/best.ckpt`
- **Evaluation:** `python -m src.eval.report`

## Future (not in MVP)
- Conversation loop (Doctor/Patient) and LangGraph orchestration; POMDP/GNN Suggestion Agent.
