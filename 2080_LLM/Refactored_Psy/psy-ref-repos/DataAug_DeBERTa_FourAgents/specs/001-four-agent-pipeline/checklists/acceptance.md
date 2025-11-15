# Acceptance Checklist: four-agent-pipeline

**Purpose**: Validate end-to-end outputs and gates before merge/release
**Created**: 2025-10-17
**Feature**: specs/001-four-agent-pipeline/spec.md

## Data & Reproducibility
- [ ] CHK001 Splits persisted at `configs/data/splits/{train,dev,test}.jsonl`
- [ ] CHK002 Splits are deterministic with fixed seed (re-run check matches files)
- [ ] CHK003 Research-only data used; no PHI; sources are public/de-identified

## Evidence Outputs
- [ ] CHK010 `outputs/evaluation/{run}/predictions.jsonl` exists and matches `schemas/predictions.schema.json`
- [ ] CHK011 EvidenceUnit IDs are deterministic and unique per `(post_id, sentence_id, symptom)`

## Criteria Outputs
- [ ] CHK020 `outputs/evaluation/{run}/criteria.jsonl` exists and matches `schemas/criteria.schema.json`
- [ ] CHK021 100% of "likely" decisions cite ≥1 present EvidenceUnit

## Training & HPO Artifacts
- [ ] CHK030 `outputs/training/{run}/model.ckpt`, `config.yaml`, `val_metrics.json` exist
- [ ] CHK031 `outputs/hpo/{study}/best.ckpt`, `best_config.yaml`, `val_metrics.json`, `test_metrics.json` exist

## Calibration & Evaluation
- [ ] CHK040 `artifacts/calibration.json` exists (temperature + thresholds)
- [ ] CHK041 `outputs/evaluation/{run}/test_metrics.json` and `val_metrics.json` exist

## Tracking
- [ ] CHK050 MLflow tracking URI is `file:./mlruns` and runs/artifacts are present
- [ ] CHK051 Output run IDs correspond to MLflow runs (params/metrics/artifacts aligned)

## Acceptance Gates (Constitution)
- [ ] CHK060 Evidence macro‑F1 (present) ≥ baseline + 10 points
- [ ] CHK061 Negation precision ≥ 0.90
- [ ] CHK062 Criteria AUROC ≥ 0.80; ECE ≤ 0.05 after calibration
- [ ] CHK063 Gate check script passes: `scripts/check_gates.sh` returns exit code 0

## Documentation
- [ ] CHK070 README documents commands, outputs structure, and acceptance gates

