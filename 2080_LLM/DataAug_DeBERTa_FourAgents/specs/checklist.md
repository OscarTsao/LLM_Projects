# Acceptance Checklist
_Last updated: 2025-10-17 09:15:49 UTC_

- [ ] Reproducible GroupKFold splits persisted with seed.
- [ ] Evidence macro‑F1 (present) ≥ baseline+10 and negation precision ≥ 0.90.
- [ ] Criteria AUROC ≥ 0.80; ECE ≤ 0.05 after calibration.
- [ ] All "likely" decisions cite ≥1 present EvidenceUnit.
- [ ] HPO artifacts exist under `outputs/hpo/{study}/`:
      `best.ckpt`, `best_config.yaml`, `val_metrics.json`, `test_metrics.json`.
- [ ] Training artifacts exist under `outputs/training/{run}/`:
      `model.ckpt`, `config.yaml`, `val_metrics.json`.
- [ ] Evaluation artifacts exist under `outputs/evaluation/{run}/`:
      `predictions.jsonl`, `criteria.jsonl`, `test_metrics.json`.
- [ ] MLflow file store contains matching runs and artifacts (`./mlruns`).
- [ ] README documents commands, outputs, and limitations.
