# Quickstart: Threshold Tuning with HPO

## Minimal HPO run (toy budget)

Use Hydra overrides to enable threshold tuning and pick the optimization metric:

```
python -m src.cli.train \
  mode=hpo \
  model.encoder=bert-base-uncased \
  model.criteria_head=mlp \
  model.evidence_head=start_end_linear \
  criteria.threshold_strategy=per_class \
  hpo.tune_thresholds=true \
  hpo.metric=macro_f1 \
  ui.progress=true \
  ui.stdout_level=INFO \
  hpo.n_trials=20
```

Notes:
- Per-class thresholds are tuned in [0.30, 0.90]; evidence null/min-span thresholds tuned if applicable.
- Best thresholds are recorded in MLflow params and in the EvaluationReport JSON.
- Training/HPO will display trial and epoch progress via tqdm and periodic stdout status lines.
