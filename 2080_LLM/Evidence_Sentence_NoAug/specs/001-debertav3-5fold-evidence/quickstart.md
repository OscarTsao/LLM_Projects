# Quickstart — 5-Fold DeBERTaV3 Evidence Binding

## 0. Prereqs

1. Python 3.10+ with virtualenv
2. Install project in editable mode plus dev deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e '.[dev]'
   ```
3. Ensure data present:
   - Criteria JSON: `data/DSM5/*.json` (mirrors `data/data/DSM5/` legacy layout)
   - Posts + annotations: `data/redsm5/posts.csv`, `data/redsm5/annotations.csv`
4. Launch MLflow UI (optional but recommended) in a separate terminal:
   ```bash
   mlflow ui \
     --backend-store-uri sqlite:///mlflow.db \
     --default-artifact-root ./mlruns
   ```

## 1. Data manifest sanity check

Run the dataset builder in dry-run mode (to be added in `scripts/train_cv.py --dry-run`) or via the forthcoming utility script:

```bash
python scripts/train_cv.py \
  +pipeline.only_build_manifests=true \
  data.posts=data/redsm5/posts.csv \
  data.criteria_dir=data/DSM5 \
  data.neg_strategy=stratified data.neg_ratio=3 \
  cv.seed=1337
```

Expected artifacts:
- `outputs/datasets/evidence_pairs.parquet` with canonical `(post_id, sentence_id, criterion_id)` keys
- `outputs/manifests/folds.json` capturing per-fold counts, strategy, and deterministic seed

## 2. Train 5-fold DeBERTaV3 (Hydra CLI)

```bash
python scripts/train_cv.py \
  model.name=microsoft/deberta-v3-base \
  data.posts=data/redsm5/posts.csv \
  data.criteria_dir=data/DSM5 \
  data.neg_strategy=stratified data.neg_ratio=3 \
  trainer.max_epochs=3 \
  trainer.train_batch_size=16 \
  trainer.eval_batch_size=32 \
  trainer.optim=adamw_torch_fused \
  trainer.metric_for_best_model=f1_macro trainer.greater_is_better=true \
  precision.bf16=true \
  scheduler.type=linear scheduler.warmup_ratio=0.06 \
  cv.folds=5 cv.group=post_id \
  loss.name=weighted_ce
```

Key toggles:
- Fused AdamW fallback: override `trainer.optim=adamw_torch` if fused kernels unavailable.
- Precision: use `precision.fp16=true` for Volta/Turing; omit for FP32.
- Optional focal loss: `loss.name=focal loss.gamma=2.0`.

Outputs:
- MLflow parent run with five nested child runs (one per fold)
- Checkpoints under `mlruns/<parent>/<child>/artifacts/best_model`
- Aggregate summary `outputs/metrics/cv_summary.json` + ROC/PR/Confusion plots logged to parent run

## 3. Review aggregate metrics

```bash
python scripts/aggregate_cv.py \
  cv.summary_path=outputs/metrics/cv_summary.json \
  mlflow.parent_run_id=<RUN_ID>
```

Use MLflow UI to inspect:
- `cv_summary.json` (mean/std accuracy, macro F1, ROC-AUC, PR-AUC)
- Fold-specific metrics + confusion matrices
- Logged precision mode (`bf16`/`fp16`/`fp32`) per run

## 4. Inference smoke test

```bash
python scripts/infer_pair.py \
  --criterion "DSM-5 MDD A.1 Depressed mood most of the day" \
  --sentence "I haven't enjoyed anything for weeks." \
  model.uri=mlruns/<parent_run>/<best_child>/artifacts/best_model \
  tokenizer.uri=mlruns/<parent_run>/<best_child>/artifacts/tokenizer
```

Expected output:
```
label=1 probability=0.83 run_id=<best_child> precision=bf16
```

## 5. Troubleshooting

- Disable heavy optimizations quickly: `precision.bf16=false precision.fp16=false trainer.fp32=true`.
- Rebuild manifests if data changes: rerun step 1 with incremented `cv.seed`.
- If GPU OOM occurs, lower `trainer.train_batch_size` or add `trainer.gradient_accumulation_steps=2`.
- When running on CPU, set `trainer.use_gpu=false precision.fp32_only=true` to skip AMP setup.
- Aggregation-only rerun (no retraining): `python scripts/aggregate_cv.py --parent-run-id <RUN_ID> --manifests outputs/manifests/folds.json`.
- Manifest diff check: `python -m scripts.tools.compare_manifests outputs/manifests/folds.json outputs/manifests/folds_prev.json` (tool to be added) to confirm identical seeds before reusing metrics.

## 6. Validation checklist before handoff

1. `pytest tests/unit tests/integration -m "not gpu"` passes using fixtures.
2. `ruff check src tests scripts` and `mypy src` succeed.
3. MLflow UI displays parent + child runs, `cv_summary.json`, ROC/PR plots, and model artifacts.
4. Inference smoke test prints label/probability and logs inference metadata (if enabled).
5. Quickstart commands in this file were executed on a clean environment (record run IDs in spec appendix).
