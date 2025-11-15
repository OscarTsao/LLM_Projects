# Research Notes — 5-Fold DeBERTaV3 Evidence Binding

## Topics & Decisions

- CV splitting with group stratification:
  - Prefer scikit-learn's `StratifiedGroupKFold` (sklearn>=1.3). Feature gate detection via `hasattr(StratifiedGroupKFold, "split")` and raise actionable error if missing.
  - Fallback order: (1) `iterstrat.MultilabelStratifiedKFold` on per-post aggregates, (2) vanilla `GroupKFold` followed by rejection sampling + balancing heuristics (swap samples between folds until class deltas ≤ 2%).
  - Groups: `post_id`; target: binary label. Persist per-fold summaries (positives/negatives, % truncated sequences, neg_sampling_strategy, seed) to `outputs/manifests/folds.json` so regressions are diffable.
  - Guardrail: if any fold drops below 10 samples, log warning and abort unless `allow_small_folds=true` override provided.
- Optimizer:
  - Use HF Trainer with `optim=adamw_torch_fused`; fallback to `adamw_torch` if
    fused unsupported on platform/PyTorch build.
- Fine-tuning:
  - RESOLVED — Full fine‑tune of all Transformer layers (no backbone freezing
    by default). Optionally expose config to freeze for low‑VRAM scenarios.
- Loss & Imbalance:
  - Default weighted cross-entropy (inverse-frequency weights per fold).
  - Optional Focal Loss (γ=2.0; α derived from class frequencies) via Hydra flag.
  - Class weights pulled from manifest stats; if manifest missing fields fall back to dataset scan.
- Metrics:
  - Accuracy, macro-F1, positive-class F1, ROC-AUC, PR-AUC.
  - Log per-fold; aggregate mean/std in parent MLflow run.
  - Selection: Macro-F1 is the primary model selection metric (resolved).
- Aggregation & reporting:
  - Parent MLflow run stores `cv_summary.json`, ROC/PR plots (per fold + aggregate), and confusion matrices.
  - Use shared metrics module to avoid drift between Trainer callbacks and aggregation; ensure identical threshold policy (p>0.5) across callers.
  - Aggregation script rerunnable without retraining; reads child run IDs from manifest.
  - Tie-breaking strategy: select lowest fold index among folds sharing best Macro-F1; log reasoning as MLflow tag `best_fold_selection`.
- Precision:
  - RESOLVED — Prefer BF16; fall back to FP16; else FP32. Use HF Trainer
    `bf16`/`fp16` flags and log selected mode.
 - Scheduler:
   - RESOLVED — Linear LR schedule with warmup_ratio=0.06 via TrainingArguments.
 - HPO:
   - RESOLVED — Not in scope for this feature (fixed hyperparameters). If added
     later, use Optuna with `sqlite:///optuna.db` and MLflow logging.
- NSP-style inputs:
  - Use HF tokenizers with sentence-pair encoding: `encode_plus(text=criterion,
    text_pair=sentence, truncation=True, padding)`. Preserve max_length and
    special tokens per model.
- Inference surfaces:
  - CLI + library function load model/tokenizer from MLflow artifacts.
  - Output includes label, probability, model run ID, precision mode, config digest, and the manifest hash to trace datasets.
  - Optionally log inference run (nested) for provenance.
- Data validation:
  - Pre-flight checks ensure all required columns exist (`post_id`, `sentence_id`, `sentence_text`, `DSM5_symptom`, `status`, `criterion_id`).
  - Validate dataset cardinalities vs manifests (e.g., positive count matches annotations). Fail-fast before tokenization if mismatched.
- Failure handling:
  - Any fold-level OOM/CUDA/Trainer failure aborts the entire CV run. Record failure_reason tag and propagate exception (no retries).
  - Missing HF checkpoint/network hiccups: surface actionable error and suggest running `transformers-cli login` / offline model cache.
- Logging & observability:
  - Use DEBUG logging for batch-level loss + LR; INFO for fold boundaries and artifact locations.
  - MLflow tags: `precision_mode`, `optimizer_used`, `neg_ratio`, `split_seed`, `manifest_sha1`.
- Sample identity & manifests:
  - Canonical ID = natural composite `(post_id, sentence_id, criterion_id)`; store as tuple and deterministic SHA1 to detect duplicates.
  - Each manifest row records `neg_sampling_strategy`, `source_label` (original vs synthetic), checksum of tokenized ids, and fold index to debug leakage quickly.
- Reproducibility:
  - Log Hydra config, seeds, `pip freeze`, git SHA, data manifest (filenames +
    checksums). Set deterministic flags where feasible.
 - Negative sampling:
   - RESOLVED — Stratified random negatives to achieve a 1:3 pos:neg ratio at
     dataset construction; maintain grouping by `post_id` for CV.

## Open Items

- Validate availability/licensing of `iterstrat` for fallback (use vendored helper if packaging is an issue).
- Decide if inference CLI should accept batch files (CSV) in addition to single pair; default to single for MVP.
- Determine whether manifest validation should compute per-fold token-length histograms (nice-to-have for QA).
- Tokenization decision: RESOLVED — max_length=512; truncation=longest_first.
