# Data Model — Evidence Binding (NSP-Style)

## Entities

- Sample
  - Identity: natural composite `(post_id, sentence_id, criterion_id)`; hashed form stored as `sample_sha1` for manifest diffs.
  - Fields: `post_id`, `sentence_id`, `criterion_id`, `criterion_text`,
    `sentence_text`, `label` ∈ {0,1}, `source_label` ∈ {"annotated","neg_sampled"},
    `neg_sampling_strategy` (e.g., stratified_random), `token_checksum`, `truncated` (bool), `created_at`.
  - Notes: NSP-style input pairs are derived from criterion + sentence; deterministic identity avoids duplicate fold assignments.

- DatasetManifest
  - Fields: `manifest_sha1`, `generated_at`, `seed`, `neg_ratio`, `strategy`, `splitter` (StratifiedGroupKFold|IterStratFallback|GroupKFold), `positive_count`, `negative_count`, `source_files` (list of path + checksum).
  - Notes: Captures provenance for the entire dataset build; referenced by FoldSplit and CVSummary to guarantee reproducibility.

- FoldSplit
  - Fields: `sample_sha1`, `fold_index` ∈ {0..4}, `fold_name`, `is_validation`, `manifest_sha1`.
  - Metadata: `seed`, `grouping` = `post_id`, `strategy` ∈ {StratifiedGroupKFold, GroupKFold, IterStratFallback}, `pos_count`, `neg_count`, `truncation_rate`.

- ModelArtifact
  - Fields: `run_id`, `fold_index` (optional), `model_uri`, `tokenizer_files`,
    `config.json`, `metrics.json`, `precision_mode`, `best_metric`.
  - Notes: Stored via MLflow (artifacts under `mlruns/`) and referenced by inference CLI via `model_uri`.

- CVSummary
  - Fields: `parent_run_id`, `manifest_sha1`, `fold_metrics` (list of per-fold dicts), `best_fold_index`, `tie_break_reason`, `mean_metrics`, `std_metrics`, `artifact_path`.
  - Notes: Written as `outputs/metrics/cv_summary.json` and logged to parent MLflow run.

- InferenceRequest
  - Fields: `request_id`, `criterion_text`, `sentence_text`, `model_uri`, `tokenizer_uri`, `timestamp`, `run_context`, `manifest_sha1`.
  - Notes: Passed to inference helper/CLI; optionally logged to MLflow as parameters.

- InferenceResult
  - Fields: `request_id`, `label`, `probability`, `model_run_id`, `precision_mode`, `config_digest`, `latency_ms`.
  - Notes: Printed to CLI and logged as MLflow metrics/artifacts for downstream auditing.

## Relationships

- One `post_id` maps to many `Sample` rows, all sharing the same group id.
- One `Sample` maps to exactly one `FoldSplit` assignment in each CV manifest.
- One `DatasetManifest` owns many `Sample` records and zero or more `FoldSplit` assignments.
- One parent MLflow run aggregates many child fold runs, each of which owns a `ModelArtifact`.
- `CVSummary` references the parent run + `DatasetManifest` and summarizes all `ModelArtifact` entries for quick reporting.
- Each `InferenceResult` references exactly one `ModelArtifact` (best fold) and is traceable via `request_id`.
