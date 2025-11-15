# Feature Specification: DL Experiments — Criteria Matching & Evidence Binding HPO

**Feature Branch**: `001-dl-experiments-criteria`  
**Created**: 2025-10-10  
**Status**: Draft  
**Input**: User description: "/speckit.specify so I am going to run an dl experiments that will perform criteria matching agent and evidence binding agent training. They are both using bert-like encoder model plus a classification head. I am going to use optuna to perform hyperparameters optimization to find the best combination among models, loss function, methods, training both agents together or separately, schedulers, optimizers, which data augmentation method is affective and which are the best augmentation combination."

## Clarifications

### Session 2025-10-10
- Q: Regarding the dataset schema (FR-016), what is the label format for the classification tasks? → A: Multi-label
- Q: Regarding the maximum sequence length (FR-017), what is the preferred trade-off between performance and computational cost? → A: 512 tokens
- Q: Regarding back-translation (FR-018), are offline or cached models permissible for data augmentation? → A: Yes, use pre-computed back-translations from a cached dataset to speed up training.
- Q: What is the acceptable maximum wall-clock time for a complete HPO study? → A: 3-7 days (extensive search)
- Q: What fields does each dataset record contain? → A: BERT NSP-like format with [CLS] text [SEP] criteria [SEP]
- Q: What should happen when a trial fails (e.g., OOM, NaN loss, crash)? → A: Log and skip; continue until max_failures threshold reached
- Q: How should study resumption work if interrupted? → A: Checkpoint every N trials; resume from last checkpoint
- Q: Does the Evidence Binding agent use a different second segment, or the same format? → A: Same format for both agents (both use criteria)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run HPO for Both Agents End-to-End (Priority: P1)

Run a single command to launch Optuna-driven HPO that trains the Criteria Matching Agent and the Evidence Binding Agent using BERT-like encoders with classification heads, logs to MLflow, and saves best configs/models.

**Why this priority**: Delivers the core experiment capability to search across architectures, loss functions, optimizers, schedulers, and augmentations with reproducible tracking.

**Independent Test**: Launch HPO on a small budget (e.g., 10 trials, 1 epoch) over a toy split; verify MLflow runs, best-trial JSON, and saved checkpoints for both agents.

**Acceptance Scenarios**:

1. Given dataset paths and a search space, When HPO runs for N trials, Then MLflow logs metrics/artifacts per trial and a best-trial summary is saved under `outputs/`.
2. Given successful HPO, When re-running with the same seed/config, Then results reproduce within tolerance and the same best config is selected.

---

### User Story 2 - Joint vs. Separate Training Modes (Priority: P2)

Toggle training both agents jointly (multi-task) vs. separately and compare outcomes within the same HPO study.

**Why this priority**: Determines whether shared representations help both tasks and impacts training efficiency.

**Independent Test**: Run two comparable HPO studies with a fixed budget; compare best validation metrics and report deltas in a summary table.

**Acceptance Scenarios**:

1. Given `train_mode=joint`, When HPO completes, Then a best joint config and metrics are produced and logged.
2. Given `train_mode=separate`, When HPO completes, Then best configs for each agent are produced and a comparison report is saved to `outputs/`.

---

### User Story 3 - Data Augmentation Ablations & Combinations (Priority: P3)

Evaluate which augmentation(s) are effective and the best combinations for text classification with BERT-like encoders.

**Why this priority**: Augmentation significantly impacts generalization, especially under class imbalance.

**Independent Test**: Fix backbone and optimizer; sweep augmentation toggles and intensities; generate ablation chart and store to MLflow and `outputs/`.

**Acceptance Scenarios**:

1. Given a fixed model/loss, When sweeping augmentations, Then the system ranks single augmentations by macro-F1 and saves a bar chart.
2. Given top-k single augmentations, When testing combinations up to size m, Then the system reports best combo and its contribution vs. baseline.

---

### User Story 4 - Backbone/Loss/Optimizer/Scheduler Search (Priority: P3)

Provide pluggable, Hydra-configured choices for backbones, losses, optimizers, and schedulers with Optuna search spaces.

**Why this priority**: Enables broad exploration of modeling/training strategies beyond defaults.

**Independent Test**: Run short HPO that samples across at least 3 backbones, 3 losses, 2 optimizers, and 2 schedulers; verify each choice appears in the trials table.

**Acceptance Scenarios**:

1. Given configured search spaces, When HPO runs, Then trials include varied backbone/loss/optimizer/scheduler selections.
2. Given trial artifacts, When loading the best trial, Then the exact choices can be reconstructed and retrained to within tolerance.

### Edge Cases

- Extremely long inputs exceeding `max_length` → must truncate consistently and warn.
- Severe class imbalance → support focal loss, class weights, and stratified splits.
- Limited GPU memory → support gradient accumulation, smaller batch size, fp16, gradient checkpointing.
- Missing or malformed fields in dataset → skip with reason logged; count and report.
- Tokenizer/backbone mismatch → validate model/tokenizer pair compatibility before training.
- Joint training instability → configurable loss weights, warmup schedules, and gradient clipping.
- Reproducibility across trials → seed everything per trial; detect non-deterministic ops and warn.
- Trial failures (OOM, NaN loss, runtime errors) → log failure details with full traceback; mark trial as failed; increment failure counter; continue study until max_failures threshold reached, then terminate gracefully with summary of failed trials.

## Requirements *(mandatory)*

### Functional Requirements

- FR-001: Provide two agents (Criteria Matching, Evidence Binding) each using a BERT-like encoder plus classification head.
- FR-002: Support `train_mode` in {`joint`, `separate`} with configurable loss weighting for joint mode.
- FR-003: Implement Optuna HPO with resumable studies, pruning, and budget control (trials, time, max failures). Maximum wall-clock time per study: 3-7 days. Failed trials (OOM, NaN loss, crashes) are logged and skipped; study continues until max_failures threshold is reached. Checkpoint study state every N trials (configurable); support automatic resumption from last checkpoint if interrupted.
- FR-004: Integrate Hydra configs for all tunables; allow CLI overrides for quick runs.
- FR-005: Log metrics, params, artifacts to MLflow (local backend by default). Save best trial config to `outputs/best_trial.json|yaml` and checkpoints under `outputs/checkpoints/`.
- FR-006: Implement text augmentations as composable transforms with per-method probability/intensity:
  - EDA (synonym replace, random swap, random delete)
  - Back-translation (optional, can be disabled in offline runs)
  - Token-level dropout/masking
  - Mixup in embedding space (sentence-level)
  - Cutoff/Span masking (SpanBERT-style)
- FR-007: Provide backbone choices (non-exhaustive examples): `bert-base-uncased`, `roberta-base`, `distilbert-base-uncased`, `allenai/scibert_scivocab_uncased`, `dmis-lab/biobert-base-cased-v1.1`.
- FR-008: Loss functions: cross-entropy, label smoothing cross-entropy, focal loss; support class weights.
- FR-009: Optimizers: AdamW, Adafactor, SGD+momentum; hyperparams include lr, weight decay, betas/momentum, eps.
- FR-010: Schedulers: linear warmup, cosine, cosine with restarts, step decay; hyperparams include warmup ratio/steps.
- FR-011: Training utilities: early stopping, gradient clipping, gradient accumulation, mixed precision, gradient checkpointing.
- FR-012: Data handling: stratified train/val/test splits; configurable `max_length`, padding/truncation, batch size, num_workers. Input format: text pairs tokenized as [CLS] text [SEP] criteria [SEP]; both Criteria Matching and Evidence Binding agents use identical input format.
- FR-013: Evaluation: per-agent metrics (accuracy, precision/recall, micro/macro F1), confusion matrix, PR/ROC curves where applicable.
- FR-014: Reproducibility: global/trial seeds, deterministic dataloader ordering, log software/hardware env.
- FR-015: Reporting: auto-generate study summary (top-k trials table, best configs per mode, augmentation ablation chart) saved under `outputs/reports/`.

Clarified requirements:

- FR-016: Dataset schema: text pairs formatted as [CLS] text [SEP] criteria [SEP]; multi-label classification.
- FR-017: The target `max_length` is 512 tokens.
- FR-018: Back-translation will use pre-computed translations from a cached dataset.

### Key Entities *(include if feature involves data)*

- AgentConfig: identifies agent (`criteria|evidence`), backbone, head dims, loss, optimizer, scheduler, augmentation set.
- AugmentConfig: list of transforms with per-transform params/probabilities and composition order.
- HPOStudyConfig: Optuna sampler, pruner, budgets (n_trials, timeout), directions, storage, checkpoint_interval (N trials between checkpoints for resumption).
- TrialResult: metrics per epoch, best epoch, params, artifacts URIs, seed.
- DatasetSpec: file paths, splits, tokenizer settings, label schema (multi-label), class weights; records contain text pairs (text, criteria) tokenized as [CLS] text [SEP] criteria [SEP].

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Achieve ≥ +3.0 macro-F1 improvement over baseline (no augmentation, default AdamW, CE loss) for at least one agent on validation.
- **SC-002**: Identify a statistically better augmentation combination vs. best single augmentation at p < 0.05 (paired t-test across 3 seeds) or document no significant gain.
- **SC-003**: Demonstrate reproducibility: best-trial re-train within ±0.5 macro-F1 across 3 runs with fixed seed and config.
- **SC-004**: Produce a complete MLflow experiment with ≥ N trials (configurable) containing metrics, params, and artifacts, plus a summary report in `outputs/reports/`.
- **SC-005**: Provide a comparison table for joint vs. separate training with clear winner per agent (or tie) using macro-F1.

