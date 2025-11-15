# Feature Specification: 5-Fold DeBERTaV3-base Evidence Binding (NSP-Style)

**Feature Branch**: `001-debertav3-5fold-evidence`  
**Created**: 2025-11-12  
**Status**: Draft  
**Input**: User description: "run five fold training using debertav3-base from huggingface with the binary classification head provided by huggingface for debertav3-base. the task should be per sentence classification where the input format should look like bert nsp [cls] criterion [sep] sentence [sep], it is a evidence binding task where it should classify whether the sentence is the evidence of the criterion. use adamw_fuse as optimizer"

## Clarifications

### Session 2025-11-12

- Q: How should we build CV splits to avoid leakage from multiple sentences of the same post? → A: GroupStratifiedKFold by `post_id`; fallback to GroupKFold by `post_id` if group‑stratified is unavailable.
- Q: If fused AdamW is unavailable, which fallback optimizer should we standardize on? → A: `adamw_torch` (torch.optim.AdamW).
 - Q: Which training engine should we use to implement CV and logging? → A: Hugging Face Trainer.
 - Q: How should we handle class imbalance during training? → A: Use inverse-frequency class weights (weighted cross-entropy) by default; optionally enable Focal Loss (γ=2.0) with α set from class frequencies via Hydra switch.
 - Q: What max token length and truncation strategy should we use for pair encoding? → A: max_length=512 with truncation strategy `longest_first`.
 - Q: What fine-tuning strategy should we use for DeBERTaV3? → A: Full fine-tune (all layers).
 - Q: What precision mode should we use for training? → A: Prefer BF16; if unsupported use FP16; otherwise FP32.
 - Q: How should we construct negatives (non‑evidence pairs) for training? → A: Stratified random negatives targeting a 1:3 pos:neg ratio.
 - Q: Should we include Optuna HPO for this feature? → A: No HPO (fixed hyperparameters).
- Q: What should be the primary model selection metric? → A: Macro-F1.
- Q: Which learning‑rate scheduler and warmup ratio should we use? → A: Linear decay with warmup_ratio=0.06.

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Train 5-Fold DeBERTaV3 Evidence Classifier (Priority: P1)

As an ML practitioner, I want to train a binary classifier using
`microsoft/deberta-v3-base` with a Hugging Face sequence classification head on
NSP-style criterion–sentence pairs so that I can determine whether a sentence is
evidence for a DSM-5 criterion.

**Why this priority**: This is the core model training capability enabling
evidence binding.

**Independent Test**: Run the training entrypoint with Hydra config to execute
5-fold cross-validation; each fold trains and logs metrics/artifacts to MLflow;
the run completes without errors and produces an aggregate metrics summary.

**Acceptance Scenarios**:

1. Given paired inputs `[CLS] <criterion> [SEP] <sentence> [SEP]` from `data/`,
   when starting training with `folds=5`, then each fold trains a
   `AutoModelForSequenceClassification` initialized from
   `microsoft/deberta-v3-base` and logs accuracy/F1/ROC-AUC.
2. Given optimizer `adamw_torch_fused` (aka fused AdamW), when supported by the
   environment, then training uses it; otherwise it falls back to
   `adamw_torch`.

---

### User Story 2 - Aggregate and Report CV Metrics (Priority: P2)

As a researcher, I want aggregated cross-validation metrics (mean/std across
folds) and artifacts (confusion matrices, ROC/PR curves) logged to MLflow so I
can evaluate performance and compare runs.

**Why this priority**: Provides decision-ready evaluation for experiments.

**Independent Test**: A completed 5-fold CV run creates a parent run with child
fold runs in MLflow, logs per-fold metrics, and writes an aggregate metrics JSON
artifact (mean/std) to the parent run.

**Acceptance Scenarios**:

1. Given 5 completed folds, when aggregation runs, then aggregated metrics are
   logged to the parent run and visual artifacts are saved.
2. Given MLflow configured with `sqlite:///mlflow.db` and `./mlruns`, when the
   run finishes, then the UI shows all fold runs nested under the parent.

---

### User Story 3 - Inference API for Criterion–Sentence (Priority: P3)

As a user of the model, I want a simple inference function/CLI that takes a
criterion and a sentence and returns a label (evidence / not evidence) and
probability so I can integrate the model into downstream tasks.

**Why this priority**: Enables practical use and validation on new data.

**Independent Test**: Calling the inference entrypoint with a criterion and
sentence returns a prediction and probability and logs the exact config/model
version.

**Acceptance Scenarios**:

1. Given a saved best model from CV, when I pass a criterion–sentence pair, then
   the system returns `label ∈ {0,1}` and probability in [0,1].
2. Given Hydra overrides, when I change tokenizer/max length, then the
   preprocessing reflects those values and the configuration is logged to MLflow.

---


Additional stories may include: exporting to MLflow Model Registry and batch
inference over datasets.

### Edge Cases

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right edge cases.
-->

- What happens when [boundary condition]?
- How does system handle [error scenario]?

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

- **FR-001 (Model)**: Use `microsoft/deberta-v3-base` with
  `AutoModelForSequenceClassification` (num_labels=2). Initialize from HF Hub.
- **FR-002 (Input)**: Build NSP-style paired inputs:
  `[CLS] <criterion> [SEP] <sentence> [SEP]` from `data/DSM5/` and
  `data/redsm5/posts.csv` (or equivalent). Tokenization uses
  `truncation=longest_first` with `max_length=512`. Persist preprocessing/
  tokenization parameters.
- **FR-003 (CV)**: Perform 5-fold cross-validation with deterministic splits
  using GroupStratifiedKFold grouped by `post_id` to avoid leakage. If
  GroupStratifiedKFold is unavailable, use GroupKFold grouped by `post_id`.
- **FR-004 (Optimizer)**: Use fused AdamW optimizer `adamw_torch_fused` if
  available; otherwise fallback to `adamw_torch`.
- **FR-005 (Hydra)**: Manage all parameters via Hydra `configs/` with CLI
  overrides. Include data paths, model/tokenizer, training hparams, CV, seeds,
  and logging.
- **FR-006 (MLflow)**: Track runs in MLflow with
  `sqlite:///mlflow.db` and `./mlruns` artifact store. Use parent run for CV and
  child runs for each fold. Log metrics, params, configs, artifacts.
- **FR-007 (Reproducibility)**: Set/log seeds; capture full Hydra config,
  `pip freeze`, git SHA, and dataset manifest (filenames + checksums). Document
  any nondeterminism.
- **FR-008 (Metrics & Artifacts)**: Log per-fold accuracy, F1 (macro and
  positive class), ROC-AUC, PR-AUC; save confusion matrix images and curves.
- **FR-009 (Inference)**: Provide a callable/CLI to score a single
  criterion–sentence pair; return label + probability and log config/model
  version.
- **FR-010 (Trainer)**: Use Hugging Face `Trainer` for training/evaluation.
  Configure optimizer to `adamw_torch_fused` when available, else
  `adamw_torch`. Implement `compute_metrics` to log accuracy, F1 (macro and
  positive class), ROC-AUC, and PR-AUC. Use MLflow callback or custom logging
  to ensure nested runs per fold.
- **FR-011 (Loss & Imbalance)**: Support class-imbalance handling via Hydra:
  `loss.name ∈ {weighted_ce, focal}`. Default to weighted cross-entropy with
  inverse-frequency class weights; when `focal`, use γ=2.0 and α from class
  frequencies. Implement custom `compute_loss` in Trainer to apply weighting.
- **FR-012 (Selection Metric)**: Use Macro-F1 as the primary model selection
  metric. Configure Trainer with `metric_for_best_model=f1_macro` and
  `greater_is_better=true`. Unless otherwise specified, use threshold 0.5 for
  converting probabilities to labels in F1 computation.
- **FR-013 (Fine-tuning Strategy)**: Fine-tune all Transformer layers (no
  backbone freezing) by default. Expose a Hydra flag to optionally freeze
  backbone layers for low-VRAM scenarios, but default MUST be full fine-tune.
- **FR-014 (Precision & AMP)**: Prefer BF16 mixed precision when hardware
  supports it; otherwise use FP16 mixed precision; otherwise train in FP32.
  Implement detection and configuration via Hydra flags (e.g.,
  `trainer.bf16`, `trainer.fp16`) and automatically log the chosen precision in
  MLflow.
- **FR-015 (HPO Scope)**: Do not run HPO for this feature; hyperparameters are
  fixed via Hydra configs/overrides. If HPO is enabled in a future iteration,
  it MUST use Optuna with local storage `sqlite:///optuna.db` and log all trials
  to MLflow.
 - **FR-016 (Negative Sampling)**: Construct negatives via stratified random
   pairing to achieve a 1:3 positive:negative ratio. Perform sampling at the
   dataset construction stage before fold assignment, grouping by `post_id` so
   all pairs from a post remain in the same fold. Expose Hydra knobs
   `data.neg_strategy=stratified`, `data.neg_ratio=3`.
 - **FR-017 (LR Schedule & Warmup)**: Use linear LR schedule with
   `warmup_ratio=0.06`. Configure Trainer via `lr_scheduler_type=linear` and
   `warmup_ratio=0.06` in TrainingArguments/Hydra.

### Key Entities *(include if feature involves data)*

- **Sample**: `criterion_text`, `sentence_text`, `label` (0/1), optional ids
  (post_id, sentence_id, criterion_id).
- **FoldSplit**: mapping of sample ids to fold index (0..4), with seed and
  grouping strategy.
- **ModelArtifact**: Hugging Face model card, tokenizer files, config.json,
  MLflow model signature.

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: A 5-fold CV training completes successfully with per-fold and
  aggregate metrics logged to MLflow.
- **SC-002**: End-to-end run is reproducible using a documented Hydra command
  (seed + overrides), producing metrics within expected tolerance.
- **SC-003**: Inference for a given criterion–sentence returns a label and
  probability within 300 ms on CPU for single example (target; actual perf logged).
- **SC-004**: All configs, seeds, code version, and data manifests are captured
  as MLflow artifacts.
