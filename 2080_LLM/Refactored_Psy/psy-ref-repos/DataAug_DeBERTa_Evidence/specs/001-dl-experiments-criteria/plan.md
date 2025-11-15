# Implementation Plan: DL Experiments — Criteria Matching & Evidence Binding

**Branch**: `001-dl-experiments-criteria` | **Date**: 2025-10-10 | **Spec**: spec.md

**⚠️ PREREQUISITE**: This feature extends Feature 002 (Storage-Optimized Training & HPO Pipeline). Feature 002 must be implemented first to provide the foundational HPO infrastructure.

## Summary

Design threshold tuning for the dual-agent HPO framework (criteria matching + evidence binding).
This plan adds per-criterion decision thresholds for multi-label classification heads and
evidence span scoring thresholds. Thresholds are calibrated post-training on the validation
set to maximize macro-F1, then stored in trial configs and test reports for reproducibility.

**Sequential Dependency**: This feature (001) extends Feature 002's HPO infrastructure by adding
post-training threshold calibration. Feature 002 provides the base training pipeline, checkpoint
management, and MLflow tracking; this feature adds threshold optimization on top.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: PyTorch, transformers, datasets, Optuna, MLflow, Hydra
**Project Type**: ML research experiments (dual agents) with HPO

## Constitution Check

Status: ✅ Satisfies principles I–VII (reproducibility, storage-aware, dual-agent,
MLflow tracking, resume, portable env, Makefile-driven). Threshold tuning only
adjusts inference decision rules and does not conflict with storage optimization.

**Note**: Infrastructure requirements (Principles II, IV, V, VI, VII) are satisfied by
Feature 002. This feature focuses on Principle III (Dual-Agent Architecture) by adding
agent-specific threshold calibration and enhanced metrics (PR AUC, confusion matrices).

## Design Updates: Per-Criterion Thresholding and Post-Training Calibration

**Approach**: Post-training threshold calibration on validation set (not part of HPO search space).

**Rationale**: Separates model training from decision boundary tuning, reduces HPO search space
complexity, and allows threshold optimization after model convergence.

### Criteria Matching (multi-label)

- Head type: multi-label classifier with `C` outputs (one per criterion).
- Inference decision: positive if `sigmoid(logit[c]) >= threshold[c]`.
- New config fields (added to Feature 002's TrialConfig):
  - `criteria_thresholds`: list[float] length `C`, default all 0.5
  - `criteria_threshold_strategy`: one of {`global`, `per_class`} (default `per_class`)
- Calibration procedure (post-training):
  - After training completes, evaluate model on validation set
  - For each criterion, sweep threshold from 0.30 to 0.90 in steps of 0.05
  - Select threshold that maximizes per-criterion F1
  - If `global` strategy, use single threshold that maximizes macro-F1
- Validation objective: maximize macro-F1 (primary) with tie-breaker by macro-precision.

### Criteria Agent Metrics (additions)

- Add PR AUC (Area Under Precision-Recall Curve) to criteria metrics:
  - Compute per-criterion PR AUC and macro-average across criteria.
  - Log per-criterion and macro PR AUC to MLflow and include in EvaluationReport.
- Add confusion matrix for each criterion (binary 2x2: [[tn, fp], [fn, tp]]):
  - Provide per-criterion matrices; aggregate counts reported alongside macro metrics.
  - Persist in EvaluationReport under `test_metrics.criteria_matching.per_criterion[*].confusion_matrix`.

### Evidence Binding (span extraction)

- Heads supported: `start_end_*`, `bio_crf`, `sentence_reranker` (from Feature 002).
- New config fields (added to Feature 002's TrialConfig):
  - `evidence_null_threshold`: float in [0.30, 0.95], default 0.5 (probability mass to emit "no evidence")
  - `evidence_min_span_score`: float in [0.00, 0.50], default 0.0 (suppress low-confidence spans)
- Calibration procedure (post-training):
  - After training completes, evaluate model on validation set
  - Sweep `null_threshold` from 0.30 to 0.95 in steps of 0.05
  - Sweep `min_span_score` from 0.00 to 0.50 in steps of 0.05
  - Select thresholds that maximize token-level F1 (or sentence-level F1 for reranker)
  - Note: For CRF heads, `null_threshold` may not apply (document as N/A)
- Validation objective: maximize token-level F1 (or sentence-level F1 for reranker).

### Logging & Reproducibility

- Persist resolved thresholds in the saved TrialConfig (Hydra → dict → JSON artifact) and
  in MLflow params for each trial.
- Include thresholds used during test evaluation inside the per-trial EvaluationReport.

### Progress Visualization (training + HPO UI)

- Epoch/trial progress bars using `tqdm`:
  - Trial-level bar (Optuna trials) and epoch-level bar within training.
  - Respect `ui.progress=true|false` to enable/disable bars (non-interactive CI).
- Stdout status lines:
  - Emit concise, human-readable updates (epoch N/M, loss/metric snapshot, disk status).
  - Controlled by `ui.stdout_level` (INFO/DEBUG/WARNING). Always flush on update.
  - Coexist with structured JSON logs; no duplication of high-frequency metrics.

## Phase 0: Research

Topics and rationale captured in `research.md`:

- Empirical impact of per-class vs global thresholding on macro-F1.
- Recommended ranges and discretization for thresholds to stabilize HPO.
- Handling calibration drift across criteria (e.g., class imbalance effects).

## Phase 1: Data Model & Contracts

Produce `data-model.md` and contracts reflecting new fields:

- TrialConfig additions:
  - `criteria.thresholds: list[float] (len=C, [0,1])`
  - `criteria.threshold_strategy: str in {global, per_class}`
  - `evidence.null_threshold: float [0,1]`
  - `evidence.min_span_score: float [0,1]`
  - `hpo.tune_thresholds: bool` (default true)
- EvaluationReport additions:
  - `decision_thresholds.criteria: list[float]`
  - `decision_thresholds.evidence: { null_threshold, min_span_score }`
  - `test_metrics.criteria_matching` includes:
    - `macro_pr_auc: float`
    - `per_criterion`: array of objects with fields `{ id, f1, precision, recall, pr_auc, confusion_matrix }`
  - `ui` config options:
    - `ui.progress: bool` (default true)
    - `ui.stdout_level: enum[INFO, DEBUG, WARNING]` (default INFO)

Contracts directory:

- `contracts/config_schema.yaml`: include schema for the new fields and ranges
- `contracts/trial_output_schema.json`: include `decision_thresholds` block and criteria metrics (PR AUC, confusion matrices)

## Phase 1: Quickstart

Add a runnable example for threshold tuning:

```
python -m src.cli.train mode=hpo \
  model.head=criteria_multi_label \
  criteria.threshold_strategy=per_class \
  hpo.tune_thresholds=true \
  hpo.metric=macro_f1 \
  hpo.directions=[maximize]
```

## Integration with Feature 002

**Prerequisite**: Feature 002 must be implemented first (User Story 1 minimum: storage-optimized HPO with resume).

**Integration Points**:

1. **TrialConfig Schema**: Threshold fields already added to Feature 002's data-model.md as optional fields
2. **Evaluator Extension**: Add threshold calibration to `src/training/evaluator.py` after training completes
3. **Metrics Enhancement**: Add per-criterion PR AUC and confusion matrices to evaluation metrics
4. **Report Schema**: Threshold values stored in EvaluationReport's `decision_thresholds` field

**Implementation Sequence**:
1. Wait for Feature 002 User Story 1 completion (storage-optimized HPO with resume)
2. Extend `src/training/evaluator.py` with threshold calibration logic
3. Add PR AUC and confusion matrix computation to metrics
4. Update EvaluationReport generation to include calibrated thresholds
5. Test on small dataset (10 trials) before scaling

## Notes

- Thresholds are decision parameters, not trainable weights.
- Post-training calibration approach chosen to reduce HPO search space complexity.
- Calibration happens once per trial after training completes, on validation set.
- For CRF evidence heads, `null_threshold` may not apply; document as N/A.
- Threshold values are stored in TrialConfig and EvaluationReport for reproducibility.

## Implementation Hints

- Use `sklearn.metrics.precision_recall_curve` + `auc` (or TorchMetrics equivalent) for PR AUC.
- Build per-criterion confusion matrices by thresholding per-class probabilities independently.
- For tqdm in nested loops, prefer `position` and `leave=False` for clean output in long HPO runs.
