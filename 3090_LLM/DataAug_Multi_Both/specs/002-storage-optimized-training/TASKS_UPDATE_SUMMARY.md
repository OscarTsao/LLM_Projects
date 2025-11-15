# Tasks.md Update Summary

**Date**: 2025-10-10  
**Feature**: 002-storage-optimized-training  
**Update Type**: Alignment with Constitution v1.1.0 and Spec.md (Per-Study Evaluation)

---

## Changes Made

### 1. Header Update (Lines 1-10)

**Before**:
```markdown
**Updated**: 2025-10-10 (realign to per-trial evaluation per constitution)

**Recent Updates**:
- User Story 3 clarified to be **per-trial evaluation** (constitution + spec). Each trial writes its own JSON report after completion.
```

**After**:
```markdown
**Updated**: 2025-10-10 (updated to per-study evaluation per constitution v1.1.0)

**Recent Updates**:
- User Story 3 updated to **per-study test evaluation** (constitution v1.1.0 + spec). Hybrid approach: each trial evaluates on validation set during training; test set evaluation occurs once per study after all trials complete, evaluating only the best model from the entire study. Study-level JSON report saved in study directory.
```

---

### 2. Phase 5 Title Update (Line 22)

**Before**: `Phase 5: User Story 3 - Per-Trial Test Evaluation & JSON Reports`  
**After**: `Phase 5: User Story 3 - Per-Study Test Evaluation & JSON Reports`

---

### 3. Phase 5 Section Header (Lines 536-542)

**Before**:
```markdown
## Phase 5: User Story 3 - Per-Trial Test Evaluation & JSON Reports [P3]

**Goal**: Enable researchers to evaluate each trial's best model on the test set and generate a per-trial machine-readable JSON report.

**Clarification**: Evaluation is **per-trial** and runs after each trial completes. Co-best checkpoints within a trial are evaluated and included in that trial's report.

**Independent Test**: After HPO completes, verify every trial directory contains a valid JSON report with test metrics.
```

**After**:
```markdown
## Phase 5: User Story 3 - Per-Study Test Evaluation & JSON Reports [P3]

**Goal**: Enable researchers to evaluate the best model from the entire HPO study on the test set and generate a study-level machine-readable JSON report.

**Clarification**: Evaluation is **per-study** (hybrid approach) - each trial evaluates on validation set during training; test set evaluation runs once after all trials complete, evaluating only the best model from the entire study. Co-best checkpoints within the best trial are all evaluated and included in the study report.

**Independent Test**: After HPO completes, verify the study directory contains a valid JSON report with test metrics for the best model from the entire study.
```

---

### 4. T037: Implement Test Set Evaluator (Lines 546-557)

**Key Changes**:
- Input: `trial_id` → `study_id`
- Added: "Identify best trial from the study (by optimization metric)"
- Load checkpoints: "from the trial" → "from the best trial"
- Schema: `trial_output_schema.json` → `study_output_schema.json`
- Acceptance: "for best model from study" added

---

### 5. T038: Implement JSON Report Generator (Lines 560-571)

**Key Changes**:
- Description: "per trial" → "study-level"
- Fields: `trial_id` → `study_id, best_trial_id`
- Checkpoints: "from the trial" → "from the best trial"
- Path: `experiments/trial_<uuid>/` → `experiments/study_<uuid>/`
- Schema: `trial_output_schema.json` → `study_output_schema.json`
- Acceptance: "per trial" → "per study"

---

### 6. T039: Integrate Evaluation into Trial Completion (Lines 574-584)

**Key Changes**:
- Description: "after each trial completes" → "after HPO study completes"
- Added: "After all trials complete, identify best trial by optimization metric"
- Call evaluator: "on that trial's best checkpoint(s)" → "on the best trial"
- Save report: "to the trial directory" → "to the study directory"
- Co-best: "within a trial" → "within best trial"
- CLI: `--trial-id` → `--study-id`
- Acceptance: "after each trial completion, report saved to trial directory" → "after study completion, report saved to study directory"

---

### 7. T040: Contract Test - EvaluationReport Schema (Line 593)

**Key Changes**:
- Schema: `trial_output_schema.json` → `study_output_schema.json`

---

### 8. T041: Integration Test (Lines 601-611)

**Key Changes**:
- Title: "Per-Trial" → "Per-Study"
- Filename: `test_per_trial_evaluation.py` → `test_per_study_evaluation.py`
- Verify: "each trial directory contains" → "the study directory contains"
- Fields: `trial_id` → `study_id, best_trial_id`
- Metrics: "for that trial" → "from the entire study"
- Co-best: "handled correctly" → "within best trial handled correctly"
- Timing: "after each trial completes" → "after all trials complete"

---

### 9. Checkpoint Message (Line 627)

**Before**: `User Story 3 Complete - Per-trial test evaluation and JSON reporting functional`  
**After**: `User Story 3 Complete - Per-study test evaluation and JSON reporting functional`

---

## Rationale

These updates align tasks.md with:

1. **Constitution v1.1.0** - Amended to allow per-study test evaluation for large-scale HPO (1000+ trials)
2. **Spec.md User Story 3** - Updated to per-study evaluation (hybrid approach)
3. **Data-model.md** - EvaluationReport schema updated to per-study

**Hybrid Approach**:
- **Per-trial**: Validation set evaluation during training (guides HPO optimization)
- **Per-study**: Test set evaluation once after all trials complete (prevents overfitting)

---

## Files Affected

- `specs/002-storage-optimized-training/tasks.md` (updated)

---

## Verification

To verify the updates are correct:

```bash
# Check Phase 5 title
grep "Phase 5:" specs/002-storage-optimized-training/tasks.md

# Check evaluation approach
grep -A 2 "Clarification:" specs/002-storage-optimized-training/tasks.md | grep -A 2 "Phase 5"

# Check T037-T041 descriptions
grep -A 5 "T037:\|T038:\|T039:\|T040:\|T041:" specs/002-storage-optimized-training/tasks.md | grep "study"

# Verify no remaining "per-trial" references in Phase 5 (except optional summary task)
grep -n "per-trial" specs/002-storage-optimized-training/tasks.md
```

---

## Status

✅ **All updates complete**

Tasks.md now correctly reflects:
- Per-study test evaluation (not per-trial)
- Hybrid approach (validation per-trial, test per-study)
- Study-level JSON reports
- Evaluation after all trials complete
- Best model from entire study evaluated

---

## Next Steps

1. ✅ Update contracts/study_output_schema.json (if it doesn't exist, create it)
2. ✅ Update data-model.md EvaluationReport (already done)
3. ✅ Update plan.md Phase 2.5 (already done)
4. ✅ Update spec.md User Story 3 (already done)
5. ⏳ Implement tasks T037-T042 with per-study evaluation logic

---

**Last Updated**: 2025-10-10  
**Verified By**: AI Assistant

