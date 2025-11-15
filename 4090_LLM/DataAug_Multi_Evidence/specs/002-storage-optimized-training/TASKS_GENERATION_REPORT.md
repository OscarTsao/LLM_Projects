# Task Generation Report: Storage-Optimized Training & HPO Pipeline

**Feature**: 002-storage-optimized-training  
**Generated**: 2025-10-10  
**Status**: ✅ Updated with clarifications

---

## Summary

The tasks.md file has been **updated** to reflect the key clarification from the comprehensive plan review:

Alignment: User Story 3 implements **per-trial evaluation** as mandated by the constitution and spec. An optional study-level summary report may be generated in addition, without replacing per-trial outputs.

---

## Task Breakdown

### Total Tasks: 47

**By Phase**:
- Phase 1: Setup & Infrastructure (T001-T008) - 8 tasks
- Phase 2: Foundational Components (T009-T018) - 10 tasks
- Phase 3: User Story 1 - Storage-Optimized Training/HPO with Resume (T019-T030) - 12 tasks [P1]
- Phase 4: User Story 2 - Portable Environment (T031-T036) - 6 tasks [P2]
- Phase 5: User Story 3 - Per-Trial Test Evaluation & JSON Reports (T037-T042) - 6 tasks [P3]
- Phase 6: Polish & Integration (T043-T047) - 5 tasks

**By User Story**:
- User Story 1 (P1): 12 tasks - Storage-optimized HPO with resume
- User Story 2 (P2): 6 tasks - Portable containerized environment
- User Story 3 (P3): 6 tasks - Per-trial test evaluation and JSON reports
- Setup/Infrastructure: 8 tasks
- Foundational: 10 tasks
- Polish: 5 tasks

---

## Key Updates Made

### User Story 3 Clarifications

Per-trial test evaluation remains the authoritative protocol (evaluate each trial's best model). To aid high-level reporting, an optional study-level summary artifact may be produced that references the best trial's per-trial report.

### Updated Tasks

1. **T037**: Implement Test Set Evaluator
   - Accepts `trial_id` as input
   - Evaluates best checkpoint(s) for that trial
   - Supports Feature 001 metrics (PR AUC, confusion matrices)

2. **T038**: Implement JSON Report Generator
   - Per-trial report saved to `experiments/trial_<uuid>/evaluation_report.json`
   - Includes `trial_id`, config snapshot, optimization metric name, best validation score
   - Optional `decision_thresholds` for Feature 001 integration

3. **T039**: Integrate Evaluation into Trial Completion
   - Evaluation runs after each trial completes
   - Adds CLI command: `python -m src.cli.evaluate --trial-id <uuid>`
   - Report saved to the trial directory

4. **T041**: Integration Test - Per-Trial Test Evaluation
   - Verifies evaluation runs after each trial completes
   - Verifies a report per trial in each trial directory

5. **T043**: End-to-End Integration Test
   - Updated workflow to verify per-trial reports and evaluation timing

---

## Parallel Opportunities

### Phase 1: Setup (All Parallelizable)
- T002: Poetry dependency management [P]
- T003: Makefile [P]
- T004: Hydra configuration [P]
- T005: Retry utility [P]
- T006: Logging system [P]
- T007: Storage monitor [P]
- T008: Contract validation [P]

**Total**: 7 tasks can run in parallel after T001

### Phase 2: Foundational (Partial Parallelization)
- T010, T011, T012, T013, T015, T017, T018 can run in parallel
- T014 (Multi-Task Model) depends on T011, T012, T013
- T016 (Optuna Search Space) depends on T014

**Total**: 7 tasks can run in parallel, then 2 sequential

### Phase 3: User Story 1 (Partial Parallelization)
- T019, T020, T021, T024, T025, T026 can run in parallel
- T022, T023, T027, T028, T029, T030 are sequential or have dependencies

**Total**: 6 tasks can run in parallel initially

### Phase 4: User Story 2 (High Parallelization)
- T031, T032, T033, T034, T035 can run in parallel
- T036 is sequential (integration test)

**Total**: 5 tasks can run in parallel

### Phase 5: User Story 3 (Sequential)
- T037 → T038 → T039 → T040 → T041 → T042
- Mostly sequential due to dependencies

**Total**: Limited parallelization

### Phase 6: Polish (Partial Parallelization)
- T044, T045, T046 can run in parallel
- T043, T047 are sequential

**Total**: 3 tasks can run in parallel

---

## Independent Test Criteria

### User Story 1 (P1)
**Test**: Launch training with aggressive retention policy; verify:
- Metrics fully logged to MLflow
- Only latest N and best K checkpoints retained
- Interrupted job resumes from latest checkpoint
- Storage reduction ≥60% vs. keep-all

### User Story 2 (P2)
**Test**: On fresh machine with container runtime:
- Start container (including image pull)
- Run sample training with experiment tracking
- Complete entire process within 15 minutes

### User Story 3 (P3)
**Test**: After HPO completes, verify:
- Each trial directory contains `evaluation_report.json`
- Report has all required fields (trial_id, test_metrics, config)
- Test metrics correspond to evaluation of the trial’s best checkpoint(s)
- Evaluation ran after each trial completed

---

## Implementation Strategy

### MVP First (User Story 1 Only)
**Timeline**: 4-5 weeks (3 developers) or 8-10 weeks (1 developer)

1. Complete Phase 1: Setup (1 week)
2. Complete Phase 2: Foundational (1 week)
3. Complete Phase 3: User Story 1 (2 weeks)
4. **STOP and VALIDATE**: Test independently
5. Deploy/demo if ready

**Deliverables**:
- Storage-optimized HPO with resume
- ≥60% storage reduction
- Resume in ≤2 minutes
- All metrics preserved

### Incremental Delivery
**Timeline**: 8-12 weeks total

1. MVP (US1): 4-5 weeks
2. Add US2 (Portable Environment): +2 weeks
3. Add US3 (Per-Trial Evaluation): +2 weeks
4. Polish & Integration: +1 week

### Parallel Team Strategy
**Timeline**: 4-6 weeks (3 developers)

1. **Week 1-2**: All developers on Setup + Foundational
2. **Week 3-4**: 
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. **Week 5-6**: Integration, testing, polish

---

## Dependencies

### Critical Path
```
T001 (Project Structure)
  ↓
T002-T008 [Parallel Setup]
  ↓
T009 (Data Loading) [BLOCKING]
  ↓
T010-T018 [Parallel Foundational]
  ↓
[User Stories can start in parallel]
  ├─ T019-T030 [User Story 1]
  ├─ T031-T036 [User Story 2]
  └─ T037-T042 [User Story 3]
  ↓
T043-T047 [Polish & Integration]
```

### User Story Dependencies
- **US1**: No dependencies on other stories (can start after Foundational)
- **US2**: No dependencies on other stories (can start after Foundational)
- **US3**: Depends on US1 (needs trial executor and checkpoint manager)

**Recommendation**: Implement US1 first (MVP), then US2 and US3 in parallel.

---

## Validation Checklist

### Before Starting Implementation
- [ ] All design documents reviewed and approved
- [ ] Hugging Face Hub dataset ID determined for RedSM5
- [ ] Project tracking set up (GitHub Projects/Jira)
- [ ] Developers assigned to phases

### After Each Phase
- [ ] Phase 1: All setup tasks complete, `make help` works
- [ ] Phase 2: All foundational components tested, imports work
- [ ] Phase 3: User Story 1 independently tested and validated
- [ ] Phase 4: User Story 2 independently tested and validated
- [ ] Phase 5: User Story 3 independently tested and validated
- [ ] Phase 6: All quality gates passed, constitution compliance verified

### Final Validation
- [ ] All 47 tasks complete
- [ ] Test coverage ≥80%
- [ ] All linters pass (ruff, black, mypy)
- [ ] All 3 user stories functional
- [ ] Constitution compliance verified
- [ ] README complete and accurate

---

## Next Steps

1. ✅ **DONE**: Align tasks.md with per-trial evaluation; add optional study summary task
2. ✅ **DONE**: Determine Hugging Face Hub dataset ID for RedSM5 (irlab-udc/redsm5)
3. ⏳ **TODO**: Set up project tracking with 47 tasks
4. ⏳ **TODO**: Begin Phase 1 implementation (T001-T008)
5. ⏳ **TODO**: Target MVP (User Story 1) completion in 4-5 weeks

---

## Notes

- Tasks are organized by user story for independent implementation
- Per-trial evaluation is constitutionally mandated; add optional study summary if needed
- MVP scope is User Story 1 only (storage-optimized HPO with resume)
- User Stories 2 and 3 can be added incrementally
- Parallel opportunities identified for team-based development
- All tasks have clear acceptance criteria and file paths

**Status**: ✅ Ready for implementation
