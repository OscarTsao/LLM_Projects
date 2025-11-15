# Checklists Directory

**Feature**: 002-storage-optimized-training  
**Created**: 2025-10-10

---

## Available Checklists

### 1. requirements-quality.md

**Purpose**: Validate requirements quality for Storage + Resume + Data Integrity focus areas

**Type**: Requirements quality audit (unit tests for requirements writing)

**Audience**: Iterative stakeholder review
- Author self-review before implementation
- Reviewer audit during spec review
- QA gate before release

**Focus Areas**:
- Storage optimization requirements (8 items)
- Resume reliability requirements (7 items)
- Data integrity requirements (7 items)
- Clarity & measurability (13 items)
- Consistency checks (9 items)
- Scenario coverage (15 items)
- Non-functional requirements (13 items)
- Dependencies & assumptions (8 items)
- Recovery/rollback gaps (7 items - flagged, not blocking)
- Ambiguities & conflicts (6 items)
- Traceability (7 items)

**Total Items**: 100

**Key Features**:
- 87% traceability (items reference spec sections or mark gaps)
- Tests requirements quality, NOT implementation
- Flags recovery/rollback gaps without blocking iteration
- Designed for progressive refinement (author → reviewer → QA)

---

## How to Use Checklists

### For Authors (Self-Review)

1. **Before implementation**: Review all items in requirements-quality.md
2. **Mark completed items**: Check boxes for requirements that are complete, clear, and consistent
3. **Identify gaps**: Note items marked [Gap] that need requirements added
4. **Resolve ambiguities**: Clarify items marked [Ambiguity] with specific criteria
5. **Update spec**: Add missing requirements, clarify vague terms, resolve conflicts

### For Reviewers (Spec Audit)

1. **Focus on clarity**: Review CHK023-CHK034 (ambiguous terms, measurability)
2. **Check consistency**: Review CHK035-CHK044 (internal and cross-document consistency)
3. **Validate coverage**: Review CHK045-CHK059 (scenario coverage, edge cases)
4. **Flag issues**: Comment on items that fail quality checks
5. **Request updates**: Ask author to address critical gaps before approval

### For QA (Pre-Release Gate)

1. **Validate measurability**: Review CHK030-CHK034 (can requirements be objectively tested?)
2. **Check scenario coverage**: Review CHK045-CHK059 (all flows covered?)
3. **Verify traceability**: Review CHK094-CHK100 (requirements traceable to user stories?)
4. **Assess testability**: Confirm all requirements have clear acceptance criteria
5. **Gate decision**: Block release if critical quality issues remain

---

## Checklist Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ Author Self-Review                                          │
│ - Review all 100 items                                      │
│ - Mark completed items                                      │
│ - Identify gaps (CHK014, CHK020, CHK022, etc.)             │
│ - Update spec.md to address gaps                           │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ Reviewer Audit                                              │
│ - Focus on clarity (CHK023-CHK034)                         │
│ - Check consistency (CHK035-CHK044)                        │
│ - Validate coverage (CHK045-CHK059)                        │
│ - Comment on failed items                                  │
│ - Request spec updates                                     │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ QA Gate                                                     │
│ - Validate measurability (CHK030-CHK034)                   │
│ - Check scenario coverage (CHK045-CHK059)                  │
│ - Verify traceability (CHK094-CHK100)                      │
│ - Assess testability                                       │
│ - Gate decision: PASS / FAIL                               │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ Implementation Ready                                        │
│ - All critical items checked                               │
│ - Gaps addressed or deferred with justification           │
│ - Requirements clear, consistent, measurable               │
└─────────────────────────────────────────────────────────────┘
```

---

## Flagged Gaps (Not Blocking This Iteration)

The following items are flagged as gaps but **not blocking** for this iteration:

**Recovery Scenarios** (CHK081-CHK084):
- CHK081: Corrupted MLflow database recovery
- CHK082: Orphaned artifacts recovery
- CHK083: Partial HPO study recovery
- CHK084: Disk full recovery

**Rollback Scenarios** (CHK085-CHK087):
- CHK085: Rollback after training divergence
- CHK086: Rollback after data contamination
- CHK087: Rollback after dependency upgrade failures

**Rationale**: These are advanced resilience scenarios that can be addressed in future iterations. Current requirements focus on core storage optimization, resume reliability, and data integrity.

**Decision Point**: Team should decide whether to address these gaps before implementation or defer to future work.

---

## Adding New Checklists

Each `/speckit.checklist` invocation creates a new checklist file. To add domain-specific checklists:

**Example domains**:
- `security.md` - Security requirements quality
- `performance.md` - Performance requirements quality
- `api.md` - API contract requirements quality
- `ux.md` - UX requirements quality

**Naming convention**: Use short, descriptive names (e.g., `security.md`, not `security-checklist-2025-10-10.md`)

**Structure**: Follow the template in `.specify/templates/checklist-template.md`

---

## Maintenance

**When to update checklists**:
- After spec.md changes (add/remove/modify requirements)
- After discovering new edge cases during implementation
- After QA identifies missing scenarios
- After production incidents reveal gaps

**How to update**:
- Run `/speckit.checklist` with updated focus areas
- Manually add items to existing checklist
- Mark obsolete items as `[-]` (cancelled)

**Archiving**:
- Move completed checklists to `checklists/archive/` after release
- Keep active checklists in `checklists/` root

---

## Quality Metrics

**Target Coverage**:
- ≥80% of functional requirements have traceability items
- ≥90% of edge cases have coverage items
- 100% of success criteria have measurability items

**Current Coverage** (requirements-quality.md):
- Traceability: 87% ✅
- Edge case coverage: 100% ✅
- Success criteria measurability: 100% ✅

---

## References

- **Spec**: [../spec.md](../spec.md)
- **Plan**: [../plan.md](../plan.md)
- **Data Model**: [../data-model.md](../data-model.md)
- **Tasks**: [../tasks.md](../tasks.md)
- **Constitution**: [../../../.specify/memory/constitution.md](../../../.specify/memory/constitution.md)

