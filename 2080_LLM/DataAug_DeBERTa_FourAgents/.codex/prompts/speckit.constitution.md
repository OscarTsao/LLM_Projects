---
description: Create or update the project constitution from interactive or provided principle inputs, ensuring all dependent templates stay in sync.
---

## User Input

```text
# Project Constitution — Four-Agent Psychiatric Evidence Pipeline
_Last updated: 2025-10-17 09:15:49 UTC_

## Purpose & Scope
This repository implements a **non‑conversational four‑agent pipeline** for psychiatric evidence processing over sentence‑labeled texts (e.g., ReDSM5). The four agents are:
1. **Evidence Agent** — classifies each (sentence, symptom) as present/absent with confidence and provenance.
2. **Criteria Agent** — aggregates evidence into a calibrated provisional probability and decision.
3. **Suggestion Agent** — recommends next most informative symptoms to verify (value‑of‑evidence).
4. **Evaluation Agent** — evaluates metrics, fits calibration/thresholds, checks faithfulness, and emits feedback.

**Research‑only**; not a clinical diagnostic tool. No DSM verbatim content is embedded. Criteria logic is implemented as our own JSON policies and simple classifiers.

## Non‑Negotiable Principles
- **Safety & Ethics:** research use only; no crisis routing in this MVP. No PHI; only public/de‑identified data.
- **Transparency:** every criteria decision must cite supporting EvidenceUnits (sentence‑level quotes). If not supported, mark the decision as *uncertain*.
- **Reproducibility:** fixed random seeds; GroupKFold by `post_id`; Hydra‑managed configurations; MLflow (file backend) logging.
- **Outputs (mandatory layout):**
  - HPO: `outputs/hpo/{study_name}/`
    - `best.ckpt`, `best_config.yaml`, `val_metrics.json`, `test_metrics.json`
  - Training: `outputs/training/{run_id}/`
    - `model.ckpt`, `config.yaml`, `val_metrics.json`
  - Evaluation: `outputs/evaluation/{run_id}/`
    - `predictions.jsonl`, `criteria.jsonl`, `test_metrics.json`
  - Calibration artifacts: `artifacts/calibration.json`
- **Tracking:** `mlflow.set_tracking_uri("file:./mlruns")` is required.
- **Quality Gates:**
  - Evidence macro‑F1 (present) >= target baseline + 10 points.
  - Negation precision ≥ 0.90.
  - Criteria AUROC ≥ 0.80; ECE ≤ 0.05 after temperature scaling.
  - 100% of "likely" decisions have at least one supporting present EvidenceUnit.
- **Acceptance:** changes must pass analysis & checklist before merging to `main`.

## Out of Scope (MVP)
Conversational interviewing, risk triage, multimodal inputs, DSM verbatim text, and deployment.
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

You are updating the project constitution at `.specify/memory/constitution.md`. This file is a TEMPLATE containing placeholder tokens in square brackets (e.g. `[PROJECT_NAME]`, `[PRINCIPLE_1_NAME]`). Your job is to (a) collect/derive concrete values, (b) fill the template precisely, and (c) propagate any amendments across dependent artifacts.

Follow this execution flow:

1. Load the existing constitution template at `.specify/memory/constitution.md`.
   - Identify every placeholder token of the form `[ALL_CAPS_IDENTIFIER]`.
   **IMPORTANT**: The user might require less or more principles than the ones used in the template. If a number is specified, respect that - follow the general template. You will update the doc accordingly.

2. Collect/derive values for placeholders:
   - If user input (conversation) supplies a value, use it.
   - Otherwise infer from existing repo context (README, docs, prior constitution versions if embedded).
   - For governance dates: `RATIFICATION_DATE` is the original adoption date (if unknown ask or mark TODO), `LAST_AMENDED_DATE` is today if changes are made, otherwise keep previous.
   - `CONSTITUTION_VERSION` must increment according to semantic versioning rules:
     * MAJOR: Backward incompatible governance/principle removals or redefinitions.
     * MINOR: New principle/section added or materially expanded guidance.
     * PATCH: Clarifications, wording, typo fixes, non-semantic refinements.
   - If version bump type ambiguous, propose reasoning before finalizing.

3. Draft the updated constitution content:
   - Replace every placeholder with concrete text (no bracketed tokens left except intentionally retained template slots that the project has chosen not to define yet—explicitly justify any left).
   - Preserve heading hierarchy and comments can be removed once replaced unless they still add clarifying guidance.
   - Ensure each Principle section: succinct name line, paragraph (or bullet list) capturing non‑negotiable rules, explicit rationale if not obvious.
   - Ensure Governance section lists amendment procedure, versioning policy, and compliance review expectations.

4. Consistency propagation checklist (convert prior checklist into active validations):
   - Read `.specify/templates/plan-template.md` and ensure any "Constitution Check" or rules align with updated principles.
   - Read `.specify/templates/spec-template.md` for scope/requirements alignment—update if constitution adds/removes mandatory sections or constraints.
   - Read `.specify/templates/tasks-template.md` and ensure task categorization reflects new or removed principle-driven task types (e.g., observability, versioning, testing discipline).
   - Read each command file in `.specify/templates/commands/*.md` (including this one) to verify no outdated references (agent-specific names like CLAUDE only) remain when generic guidance is required.
   - Read any runtime guidance docs (e.g., `README.md`, `docs/quickstart.md`, or agent-specific guidance files if present). Update references to principles changed.

5. Produce a Sync Impact Report (prepend as an HTML comment at top of the constitution file after update):
   - Version change: old → new
   - List of modified principles (old title → new title if renamed)
   - Added sections
   - Removed sections
   - Templates requiring updates (✅ updated / ⚠ pending) with file paths
   - Follow-up TODOs if any placeholders intentionally deferred.

6. Validation before final output:
   - No remaining unexplained bracket tokens.
   - Version line matches report.
   - Dates ISO format YYYY-MM-DD.
   - Principles are declarative, testable, and free of vague language ("should" → replace with MUST/SHOULD rationale where appropriate).

7. Write the completed constitution back to `.specify/memory/constitution.md` (overwrite).

8. Output a final summary to the user with:
   - New version and bump rationale.
   - Any files flagged for manual follow-up.
   - Suggested commit message (e.g., `docs: amend constitution to vX.Y.Z (principle additions + governance update)`).

Formatting & Style Requirements:
- Use Markdown headings exactly as in the template (do not demote/promote levels).
- Wrap long rationale lines to keep readability (<100 chars ideally) but do not hard enforce with awkward breaks.
- Keep a single blank line between sections.
- Avoid trailing whitespace.

If the user supplies partial updates (e.g., only one principle revision), still perform validation and version decision steps.

If critical info missing (e.g., ratification date truly unknown), insert `TODO(<FIELD_NAME>): explanation` and include in the Sync Impact Report under deferred items.

Do not create a new template; always operate on the existing `.specify/memory/constitution.md` file.
