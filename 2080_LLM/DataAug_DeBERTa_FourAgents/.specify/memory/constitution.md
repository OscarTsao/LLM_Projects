<!--
Sync Impact Report
- Version change: N/A → 1.0.0
- Modified principles: N/A (initial ratification)
- Added sections: Core Principles; Scope & Prohibitions (MVP); Development Workflow & Review; Governance; Version line
- Removed sections: None
- Templates requiring updates:
  - .specify/templates/plan-template.md → ✅ aligned (Constitution Check pulls rules from this file)
  - .specify/templates/spec-template.md → ✅ aligned (no mandatory section changes)
  - .specify/templates/tasks-template.md → ✅ aligned (no checklist structure changes)
- Follow-up TODOs: None
-->

# Four-Agent Psychiatric Evidence Pipeline Constitution

## Core Principles

### I. Safety & Ethics (Non‑Negotiable)
Research use only. Do not provide clinical advice or crisis routing in this MVP. No PHI may be stored or processed; only public or de‑identified data are permitted.

### II. Transparency & Faithfulness
Every criteria decision MUST cite supporting EvidenceUnits (sentence‑level quotes). When support is insufficient, the decision MUST be marked uncertain and surfaced in outputs.

### III. Reproducibility
All experiments MUST fix random seeds. Use GroupKFold by `post_id`. Manage configuration via Hydra. Track all runs with MLflow (file backend) to guarantee repeatability.

### IV. Outputs & Tracking (Mandatory Layout)
Outputs MUST follow the exact directory and filename layout:
- HPO: `outputs/hpo/{study_name}/` → `best.ckpt`, `best_config.yaml`, `val_metrics.json`, `test_metrics.json`
- Training: `outputs/training/{run_id}/` → `model.ckpt`, `config.yaml`, `val_metrics.json`
- Evaluation: `outputs/evaluation/{run_id}/` → `predictions.jsonl`, `criteria.jsonl`, `test_metrics.json`
- Calibration artifacts: `artifacts/calibration.json`
Tracking MUST set `mlflow.set_tracking_uri("file:./mlruns")`.

### V. Quality Gates & Acceptance
Minimum gates for merge or release:
- Evidence macro‑F1 (present) ≥ baseline + 10 points
- Negation precision ≥ 0.90
- Criteria AUROC ≥ 0.80; ECE ≤ 0.05 after temperature scaling
- 100% of “likely” criteria decisions have ≥1 present EvidenceUnit cited
Acceptance requires passing analysis and checklist before merging to `main`.

## Scope & Prohibitions (MVP)
Out of scope for this milestone: conversational interviewing, crisis/risk triage, multimodal inputs, DSM verbatim text, and deployment targets.

## Development Workflow & Review
- Spec, plan, tasks, and analysis live under `specs/[###-feature]/`.
- “Constitution Check” in planning MUST confirm conformance to Core Principles.
- Outputs and artifacts MUST match required paths to pass review.
- Any calibration artifacts produced by Evaluation MUST be consumed by Criteria when present.

## Governance
- Authority: This constitution supersedes other process documents for non‑negotiable rules.
- Amendments: Propose changes via PR with a rationale, migration plan if applicable, and an explicit version bump.
- Versioning policy:
  - MAJOR: Backward‑incompatible governance changes or principle removals/redefinitions
  - MINOR: New principle/section or materially expanded guidance
  - PATCH: Clarifications, wording, or non‑semantic refinements
- Compliance: Reviewers MUST verify “Quality Gates & Acceptance” and directory/MLflow rules. Violations block merge unless a temporary exception is documented and approved.

**Version**: 1.0.0 | **Ratified**: 2025-10-17 | **Last Amended**: 2025-10-17
