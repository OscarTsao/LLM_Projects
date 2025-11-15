---
description: Execute the implementation planning workflow using the plan template to generate design artifacts.
---

## User Input

```text
# Technical Plan
_Last updated: 2025-10-17 09:15:49 UTC_

## Stack
Python 3.10+, PyTorch, HuggingFace Transformers, Hydra, Optuna, MLflow (file backend), numpy/pandas, scikit‑learn.

## Architecture
Linear pipeline with typed contracts:
- Evidence → Criteria → Suggestion → Evaluation
- Shared artifacts: `artifacts/calibration.json`

## Data Model (Dataclasses)
- **EvidenceUnit**: `eu_id, post_id, sentence_id, sentence, symptom, assertion, score`
- **CriteriaResult**: `post_id, p_dx, decision, supporting{symptom→[eu_id]}, conflicts[], missing[]`
- **Suggestion**: `post_id, ranked:[{symptom, delta_p, reason}]`

## Directory Layout
```
psy-msa/
  .specify/
  configs/
    data/redsm5.yaml
    evidence/pairclf.yaml
    criteria/aggregator.yaml
    suggest/voi.yaml
    eval/default.yaml
    pipeline/default.yaml
    hpo/evidence_pairclf.yaml
  src/
    schema/types.py
    agents/{evidence_agent.py,criteria_agent.py,suggestion_agent.py,evaluation_agent.py}
    evidence/{train_pairclf.py,infer_pairclf.py}
    criteria/aggregate.py
    suggestion/voi.py
    eval/{metrics.py,calibration.py,report.py}
    pipeline/run_pipeline.py
    utils/{hydra_mlflow.py,seed.py,io.py}
  scripts/{run_pipeline.sh,run_hpo_evidence.sh}
  outputs/{hpo,training,evaluation}/
  mlruns/
  artifacts/
```

## Hydra Configuration Tree
- `data/redsm5.yaml`: dataset paths, split files.
- `evidence/pairclf.yaml`: model, optimizer, loss, trainer knobs.
- `criteria/aggregator.yaml`: symptoms, key symptoms, thresholds path, temperature.
- `suggest/voi.yaml`: top_k, uncertain band.
- `eval/default.yaml`: primary metrics, gates.
- `pipeline/default.yaml`: orchestrator wiring, output paths, mlflow URI.
- `hpo/evidence_pairclf.yaml`: search space & study params.

## HPO
- Optuna TPE + ASHA pruner; study per component (start with Evidence).
- Objective: `macro_F1_present + 0.2*neg_precision − 0.5*ECE`.
- Export best: `outputs/hpo/{study}/best.ckpt`, `best_config.yaml`, `val_metrics.json`, `test_metrics.json`.

## Evaluation
- Evidence: per‑symptom P/R/F1 (present), negation precision.
- Criteria: AUROC/F1, ECE, contradictions, faithfulness.
- Calibration fitting on dev → write `artifacts/calibration.json`.
- Writers produce evaluation JSONL and metrics JSON files.

## Runbook
- **HPO:** `python -m src.evidence.train_pairclf +hpo.enable=true hpo.study_name=evidence_hpo`
- **Pipeline:** `python -m src.pipeline.run_pipeline evidence.ckpt_path=outputs/hpo/evidence_hpo/best.ckpt`
- **Evaluation:** `python -m src.eval.report`

## Future (not in MVP)
- Conversation loop (Doctor/Patient) and LangGraph orchestration; POMDP/GNN Suggestion Agent.
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

1. **Setup**: Run `.specify/scripts/bash/setup-plan.sh --json` from repo root and parse JSON for FEATURE_SPEC, IMPL_PLAN, SPECS_DIR, BRANCH. For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

2. **Load context**: Read FEATURE_SPEC and `.specify/memory/constitution.md`. Load IMPL_PLAN template (already copied).

3. **Execute plan workflow**: Follow the structure in IMPL_PLAN template to:
   - Fill Technical Context (mark unknowns as "NEEDS CLARIFICATION")
   - Fill Constitution Check section from constitution
   - Evaluate gates (ERROR if violations unjustified)
   - Phase 0: Generate research.md (resolve all NEEDS CLARIFICATION)
   - Phase 1: Generate data-model.md, contracts/, quickstart.md
   - Phase 1: Update agent context by running the agent script
   - Re-evaluate Constitution Check post-design

4. **Stop and report**: Command ends after Phase 2 planning. Report branch, IMPL_PLAN path, and generated artifacts.

## Phases

### Phase 0: Outline & Research

1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

### Phase 1: Design & Contracts

**Prerequisites:** `research.md` complete

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Agent context update**:
   - Run `.specify/scripts/bash/update-agent-context.sh codex`
   - These scripts detect which AI agent is in use
   - Update the appropriate agent-specific context file
   - Add only new technology from current plan
   - Preserve manual additions between markers

**Output**: data-model.md, /contracts/*, quickstart.md, agent-specific file

## Key rules

- Use absolute paths
- ERROR on gate failures or unresolved clarifications
