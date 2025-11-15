<!--
Sync Impact Report
- Version change: 1.0.0 → 1.1.0
- Modified principles: Added P6 Reproducibility & Replication
- Added sections: none
- Templates requiring updates:
  - .specify/templates/plan-template.md: ✅ aligned (Constitution Check present)
  - .specify/templates/spec-template.md: ✅ aligned (no conflicts)
  - .specify/templates/tasks-template.md: ⚠ consider adding explicit reproducibility tasks scaffold
- Runtime docs:
  - README.md: ✅ mentions seeding and MLflow config
- Deferred TODOs: none
-->

# Evidence_Sentence_NoAug Constitution

## Core Principles

### P1. BERT-Based Binary Classification (Hugging Face)
The project MUST use BERT-family models from Hugging Face Transformers for a
binary classification task. Acceptable variants include base BERT derivatives
(e.g., `bert-base-uncased`) and compatible tokenizers. The model head MUST be a
binary classifier. Selection and all train/eval parameters MUST be configurable
via Hydra (see P3) and tracked in MLflow (see P4).

### P2. NSP-Style Criterion–Sentence Input Format
Training and inference inputs MUST follow BERT Next Sentence Prediction style:
`[CLS] <criterion> [SEP] <sentence> [SEP]`. Criterion text comes from `data/DSM5/`
(DSM-5 criteria files), and sentences come from post content (e.g.,
`data/redsm5/posts.csv`). Each example represents a criterion–post sentence
pair. Data readers MUST construct paired inputs consistently and record the
exact preprocessing/tokenization parameters in MLflow.

### P3. Configuration Management via Hydra
All parameters (data paths, model name, tokenizer, training hyperparameters,
seeds, MLflow URIs, and Optuna settings) MUST be managed by Hydra using modular
YAML configs under `configs/`. Defaults SHOULD be minimal and overridable via
CLI (e.g., `+trainer.max_epochs=3 model.name=bert-base-uncased`). Reproducible
seeding MUST be provided.

### P4. Experiment Tracking and Registry via MLflow (Local)
Experiments MUST be tracked with MLflow using the local SQLite DB and artifact
store:
- Tracking URI: `sqlite:///mlflow.db`
- Default artifact root: `./mlruns`

All runs MUST log parameters, metrics, tags, and artifacts (including Hydra
configs and code version when available). Models eligible for reuse MUST be
logged and registered in MLflow’s Model Registry backed by the same database.

### P5. Optional HPO via Optuna
If hyperparameter optimization is used, it MUST be implemented with Optuna. The
storage SHOULD be local SQLite `sqlite:///optuna.db`. Trials MUST integrate with
MLflow for metrics logging, and Hydra for search space configuration.

### P6. Reproducibility & Replication
End‑to‑end runs MUST be replicable. Requirements:
- Determinism: Set and log seeds for Python, NumPy, PyTorch, and Transformers;
  configure deterministic backends where possible (e.g., `cudnn.deterministic=True`,
  `cudnn.benchmark=False`). Document any unavoidable sources of nondeterminism.
- Environment capture: Pin package versions; log `pip freeze` (or environment
  export) and Git commit SHA as MLflow artifacts; record CUDA/cuDNN versions.
- Config capture: Log the full Hydra config and CLI overrides with each run.
- Data traceability: Record dataset locations, splits, and counts; store a
  manifest (filenames + checksums) or snapshot reference sufficient to rebuild
  the training/eval sets.
- Execution recipe: Provide a minimal command to reproduce a reported result
  (Hydra overrides + seed) and ensure it runs from a clean checkout.

## Technology Stack Constraints

- Data locations: posts/ground truth reside under `data/`; DSM-5 criteria under
  `data/` (e.g., `data/DSM5/`).
- Input pairing: criterion–sentence NSP format is the single source of truth for
  dataset construction and MUST be preserved in preprocessing.
- Transformers/HF: use official libraries; avoid custom forks unless justified
  and documented.

## Development Workflow & Quality Gates

- Style and docs: Follow Google-style docstrings with full type hints.
- Formatting/linting/type-checking: Use Black (line length 100), Ruff, and
  MyPy as configured in this repository. These checks MUST pass before merge.
- Automation: Prefer pre-commit hooks and CI to auto-run format/lint/type
  checks (serving as the “auto checking/auto formatting” gate).
- Utilities: Keep functions small and pure in `utils/`; avoid global state.

## Governance

This constitution defines non‑negotiable project rules. All PRs and reviews MUST
verify compliance. Changes to any principle or governance require a documented
amendment PR that updates this file, bumps the version per semver, and updates
affected docs/configs when applicable.

**Version**: 1.1.0 | **Ratified**: 2025-11-12 | **Last Amended**: 2025-11-12
