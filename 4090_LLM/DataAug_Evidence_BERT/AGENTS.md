# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the Python package for data prep, modeling, training, HPO, and the Typer CLI (`src/cli.py`, `src/train.py`, `src/hpo.py`, etc.).
- `configs/` contains Hydra configuration groups (`data/`, `model/`, `training/`, `logging/`, `hpo/`) composed via `configs/config.yaml`.
- `data/` stores the REDSM5 CSV assets; keep raw data immutable and versioned externally.
- `notebooks/` includes Colab-ready workflows (e.g., `evidence_qa_colab.ipynb` for TPU runs).
- `outputs/`, `mlruns/`, and `optuna.db` are generated artifacts; clean with `make clean`.

## Build, Test, and Development Commands
- `make install` – upgrade `pip` and install project dependencies from `requirements.txt`.
- `make train OVERRIDES="key=value"` – launch Hydra-configured training via `src.cli`.
- `make hpo OVERRIDES="key=value"` – run Optuna sweeps; trials log to MLflow automatically.
- `python -m src.cli train 'training.max_train_samples=64'` – example ad-hoc invocation.

## Coding Style & Naming Conventions
- Python code should follow PEP 8 with 4-space indentation; prefer descriptive snake_case identifiers.
- Configuration keys stay lower-case with dots for nesting (e.g., `training.learning_rate`).
- Keep docstrings concise and add comments only for non-obvious logic; avoid redundant commentary.

## Testing Guidelines
- Add unit tests under `tests/` (create if absent) using `pytest`; mirror module paths (e.g., `tests/test_data.py`).
- Ensure tokenization and evaluation helpers have coverage; mock external services (MLflow/Optuna) where practical.
- Run `pytest` locally before submitting changes; include smoke runs (`training.max_train_samples=32`) for sanity.

## Commit & Pull Request Guidelines
- Write commits in imperative mood (“Add Optuna sweep CLI”), scoped to a single logical change.
- Reference related issues in the body (`Refs #123`); summarize key config overrides or data assumptions.
- Pull requests should: describe motivation, list functional and configuration changes, note testing performed, and attach relevant artifacts/screenshot links for MLflow dashboards when applicable.
