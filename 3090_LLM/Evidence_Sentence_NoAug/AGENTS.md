# Repository Guidelines

## Project Structure & Module Organization
- `src/Project/SubProject/` — core code: `models/`, `utils/`, and `engine/` stubs.
- `configs/` — Hydra configuration (YAML). Keep env/model/data configs modular.
- `data/DSM5/`, `data/redsm5/` — inputs. Posts and criteria are paired in BERT NSP style: `[CLS] <criterion> [SEP] <sentence> [SEP]`.
- `tests/` — pytest tests mirroring `src/` layout.
- `mlruns/` — MLflow artifacts; `mlflow.db` — local tracking DB; `optuna.db` — Optuna storage.
- `outputs/`, `artifacts/` — generated assets; ignored except `.gitkeep`.

## Build, Test, and Development Commands
- Create env and install deps:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -e '.[dev]'`
- Lint/format/type-check:
  - `ruff check src tests`
  - `black src tests`
  - `mypy src`
- Run tests: `pytest`
- MLflow UI (local): `mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns`

## Coding Style & Naming Conventions
- Python 3.10+. Use type hints and Google-style docstrings.
- Formatting: Black (line length 100). Linting: Ruff. Type checking: MyPy.
- Naming: `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Keep functions small; prefer pure helpers in `utils/`. Avoid global state.

## Testing Guidelines
- Place tests under `tests/` with `test_*.py`; mirror package paths.
- Use pytest fixtures; avoid touching real `data/` by default. Provide minimal sample inputs.
- Add unit tests for new logic and parsing of NSP inputs.

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- PRs must include: purpose, linked issue, runnable steps (commands), and expected metrics/outputs (e.g., MLflow run name, artifacts).
- Keep diffs focused; update README/configs when behavior or params change.

## Configuration Tips
- Hydra configs live in `configs/`; prefer overrides over hardcoding (e.g., `+trainer.max_epochs=3 model.name=bert-base-uncased`).
- For reproducibility: set seeds via `Project.SubProject.utils.set_seed`; log params/tags with `configure_mlflow` and `mlflow_run`.
- Optuna storage: `sqlite:///optuna.db`; prefer study names per feature/experiment.

## Active Technologies
- Python 3.10+ + PyTorch, Transformers, Datasets/Pandas, scikit-learn, (001-debertav3-5fold-evidence)
- Local files for data; MLflow SQLite DB `mlflow.db`, artifacts under `mlruns/`; optional (001-debertav3-5fold-evidence)

## Recent Changes
- 001-debertav3-5fold-evidence: Added Python 3.10+ + PyTorch, Transformers, Datasets/Pandas, scikit-learn,
