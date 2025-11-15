# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains modules `data/`, `models/`, `utils/` plus entry points `train.py` and `hpo.py`; expose shared utilities through the package `__init__.py`.
- `configs/` holds Hydra groups (`data/`, `model/`, `training/`, `hpo/`). Add variant YAMLs rather than editing `config.yaml` so runs stay reproducible.
- `Data/` stores raw corpora, `outputs/` captures Hydra run folders, and `artifacts/` retains promoted checkpoints; clear scratch runs with `make clean`.
- Treat the `Makefile` as the workflow index and extend it whenever new automation or scripts are added.

## Build, Test, and Development Commands
- `make install` / `make dev-install`: install runtime or dev dependencies from `pyproject.toml`.
- `make train`, `make train-bert`, `make train-deberta`, `make train-roberta`: launch training presets.
- `make experiment ARGS='training.batch_size=16 training.learning_rate=1e-5'`: run custom overrides without touching source.
- `make hpo`: execute Optuna hyperparameter search through `src/hpo.py`.
- `make format`, `make lint`, `make check`: run Black, Ruff, or both; use `make check` as the pre-PR gate.

## Coding Style & Naming Conventions
- Format with Black defaults (88 columns, 4-space indents) via `make format`; order imports stdlib → third-party → local.
- Resolve Ruff findings with `make lint`; favour explicit type hints and dataclasses for configuration objects.
- Use snake_case for modules, functions, and Hydra keys; reserve PascalCase for PyTorch modules.
- Park exploratory notebooks or scripts under `artifacts/` to keep the package clean.

## Testing Guidelines
- Establish a PyTest-based `tests/` package mirroring `src/`; cover losses, utilities, and configuration loaders first.
- Name files `test_<module>.py` and functions `test_<behavior>`; seed randomness for deterministic checks.
- Run `PYTHONPATH=. pytest tests -q` and narrow focus with `pytest -k token` while iterating; aim for coverage on new code even if legacy areas lag.

## Commit & Pull Request Guidelines
- Write imperative commits (e.g., `Add span evaluator`) and capture notable Hydra overrides or datasets in the body when relevant.
- Confirm `make check` and the PyTest command succeed before pushing; attach metrics or MLflow run URIs in the PR description.
- Reference issues, describe configuration impacts, and call out migration steps; include screenshots only when diagnostics improve clarity.

## Configuration & Experiment Tracking
- Treat `configs/config.yaml` as the baseline; compose runs with overrides (`PYTHONPATH=. python src/train.py model=bert_base training.max_epochs=6`) instead of editing defaults.
- Start MLflow with `make mlflow-server` when needed and note the tracking URI in PRs; keep Hydra snapshots in `outputs/` until reviews close.
