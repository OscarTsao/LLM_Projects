# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/dataaug_multi_both/` (e.g., `cli/train.py`, `hpo/search_space.py`, `models/`, `training/`, `utils/`).
- Tests: `tests/` with `unit/`, `integration/`, and helpers (`conftest.py`).
- Experiments: `experiments/` (MLflow SQLite `experiments/mlflow.db`, artifacts under `experiments/artifacts/`).
- Config & tooling: `configs/`, `.devcontainer/`, `docker/`, `scripts/`.
- Data (optional, local): `Data/redsm5/*.csv` when `DATAAUG_USE_LOCAL_REDSM5=1`.

## Build, Test, and Development Commands
- `make install` / `make install-dev`: install runtime/dev deps via Poetry.
- `make test` / `make smoke` / `make test-unit`: run pytest (env guards for torchvision imports set automatically).
- `make test-coverage`: coverage for `src/dataaug_multi_both` → `htmlcov/`.
- `make lint` / `make format` / `make check`: ruff + mypy, black + ruff --fix, then tests.
- `make hpo`, `make hpo-best`, `make retrain-best`, `make hpo-results`: Optuna/training workflows.
- `make mlflow-ui`: launch UI at http://localhost:5000.
- Direct CLI: `poetry run python -m src.dataaug_multi_both.cli.train --help`.

## Coding Style & Naming Conventions
- Python 3.10, 4-space indent. Black line length 100.
- Ruff rules: `E,W,F,I,B,C4,UP` (E501 ignored; Black governs wrapping).
- Type hints encouraged; mypy checks `src/` (non-strict).
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.

## Testing Guidelines
- Pytest config lives in `pyproject.toml` (`testpaths = ["tests"]`).
- File/class/function patterns: `test_*.py`, `Test*`, `test_*`.
- Marks: `unit`, `integration`, `slow` (e.g., `pytest -m "not slow"`).
- Prefer fast, deterministic unit tests; integration may exercise Optuna/MLflow.
- Use `make test-coverage` to inspect coverage; no hard threshold enforced.

## Commit & Pull Request Guidelines
- Use Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`) as in repo history.
- PRs should include: clear summary, rationale, before/after behavior, test updates, and MLflow/Optuna implications. Reference issues (e.g., `Closes #123`).

## Environment & Configuration Tips
- Dev Container (`.devcontainer/`) is the recommended setup for reproducibility.
- MLflow tracks to `sqlite:///experiments/mlflow.db`; artifacts under `experiments/artifacts/`.
- Offline dataset: set `DATAAUG_USE_LOCAL_REDSM5=1` to use `Data/`.
- Avoid committing large datasets or generated artifacts.

