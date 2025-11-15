# Repository Guidelines

## Project Structure & Module Organization
Keep Python package code under `src/dataaug_multi_both/`, grouping related components in their own modules. Tests live in `tests/` and should mirror the package layout to keep fixtures discoverable. Store experiment definitions in `configs/`, long-lived datasets in `Data/`, and feature specs or design notes in `specs/`. Automation files reside under `.github/`, while generated artifacts such as `mlruns/`, `outputs/`, `multirun/`, and `.hydra/` stay out of version control.

## Build, Test, and Development Commands
Install the toolchain with `poetry install`, which pins dependencies from `pyproject.toml`. Format via `poetry run black .`, lint with `poetry run ruff check .`, and type-check using `poetry run mypy src`. Run all tests through `poetry run pytest -v`; add `-m unit` or `-m slow` when you only need a subset. Package artifacts only when needed using `poetry build`.

## Coding Style & Naming Conventions
Use Python 3.10, 4-space indentation, and Black’s 100-character line length. Keep one public class or function per file unless the logic is trivial. Follow snake_case for modules, functions, and variables; CamelCase for classes; and UPPER_SNAKE for constants. Type hints and targeted docstrings are expected on public entry points, and focused helper functions keep data augmentation routines testable.

## Testing Guidelines
Author tests with pytest using `test_*.py` naming and `Test*` classes where helpful. Mark suites with `@pytest.mark.unit`, `@pytest.mark.integration`, or `@pytest.mark.slow` so CI selectors stay meaningful. Cover core sampling, augmentation, and Hydra wiring paths, and add regression tests whenever a bug is fixed to prevent recurrence.

## Commit & Pull Request Guidelines
Write Conventional Commits such as `feat: add mixup pipeline` or `fix: guard empty evidence set` in imperative mood. Keep commits narrowly scoped and reference issues using `Closes #123` when applicable. Pull requests should summarize intent, list key changes, call out config or data impacts, and include test output (`poetry run pytest -v`). Attach screenshots or metrics for training-impacting work.

## Security & Configuration Tips
Never commit credentials; rely on environment variables like `MLFLOW_TRACKING_URI` and local `.env` files that stay ignored. Validate that large artifacts remain in `Data/` or other ignored directories before opening a PR. When using remote containers, prefer the project’s devcontainer configuration for a consistent Poetry environment.
