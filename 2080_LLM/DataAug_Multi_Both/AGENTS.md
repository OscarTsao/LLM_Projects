# Repository Guidelines

## Project Structure & Module Organization
- Source code: `src/dataaug_multi_both/` (package root). Keep new modules here.
- Configs: `configs/` for Hydra YAMLs (e.g., `configs/train.yaml`, `configs/model/*.yaml`).
- Data: `Data/` holds datasets and artifacts; do not commit large files.
- Specs: `specs/` for feature specs and plans; GitHub workflow reads these.
- CI/automation: `.github/workflows/` and prompts under `.github/prompts/`.

## Build, Test, and Development Commands
- Environment: `poetry install` (installs deps from `pyproject.toml`).
- Lint: `poetry run ruff check .` (static checks + import order).
- Format: `poetry run black .` (100‑char line length).
- Types: `poetry run mypy src` (typing where practical; imports may be ignored).
- Tests: `poetry run pytest -v` (uses `tests/`, markers: `unit`, `integration`, `slow`).
- Package build (optional): `poetry build`.

## Coding Style & Naming Conventions
- Python 3.10. Black‑formatted (line length 100). Ruff rules: E,W,F,I,B,C4,UP; ignore E501, B008.
- Indentation: 4 spaces; one class/function per file unless trivial.
- Naming: modules `snake_case.py`, classes `CamelCase`, functions/vars `snake_case`, constants `UPPER_SNAKE`.
- Prefer type hints and docstrings on public functions. Keep functions small and testable.

## Testing Guidelines
- Framework: pytest. Place tests under `tests/` matching `test_*.py`; classes `Test*`; functions `test_*`.
- Use markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow` and filter via `-m`.
- Aim for meaningful coverage of core logic; add regression tests for bug fixes.

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`; imperative mood.
- Scope small, focused commits. Reference issues like `Closes #123` when applicable.
- PRs: include a brief description, rationale, screenshots/CLI output if relevant, and test coverage notes.

## Security & Configuration Tips
- Do not commit secrets. Use environment variables (e.g., `MLFLOW_TRACKING_URI`).
- Generated outputs: `mlruns/`, `outputs/`, `multirun/`, `.hydra/` are ignored (see `.gitignore`).
- Prefer devcontainer: open in VS Code Dev Containers for a consistent Poetry setup.

## Hydra/ML Workflow (when applicable)
- Put experiment configs in `configs/` and use Hydra overrides (e.g., `trainer.max_epochs=10`).
- Track runs with MLflow; keep artifacts out of Git. Use Optuna via `hydra-optuna-sweeper` for sweeps.

