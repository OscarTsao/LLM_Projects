# Repository Guidelines

## Project Structure & Module Organization
- Core source lives in `src/dataaug_multi_both/`; place new modules here so imports stay consistent with existing packages.
- Hydra configs sit under `configs/`, e.g., `configs/train.yaml` for run defaults and `configs/model/*.yaml` for architecture overrides.
- Tests mirror modules in `tests/` using `test_<module>.py`; add fixtures locally to avoid cross-test coupling.
- Store datasets, checkpoints, and MLflow artifacts in `Data/`; keep generated folders (`mlruns/`, `outputs/`, `multirun/`, `.hydra/`) out of version control.

## Build, Test, and Development Commands
- `poetry install` prepares the virtualenv defined by `pyproject.toml`.
- `poetry run ruff check .` enforces lint rules and import order.
- `poetry run black .` formats Python files to the 100-character profile.
- `poetry run mypy src` runs optional static typing on the package.
- `poetry run pytest -v` executes the full test suite; add `-m unit` or `-m slow` to scope runs.

## Coding Style & Naming Conventions
- Target Python 3.10 with 4-space indentation; keep functions focused and cohesive.
- Use `snake_case` for modules and functions, `CamelCase` for classes, and `UPPER_SNAKE` for constants.
- Follow Black formatting and Ruff rules (E,W,F,I,B,C4,UP; ignores E501, B008); resolve warnings before opening a PR.
- Add lightweight docstrings on public APIs and only brief comments when logic is non-obvious.

## Testing Guidelines
- Write pytest tests alongside related modules in `tests/`; name files `test_<module>.py`.
- Mark tests with `@pytest.mark.unit`, `integration`, or `slow` to guide CI selection.
- Favor deterministic fixtures that stub external services; avoid hitting real data sources.
- Run `poetry run pytest -v` locally before submitting changes and include failing seeds if fixing regressions.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (e.g., `feat: add augmenter registry`, `fix: handle empty dataset edge case`) in the imperative mood.
- Reference related issues with `Closes #123` when applicable and summarize behavioral impacts in the PR body.
- Document relevant command outputs or screenshots, especially for new Hydra overrides or trainer configs.
- Confirm linting and tests have run locally; note any gaps or follow-ups in the PR description.

## Security & Configuration Tips
- Never commit secrets; configure endpoints via environment variables such as `MLFLOW_TRACKING_URI`.
- Keep experiments reproducible with Hydra overrides (`trainer.max_epochs=10`, `model.name=deberta`), and track MLflow runs outside the repo.
- Prefer the provided devcontainer or a Poetry-managed environment to ensure dependency parity across contributors.
