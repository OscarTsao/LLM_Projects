# Repository Guidelines

## Project Structure & Module Organization
Refactored_Psy houses two production targets—`DataAug_Criteria_Evidence/` and `NoAug_Criteria_Evidence/`—each mirroring the same layout: `src/Project/` holds task-specific models (`Criteria/`, `Evidence/`, `Joint/`, `Share/`), `src/psy_agents_aug/` contains the augmentation-aware training application, `configs/` stores Hydra YAMLs grouped by domain (e.g., `configs/model/roberta_base.yaml`), `scripts/` provides automation entry points, and `tests/` captures regression suites. Datasets live under `data/redsm5/` and derived artifacts under `data/processed/`. Treat `psy-ref-repos/` as reference-only and keep edits inside the active targets.

## Build, Test, and Development Commands
Install dependencies with `poetry install` inside either target directory. Use `poetry run python -m psy_agents_aug.cli make_groundtruth data=hf_redsm5` to regenerate ground-truth CSVs and splits. Train agents via `poetry run python -m psy_agents_aug.cli train task=criteria augment.enabled=true`, and launch HPO with `poetry run python -m psy_agents_aug.cli hpo hpo=stage1_coarse`. Run the curated tests through `poetry run pytest` or focus on a module: `poetry run pytest tests/test_augment_pipelines.py`.

## Coding Style & Naming Conventions
Python 3.10, 4-space indentation, and 88-character lines are enforced by Black; pair with `poetry run black src tests`, `poetry run isort src tests`, and `poetry run ruff check .` before committing. Modules and Python files use `snake_case`, classes use PascalCase, and Hydra config filenames stay lowercase with underscores (e.g., `configs/task/criteria.yaml`). Keep augmentation pipeline identifiers aligned with their module names in `psy_agents_aug/augment/`.

## Testing Guidelines
All automated tests rely on PyTest (`python_files = "test_*.py"`). Mirror the existing naming convention when adding suites (`test_<area>.py`) and prefer fixtures that read from `data/processed/` rather than raw sources. For new augmentation or data loaders, clone the contract checks in `tests/test_augment_contract.py` to ensure leakage guards remain intact.

## Commit & Pull Request Guidelines
History currently holds a single `Initial commit`; maintainers expect succinct, imperative subjects (e.g., `feat: add hybrid augmentation guard`). Group related changes per commit and include before/after metrics when touching training or augmentation code. Pull requests should summarize behavioral impacts, list updated Hydra configs, and paste `poetry run pytest` output.

## Configuration & Data Tips
Hydra resolves configs relative to the active working directory—run CLI commands from within the target directory to avoid path mismatches. Large artifacts (`mlruns/`, `outputs/`, `.devcontainer/`) stay untracked; verify `.gitignore` covers new files before pushing. Store secrets and API keys in local `.env` files rather than committing them, and document any new environment variables in `docs/` for future agents.
