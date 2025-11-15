# Repository Guidelines

## Project Structure & Module Organization
Core pipeline code lives in `src/` (`augmentation/`, `data/`, `training/`, `utils/`). Hydra configs in `conf/` pair with trainers under `src/training` (e.g., `conf/dataset/original.yaml`). Scripts that regenerate datasets live in `scripts/`. Raw, ground-truth, and augmented corpora sit in `Data/`; track large files with DVC instead of git. Tests mirror the package layout in `tests/`, and experiment artefacts or logs stay in `outputs/` (do not commit).

## Build, Test, and Development Commands
Provision the environment with `make env-create` (or refresh via `make env-update`), then activate using `mamba activate redsm5`. Refresh datasets with `make augment-nlpaug`, `make augment-textattack`, or `make augment-hybrid`. Model workflows rely on `make train`, `make train-optuna`, and `make evaluate`; multi-agent variants such as `make train-criteria` target specialised trainers. Run `make lint` for Ruff checks, `make format` for Black, and `make test` (or `mamba run -n redsm5 pytest -v`) to execute unit tests. Use `python test_gpu.py` for a quick CUDA sanity check after driver or hardware changes.

## Coding Style & Naming Conventions
Python code uses four-space indentation and type hints. Maintain module names that reflect behaviour (`augmentation/textattack_pipeline.py`, `training/train_optuna.py`). Apply Black before committing and keep Ruff warnings clean. Hydra configs are lowercase with underscores (`original_nlpaug.yaml`) and should align with the dataset or model they configure. Tests follow `test_<module>.py` naming.

## Testing Guidelines
Pytest powers the suite under `tests/augmentation`, `tests/training`, `tests/utils`, and `tests/agents`. Add new tests alongside the module under test and prefix functions with `test_`. Run `make test` locally before pushing; add GPU coverage with `python test_gpu.py` whenever CUDA logic changes.

## Commit & Pull Request Guidelines
The repository has no published commit history yet, so adopt an imperative, present-tense style such as `Add hybrid augmentation pipeline`. Keep each commit scoped to a single concern and mention affected modules or configs in the body. Pull requests should summarise intent, list Hydra overrides or datasets required, attach key scores from MLflow or TensorBoard, and reference issues.

## Configuration Tips
Hydra composes from `conf/config.yaml`; override parameters with commands like `python -m src.training.train dataset=original_nlpaug`. Store secrets in environment variables rather than configuration files. Use DVC (`make dvc-init`, `make dvc-pull`, `make dvc-push`) when sharing datasets across agents.
