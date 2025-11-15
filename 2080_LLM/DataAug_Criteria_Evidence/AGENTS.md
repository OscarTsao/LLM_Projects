# Repository Guidelines

## Project Structure & Module Organization
- `src/psy_agents_noaug/`: Core package with data loaders, transformer encoders, training loop, and utilities. Module boundaries map to `data/`, `models/`, `training/`, `hpo/`, and `utils/`.
- `configs/`: Hydra configurations split into task, model, training, and HPO subfolders. Edit these to add new experiments instead of hard-coding parameters.
- `tests/`: Pytest suite covering loaders, training smoke checks, integration flows, and HPO configs. Match new code with targeted tests here.
- `outputs/`, `mlruns/`, `artifacts/`: Runtime artifacts. Keep them out of version control but inspect runs when debugging.

## Build, Test, and Development Commands
- `poetry install --with dev`: Install dependencies into the project virtualenv.
- `make test`: Run the full test suite (`pytest -v --tb=short`) inside the Poetry environment.
- `make train TASK=criteria MODEL=roberta_base`: Launch a training job with Hydra overrides; adjust `TASK`/`MODEL` as needed.
- `make hpo-s1`: Execute the coarse Optuna search defined in `configs/hpo/stage1_coarse.yaml`.
- `docker compose -f .devcontainer/docker-compose.yml up -d`: Start the GPU-enabled devcontainer + optional MLflow UI.

## Coding Style & Naming Conventions
- Python code follows Black (88-char lines), isort (Black profile), Ruff lint rules, and Bandit security checks. Run `make format` before pushing.
- Use descriptive module-level docstrings and favor snake_case for functions/variables, PascalCase for classes, and lowercase-with-dashes for config filenames.
- Keep configuration keys aligned with Hydra naming patterns (e.g., `model.encoder_name`, `training.batch_size`).

## Testing Guidelines
- Tests live under `tests/` and follow the `test_*.py` naming scheme. Pytest fixtures reside in `tests/conftest.py`.
- Add targeted unit tests for new utilities and smoke/integration tests for end-to-end flows. Aim to maintain existing coverage expectations (see `pyproject.toml` coverage settings).
- Run `make test` locally; for quick checks, scope to a file with `poetry run pytest tests/test_groundtruth.py -k new_case`.

## Commit & Pull Request Guidelines
- Write imperative, concise commit subjects (e.g., `Add ModernBERT encoders to HPO`). Reference ticket IDs when applicable.
- Each PR should summarize the change set, note testing performed, and link to MLflow runs or Optuna studies if relevant. Include screenshots or log snippets for UI/metrics changes.
- Ensure CI passes (lint + tests) before requesting review; rerun `make lint` and `make test` if the branch diverges.

## Devcontainer & GPU Tips
- Rebuild the devcontainer (`Dev Containers: Rebuild and Reopen`) after dependency or Dockerfile changes to ensure the CUDA stack refreshes.
- Verify GPU availability with `poetry run python -c "import torch; print(torch.cuda.is_available())"` before launching long runs. Use `TORCH_VERSION`/`TORCH_INDEX_URL` overrides in the compose file when upgrading CUDA.
