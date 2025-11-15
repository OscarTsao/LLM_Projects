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
- `make full-hpo-all`: Run the staged HPO pipeline (S0→S1→S2→Refit) across all agents.
- `make maximal-hpo-all`: Launch the single-stage “maximal” Optuna search for every agent.
- `make tune-evidence-max|aug|joint`: Stage-specific evidence HPO (Stage-A/B/C); accepts overrides such as `HPO_TRIALS`, `HPO_EPOCHS`, `HPO_SEEDS`.

### Monitoring & Troubleshooting HPO
- Use `tail -f outputs/logs/hpo_maximal.log` (or the stage-specific log) for live progress.
- GPU/RAM stats are appended to `hpo_monitor.log`; alerts and PID issues surface in `hpo_alerts.log`.
- `poetry run psy-agents show-best --agent <name> --study <study> --topk 5` inspects Optuna results without digging into the DB.

## Coding Style & Naming Conventions
- Python code follows Black (88-char lines), isort (Black profile), Ruff lint rules, and Bandit security checks. Run `make format` before pushing.
- Use descriptive module-level docstrings and favor snake_case for functions/variables, PascalCase for classes, and lowercase-with-dashes for config filenames.
- Keep configuration keys aligned with Hydra naming patterns (e.g., `model.encoder_name`, `training.batch_size`).

## Testing Guidelines
- Tests live under `tests/` and follow the `test_*.py` naming scheme. Pytest fixtures reside in `tests/conftest.py`.
- Add targeted unit tests for new utilities and smoke/integration tests for end-to-end flows. Aim to maintain existing coverage expectations (see `pyproject.toml` coverage settings).
- Run `make test` locally; for quick checks, scope to a file with `poetry run pytest tests/test_groundtruth.py -k new_case`.
- Augmentation/HPO specific tests:
  - `tests/test_augmentation_utils.py`, `tests/test_tfidf_cache.py`, `tests/test_seed_repro_workers.py` cover on-the-fly augmentation, TF-IDF caching, and dataloader seeding—update when touching those areas.
  - `tests/test_hpo_max_smoke.py` and `tests/test_hpo_stage_smoke.py` provide low-budget smoke runs; set `HPO_SMOKE_MODE=1` for deterministic/CPU-light execution during CI or rapid validation.

## Commit & Pull Request Guidelines
- Write imperative, concise commit subjects (e.g., `Add ModernBERT encoders to HPO`). Reference ticket IDs when applicable.
- Each PR should summarize the change set, note testing performed, and link to MLflow runs or Optuna studies if relevant. Include screenshots or log snippets for UI/metrics changes.
- Ensure CI passes (lint + tests) before requesting review; rerun `make lint` and `make test` if the branch diverges.

## GPU Tips
- Verify GPU availability with `poetry run python -c "import torch; print(torch.cuda.is_available())"` before launching long runs. Adjust `TORCH_VERSION`/`TORCH_INDEX_URL` in your environment when changing CUDA stacks.
- Keep the GPU fed during augmentation-heavy experiments by monitoring `perf.data_to_step_ratio` (logged to MLflow). Ratios above ~0.40 typically signal dataloader bottlenecks—tune `HPO_MAX_SAMPLES`, `ops_per_sample`, or `num_workers` when needed.
