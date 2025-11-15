# Repository Guidelines
This guide keeps contributions aligned with the lightweight criteria baseline rebuild. Please scan it before opening a PR.

## Project Structure & Module Organization
- `config/` holds Hydra config groups (`dataset`, `model`, `training`, `mlflow`, `evaluation`) that compose the default run.
- `src/data.py` assembles augmented datasets and deterministic folds; extend data utilities here.
- `src/model.py` houses the DeBERTa backbone, MLP head, and loss implementations; create new architectures in sibling modules.
- `src/training.py` manages seeding, the training loop with Hydra configs, MLflow logging, and evaluation helpers; reuse its helpers in new entry points.
- `train.py` delegates to the training pipeline through Hydra; `evaluate.py` scores saved checkpoints with the same config stack.
- `Makefile` exposes `train-new`, `train-resume`, `evaluate`, and `mlflow-ui` shortcuts. Generated artifacts land under `outputs/` and MLflow stores data in `mlflow.db` / `mlruns/`.
- `outputs/.../state_latest.pt` stores the full training state (model + optimizer + scheduler + scaler) used for auto-resume.
- Validation and test splits are sampled only from `is_augmented == False` rows; augmentation data feeds the training split exclusively.
- Dataset assets (original ReDSM5 posts/annotations + augmentation CSVs) live under the local `data/` directory; treat them as read-only and version-controlled. Refer to `data/redsm5/README.md` for licensing guidance.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` sets up a local virtual environment.
- `pip install -r requirements.txt` installs the exact training stack.
- `make train-new` (alias: `make train`) starts a fresh run with auto-resume disabled; override config via `HYDRA_ARGS="training.num_epochs=3"`.
- `make train-resume` continues from `outputs/.../state_latest.pt`; override the checkpoint with `RESUME_CHECKPOINT=/path/to/state_latest.pt`.
- Toggle AMP via `training.use_amp=true|false`; the pipeline auto-falls back to FP32 if half precision is unstable.
- `make evaluate` evaluates the latest checkpoint; override the target via `EVAL_CHECKPOINT=/path/to/model.pt`.
- `make mlflow-ui` launches the local tracking dashboard (http://127.0.0.1:5000 by default). Export `MLFLOW_UI_PORT` if you need a different port.
Run commands from the repo root; set `HF_HOME` if you share Hugging Face caches across projects.

## Coding Style & Naming Conventions
Target Python 3.10+, four-space indentation, and type hints matching existing modules. Module docstrings should summarize purpose in one line. Favor explicit function names (`prepare_grouped_dataloader`) and snake_case variables. Mirror the current Black-like formatting and keep imports grouped as stdlib, third-party, then local. Avoid hard-coded paths; thread configuration through function parameters or dataclasses.

## Testing Guidelines
No formal unit suite exists; treat `train.py` on a filtered dataset and `evaluate.py` runs as regression smoke tests. When adding tests, create `tests/test_*.py` using `pytest`, mock GPU-only calls, and assert deterministic outputs. Record any new fixtures or data requirements in the PR description.

## Commit & Pull Request Guidelines
Write imperative, concise commit messages (e.g., `feat: add criterion loss ablation`). Bundle related changes per commit. Pull requests should explain motivation, summarize outcomes, and list verification steps (commands run, metrics). Link upstream tickets or experiments and attach metric tables or checkpoint paths when relevant.

## Data & Configuration Tips
Configuration derives from the Hydra stack in `config/`; update the relevant group (e.g., `training/baseline.yaml`) and document overrides in commits. Use Hydra CLI overrides (`python train.py training.learning_rate=3e-5`) for experiments rather than editing defaults. Coordinate schema or directory changes with teams consuming the synced `data/redsm5/` assets. Keep secrets out of configs; load credentials from environment variables or `.env` files ignored by Git.
