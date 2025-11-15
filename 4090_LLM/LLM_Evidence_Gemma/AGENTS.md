# Repository Guidelines

## Project Structure & Module Organization
- `src/` houses Python packages: data loaders (`src/data`), model wrappers (`src/models`), training & evaluation scripts (`src/training`), and reusable utilities (`src/utils`).  
- `conf/` contains Hydra experiment presets; `data/` holds ReDSM5 assets (`data/redsm5`).  
- Automation, dependencies, and docs live in `Makefile`, `requirements.txt`, `README.md` (high-level) and this guide. Keep new assets under `outputs/` to avoid polluting source trees.

## Build, Test, and Development Commands
- `make install` / `make install-dev` – editable install with (optional) tracking extras.  
- `make train`, `make train-quick`, `make train-5fold` – launch default single-fold, quick sanity, or 5-fold CV training.  
- `make evaluate CHECKPOINT=...` – run evaluation on a saved model.  
- `make lint`, `make format`, `make test` – flake8, black, and pytest suites. Prefer these wrappers over raw commands to ensure consistent flags.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, `black` line length 100, `flake8` ignores `E203/W503`.  
- Module names stay lowercase with underscores (`evidence_dataset.py`). Classes use `CamelCase`, functions/variables use `snake_case`.  
- Insert concise docstrings and comments only where logic is non-obvious. Avoid trailing whitespace and keep files ASCII unless Unicode already exists.

## Testing Guidelines
- Pytest under `tests/`; mirror source paths (`tests/models/test_gemma_qa.py`).  
- Add targeted unit tests for new tensor logic and quick regression cases. Smoke tests (e.g., `make train-quick`) should pass before merging. Strive for coverage on masking/pooling and serialization paths when touched.

## Commit & Pull Request Guidelines
- Follow conventional, action-oriented commit subjects (e.g., `feat: add bidirectional encoder mode`, `fix: guard tokenizer pad token`).  
- Pull requests must include: summary, validation evidence (command logs), linked issues, and notes on data/model artifacts produced. Attach sample outputs or screenshots for UI/logging changes. Keep PRs focused; prefer separate PRs for orthogonal features.

## Security & Configuration Tips
- Never commit secrets or raw patient data; keep credentials in environment variables or `.env` ignored files.  
- Validate GPU availability with `make check-gpu` and confirm data files via `make check-data` before large runs. Clean outputs (`make clean`) when rotating experiments to prevent stale checkpoints from being pushed.

