# Repository Guidelines

## Project Structure & Module Organization
The repository centers on curated Reddit-derived data for DSM-5 symptom modeling. Place reusable code in `src/` with clear subpackages (for example, `src/augmentation/augmentor.py`) and keep exploratory notebooks under `notebooks/`. The `Data/` directory is canonical: `Data/Augmentation/` holds generated pairs, `Data/GroundTruth/` stores validated labels, and `Data/ReDSM5/` mirrors the raw corpus. Never overwrite records in place—add timestamped files when regenerating datasets.

## Build, Test, and Development Commands
Create an isolated environment before running scripts: `python -m venv .venv && source .venv/bin/activate`. Install dependencies with `python -m pip install -r requirements.txt` once your module defines them. Use `python -m build` to verify packaging metadata if you publish utilities. Run local sanity checks with `make data-preview` (add the target alongside any new scripts) to surface schema drift.

## Coding Style & Naming Conventions
Target Python 3.10+. Follow Black defaults (88 columns) and run `black src tests`. Complement formatting with `ruff check src tests` to catch lint issues. Use descriptive, lowercase module names (`augmentation_pipeline.py`) and PascalCase for classes modeling datasets or transformers. Prefer type hints for all public functions; enforce them with `mypy src` before opening a pull request.

## Testing Guidelines
All executable logic must include pytest coverage. Mirror package paths inside `tests/` (e.g., `tests/augmentation/test_augmentor.py`) and name tests after the behavior under scrutiny. Include lightweight fixtures for sample rows drawn from `Data/GroundTruth/Final_Ground_Truth.json` and keep larger fixtures under `tests/fixtures/`. A pull request should pass `pytest --maxfail=1 --disable-warnings` and document any skipped tests.

## Commit & Pull Request Guidelines
Use Conventional Commit prefixes (`feat:`, `fix:`, `data:`) so downstream release tooling remains predictable. Each commit should touch either code or data but not both, unless they must change atomically. Pull requests need: a concise summary, affected datasets, reproducibility steps, and metrics or counts verifying data integrity. Link to related issues and attach CSV diffs or summary tables when altering dataset content.

## Data Handling & Security
Treat Reddit source data as sensitive. Strip user identifiers before committing files and document any anonymization steps. Large artifacts belong in object storage—commit only derived samples or manifests. Validate licenses for external synonym banks and record provenance in a `docs/data-sources.md` entry.
