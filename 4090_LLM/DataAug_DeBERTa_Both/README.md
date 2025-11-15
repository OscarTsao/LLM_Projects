# DataAug Multi Both

Storage-optimised, two-stage HPO pipeline for mental-health criteria detection.

## Environment & Tooling
- Install dependencies: `poetry install`
- Check environment capabilities (CUDA / bf16 / Optuna): `make doctor`
- Lint & types: `make lint`
- Format: `make format`
- Tests: `make test`

## Two-Stage Optuna Workflow
- **Stage 1 (broad / cheap)**
  - 350 trials by default, 50% data slice, 2 epochs, aggressive ASHA pruning.
  - Run: `python -m dataaug_multi_both.cli.hpo stage1 --storage sqlite:///experiments/optuna.db --trials 350 --jobs 4`
- **Stage 2 (narrow / full)**
  - 150 trials by default, full dataset, 8-epoch budget, Median pruner.
  - Automatically narrows continuous ranges (Q10–Q90) and keeps top categorical choices from Stage 1 (top-50 trials).
  - Run: `python -m dataaug_multi_both.cli.hpo stage2 --storage sqlite:///experiments/optuna.db --trials 150 --jobs 4`
- Convenience targets:
  - `make stage1 ARGS="--storage sqlite:///experiments/optuna.db --trials 8"`
  - `make stage2 ARGS="--storage sqlite:///experiments/optuna.db --trials 8"`
  - `make hpo-report ARGS="--storage sqlite:///experiments/optuna.db --study-name stage2_exploit"`

## Search Space Highlights
- Conditional knobs for optimizer-specific momentum, scheduler cycle length, focal hyperparameters, and augmentation tweaks.
- AMP precision (`fp16`/`bf16`) gated by hardware probe with automatic fallback.
- Batch/accumulation combinations enforced so `batch_size * gradient_accumulation_steps ∈ {16,32,64,128}`.
- Stage 2 optionally toggles per-class threshold tuning for top configurations.

## Artifacts & Thresholding
- Each trial writes `experiments/trial_<id>/evaluation_report.json` plus minimalist checkpoints under `checkpoints/`.
- Reports include macro-F1, tuned thresholds (global or per-class), effective learning rate, and storage for reproducibility.
- Use `scripts/export_best.py --storage sqlite:///experiments/optuna.db --study-name stage2_exploit` to re-simulate the best Stage 2 configuration across multiple seeds (default 3) and emit `export_summary.json`.

## Quick Smoke Test
- Validate the end-to-end simulation on CPU: `poetry run pytest -q tests/test_two_stage_smoke.py`

## Migration Note
- The old `cli.train` HPO stubs are superseded by `python -m dataaug_multi_both.cli.hpo ...`.
- Existing Optuna studies remain compatible—Stage 2 automatically derives narrowed bounds from stored Stage 1 trials.

## Changelog
- feat: add conditional Stage 1/Stage 2 search space, two-stage runner, threshold tuning utilities, and Optuna CLI (2025-10-20).
