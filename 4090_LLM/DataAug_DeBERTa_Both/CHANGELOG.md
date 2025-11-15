# Changelog

## 2025-10-20
- Introduced two-stage Optuna HPO pipeline with conditional search space narrowing.
- Added CLI entry point `python -m dataaug_multi_both.cli.hpo` with `stage1`, `stage2`, and `report` commands.
- Logged evaluation threshold tuning in per-trial reports and exposed export script for best configurations.
