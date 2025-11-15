#!/usr/bin/env bash
set -euo pipefail

python -m src.hpo \
  --backend optuna \
  --n_trials 100 \
  --timeout 43200 \
  --search_space configs/search_space.yaml \
  --config configs/base.yaml \
  --data_dir data/ \
  --labels configs/labels.yaml \
  --out_dir outputs/hpo_optuna
