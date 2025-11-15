#!/usr/bin/env bash
set -euo pipefail

python -m src.train \
  --config configs/base.yaml \
  --data_dir data/ \
  --labels configs/labels.yaml \
  --out_dir outputs/run1 \
  --hf_id "" \
  --hf_config "" \
  --use_wandb false
