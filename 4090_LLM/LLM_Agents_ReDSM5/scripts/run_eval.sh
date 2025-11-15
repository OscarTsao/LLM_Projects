#!/usr/bin/env bash
set -euo pipefail

python -m src.eval \
  --ckpt outputs/run1/best \
  --data_dir data/ \
  --labels configs/labels.yaml \
  --split test
