#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around the Python gate checker for portability.
# Usage examples:
#   scripts/check_gates.sh --metrics outputs/evaluation/<run>/test_metrics.json \
#       --neg-precision-min 0.90 --criteria-auroc-min 0.80 --ece-max 0.05
#   scripts/check_gates.sh --val outputs/evaluation/<run>/val_metrics.json \
#       --test outputs/evaluation/<run>/test_metrics.json
#
# Note: Evidence F1 baseline +10 is project-dependent; pass --evidence-f1-min <value>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/check_gates.py" "$@"

