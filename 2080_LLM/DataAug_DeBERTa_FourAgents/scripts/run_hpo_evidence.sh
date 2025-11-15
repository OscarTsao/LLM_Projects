#!/usr/bin/env bash
set -euo pipefail

# Lightweight HPO runner for the evidence model

PYTHON=${PYTHON:-python3}

DATA_CONFIG=${DATA_CONFIG:-configs/data/redsm5.yaml}
EVIDENCE_CONFIG=${EVIDENCE_CONFIG:-configs/evidence/pairclf.yaml}
HPO_CONFIG=${HPO_CONFIG:-configs/hpo/evidence_pairclf.yaml}

exec "$PYTHON" -m src.hpo.evidence_hpo \
  --data-config "$DATA_CONFIG" \
  --evidence-config "$EVIDENCE_CONFIG" \
  --hpo-config "$HPO_CONFIG" \
  "$@"
