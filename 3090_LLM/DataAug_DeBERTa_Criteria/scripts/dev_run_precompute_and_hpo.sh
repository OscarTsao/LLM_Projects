#!/usr/bin/env bash
# Run augmentation precompute at max speed, then clean old HPO DBs and start a fresh HPO
# Intended to be run INSIDE the Dev Container terminal (user=vscode)

set -euo pipefail

echo "=========================================="
echo "Dev Container: Precompute + Fresh HPO"
echo "=========================================="

# Config (override via env vars before running)
WORKERS="${WORKERS:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-experiments}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-criteria_hpo_final}"
STUDY_NAME="${STUDY_NAME:-criteria_hpo_final}"
STUDY_DB="${STUDY_DB:-experiments/criteria_hpo_final.db}"

TRIALS_A="${TRIALS_A:-380}"
TRIALS_B="${TRIALS_B:-120}"
EPOCHS_A="${EPOCHS_A:-100}"
EPOCHS_B="${EPOCHS_B:-100}"

SKIP_PRECOMPUTE="${SKIP_PRECOMPUTE:-0}"
CLEAN_OLD="${CLEAN_OLD:-1}"

PYTHON=${PYTHON:-python}

mkdir -p "$EXPERIMENTS_DIR"

echo "Workspace: $(pwd)"
echo "Experiments dir: $EXPERIMENTS_DIR"
echo "Study DB: $STUDY_DB"
echo ""

# Ensure augmentation extras are installed (TextAttack + nlpaug)
echo "Checking augmentation dependencies (textattack, nlpaug)..."
if ! $PYTHON - <<'PY' 2>/dev/null
try:
    import textattack, nlpaug  # noqa: F401
    import sys; sys.exit(0)
except Exception:
    import sys; sys.exit(1)
PY
then
  echo "Installing augmentation extras via poetry..."
  if ! command -v poetry >/dev/null 2>&1; then
    echo "ERROR: poetry not found. Please install dependencies manually (poetry install -E augmentation)." >&2
    exit 1
  fi
  poetry install -E augmentation --no-interaction
else
  echo "✓ Augmentation deps present"
fi

# Precompute augmentation cache (max CPU) unless skipped
if [ "$SKIP_PRECOMPUTE" != "1" ]; then
  LOG_PRE="/tmp/precompute_parallel.log"
  echo ""
  echo "=========================================="
  echo "Starting parallel precompute (max CPU)"
  echo "Logging to: $LOG_PRE"
  echo "=========================================="

  if [ -z "$WORKERS" ]; then
    if command -v nproc >/dev/null 2>&1; then
      WORKERS=$(nproc)
    else
      WORKERS=$($PYTHON -c 'import os; print(os.cpu_count() or 1)')
    fi
  fi

  # Run in background, CPU-only, with full module path
  CUDA_VISIBLE_DEVICES='' nohup $PYTHON -u -c "import multiprocessing as mp; from scripts.precompute_augmentations_parallel import precompute_augmentations_parallel as run; run(num_workers=int($WORKERS), checkpoint_every=int($CHECKPOINT_EVERY))" \
    > "$LOG_PRE" 2>&1 &
  PRE_PID=$!
  disown || true

  echo "✓ Precompute started (PID: $PRE_PID)"
  echo "Tail logs: tail -f $LOG_PRE"
  echo "Check status: $PYTHON scripts/check_precompute_progress.py"

  echo "Waiting for cache to complete (this can take 45–90 min)..."
  while true; do
    if [ -f "$EXPERIMENTS_DIR/augmentation_cache.pkl" ] && [ -f "$EXPERIMENTS_DIR/augmentation_cache.json" ]; then
      echo "✓ Cache files detected: $EXPERIMENTS_DIR/augmentation_cache.{pkl,json}"
      break
    fi
    if ! ps -p $PRE_PID >/dev/null 2>&1; then
      echo "✗ Precompute process exited. Check $LOG_PRE for details." >&2
      exit 1
    fi
    sleep 30
  done
fi

echo ""
echo "=========================================="
echo "Cleaning old HPO study DBs"
echo "=========================================="
if [ "$CLEAN_OLD" = "1" ]; then
  # Only remove Optuna study DBs – do not touch MLflow DBs
  for db in \
    "$EXPERIMENTS_DIR/criteria_hpo.db" \
    "$EXPERIMENTS_DIR/criteria_hpo_v2.db" \
    "$EXPERIMENTS_DIR/criteria_hpo_final.db" \
    "$EXPERIMENTS_DIR/hpo_test.db" \
    ; do
    if [ -f "$db" ]; then
      echo "Removing: $db"
      rm -f "$db"
    fi
  done
  echo "✓ Old HPO DBs cleaned"
else
  echo "Skipped cleaning old DBs (CLEAN_OLD=$CLEAN_OLD)"
fi

echo ""
echo "=========================================="
echo "Starting fresh HPO run"
echo "=========================================="

LOG_HPO="/tmp/hpo_final.log"

# Ensure latest package is installed in editable mode
if command -v pip >/dev/null 2>&1; then
  echo "Installing local package (editable) to ensure latest code..."
  pip install -e . --no-deps >/dev/null 2>&1 || true
fi

nohup dataaug-train \
  --hpo \
  --experiment-name "$EXPERIMENT_NAME" \
  --study-name "$STUDY_NAME" \
  --study-db "$STUDY_DB" \
  --trials-a "$TRIALS_A" \
  --trials-b "$TRIALS_B" \
  --epochs-a "$EPOCHS_A" \
  --epochs-b "$EPOCHS_B" \
  > "$LOG_HPO" 2>&1 &
HPO_PID=$!
disown || true

echo "✓ HPO started (PID: $HPO_PID)"
echo "Logs: tail -f $LOG_HPO"
echo "Check running: pgrep -fa 'dataaug-train.*hpo'"
echo ""
echo "Done. HPO is running in background."

