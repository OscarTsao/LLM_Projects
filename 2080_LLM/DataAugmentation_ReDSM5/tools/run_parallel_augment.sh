#!/usr/bin/env bash
#
# run_parallel_augment.sh
# Launches parallel augmentation generation across 7 shards
#
# Usage:
#   ./tools/run_parallel_augment.sh [OPTIONS]
#
# Options:
#   --dry-run       Show commands without executing
#   --num-shards N  Override number of shards (default: 7)
#   --help          Show this help message
#

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default settings
NUM_SHARDS=7
DRY_RUN=false

# Paths
INPUT_CSV="${PROJECT_ROOT}/Data/ReDSM5/redsm5_annotations.csv"
METHODS_YAML="${PROJECT_ROOT}/conf/augment_methods.yaml"
OUTPUT_ROOT="${PROJECT_ROOT}/data/processed/augsets"
CACHE_DIR="${PROJECT_ROOT}/data/cache"
LOG_DIR="${PROJECT_ROOT}/logs/augment"

# Augmentation settings
VARIANTS_PER_SAMPLE=2
SEED=42
SAVE_FORMAT="csv"
MAX_COMBO_SIZE=${MAX_COMBO_SIZE:-2}
NUM_PROC=1
QUALITY_MIN_SIM=0.55
QUALITY_MAX_SIM=0.95

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --num-shards)
            NUM_SHARDS="$2"
            shift 2
            ;;
        --help)
            head -n 15 "$0" | tail -n +2 | sed 's/^# *//'
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            echo "Run with --help for usage" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ ! -f "${INPUT_CSV}" ]]; then
    echo "Error: Input CSV not found: ${INPUT_CSV}" >&2
    exit 1
fi

if [[ ! -f "${METHODS_YAML}" ]]; then
    echo "Error: Methods YAML not found: ${METHODS_YAML}" >&2
    exit 1
fi

# Create required directories
mkdir -p "${OUTPUT_ROOT}" "${CACHE_DIR}" "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Cleanup handler
# ---------------------------------------------------------------------------

declare -a PIDS=()
declare -a SHARD_IDS=()

cleanup() {
    echo ""
    echo "==> Received interrupt signal. Cleaning up..."

    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "    Terminating process $pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done

    # Wait for all processes to terminate
    wait 2>/dev/null || true

    echo "==> Cleanup complete. Exiting."
    exit 130
}

trap cleanup SIGINT SIGTERM

# ---------------------------------------------------------------------------
# Build base command
# ---------------------------------------------------------------------------

build_command() {
    local shard_id=$1
    local log_file="${LOG_DIR}/shard_${shard_id}_of_${NUM_SHARDS}.log"

    cat <<EOF
python "${PROJECT_ROOT}/tools/generate_augsets.py" \\
  --input "${INPUT_CSV}" \\
  --text-col sentence_text \\
  --evidence-col sentence_text \\
  --criterion-col DSM5_symptom \\
  --label-col status \\
  --id-col sentence_id \\
  --methods-yaml "${METHODS_YAML}" \\
  --combo-mode bounded_k \\
  --max-combo-size ${MAX_COMBO_SIZE} \\
  --variants-per-sample ${VARIANTS_PER_SAMPLE} \\
  --seed ${SEED} \\
  --output-root "${OUTPUT_ROOT}" \\
  --save-format ${SAVE_FORMAT} \\
  --num-proc ${NUM_PROC} \\
  --shard-index ${shard_id} \\
  --num-shards ${NUM_SHARDS} \\
  --disk-cache "${CACHE_DIR}/augment_shard${shard_id}.db" \\
  --quality-min-sim ${QUALITY_MIN_SIM} \\
  --quality-max-sim ${QUALITY_MAX_SIM} \\
  --force \\
  > "${log_file}" 2>&1
EOF
}

# ---------------------------------------------------------------------------
# Launch shards
# ---------------------------------------------------------------------------

echo "================================================================"
echo "Parallel Augmentation Generation"
echo "================================================================"
echo ""
echo "Configuration:"
echo "  Number of shards: ${NUM_SHARDS}"
echo "  Input CSV:        ${INPUT_CSV}"
echo "  Methods YAML:     ${METHODS_YAML}"
echo "  Output root:      ${OUTPUT_ROOT}"
echo "  Log directory:    ${LOG_DIR}"
echo "  Variants/sample:  ${VARIANTS_PER_SAMPLE}"
echo "  Max combo size:   ${MAX_COMBO_SIZE}"
echo "  Quality range:    ${QUALITY_MIN_SIM} - ${QUALITY_MAX_SIM}"
echo ""

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "==> DRY RUN MODE: Commands will be printed but not executed"
    echo ""
fi

echo "==> Launching ${NUM_SHARDS} parallel shards..."
echo ""

for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
    log_file="${LOG_DIR}/shard_${shard_id}_of_${NUM_SHARDS}.log"

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "Shard ${shard_id}: (dry run)"
        build_command "$shard_id"
        echo ""
    else
        # Clear previous log
        > "${log_file}"

        # Launch shard in background
        eval "$(build_command "$shard_id")" &
        pid=$!
        PIDS+=("$pid")
        SHARD_IDS+=("$shard_id")

        echo "  Shard ${shard_id}: PID ${pid} -> ${log_file}"
    fi
done

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "==> Dry run complete. No processes launched."
    exit 0
fi

echo ""
echo "==> All shards launched. Waiting for completion..."
echo "    Monitor progress with: python tools/monitor_augment.py"
echo "    View logs in: ${LOG_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Wait for all processes and collect exit codes
# ---------------------------------------------------------------------------

declare -a EXIT_CODES=()
SUCCESS_COUNT=0
FAILURE_COUNT=0

for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    shard_id="${SHARD_IDS[$i]}"

    # Wait for this specific process
    set +e
    wait "$pid"
    exit_code=$?
    set -e

    EXIT_CODES+=("$exit_code")

    if [[ $exit_code -eq 0 ]]; then
        echo "  Shard ${shard_id} (PID ${pid}): SUCCESS"
        ((SUCCESS_COUNT++))
    else
        echo "  Shard ${shard_id} (PID ${pid}): FAILED (exit code: ${exit_code})"
        ((FAILURE_COUNT++))
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "Execution Summary"
echo "================================================================"
echo ""
echo "  Total shards:    ${NUM_SHARDS}"
echo "  Successful:      ${SUCCESS_COUNT}"
echo "  Failed:          ${FAILURE_COUNT}"
echo ""

if [[ $FAILURE_COUNT -gt 0 ]]; then
    echo "==> Some shards failed. Check logs in: ${LOG_DIR}"
    echo ""
    echo "Failed shards:"
    for i in "${!EXIT_CODES[@]}"; do
        if [[ ${EXIT_CODES[$i]} -ne 0 ]]; then
            echo "  - Shard ${SHARD_IDS[$i]}: ${LOG_DIR}/shard_${SHARD_IDS[$i]}_of_${NUM_SHARDS}.log"
        fi
    done
    echo ""
    exit 1
else
    echo "==> All shards completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Merge manifests: ./tools/merge_shard_manifests.sh"
    echo "  2. Check output in: ${OUTPUT_ROOT}"
    echo ""
    exit 0
fi
