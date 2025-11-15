#!/usr/bin/env bash
#
# progressive_k_runner.sh
# Progressive k-value augmentation generator with decision gates
#
# Usage:
#   ./tools/progressive_k_runner.sh [OPTIONS]
#
# Options:
#   --start-k N         Starting k value (default: 2)
#   --end-k N           Ending k value (default: 4)
#   --auto-approve      Skip confirmation prompts between phases
#   --num-shards N      Number of parallel shards (default: 7)
#   --help              Show this help message
#
# Environment variables:
#   START_K             Starting k value (default: 2)
#   END_K               Ending k value (default: 4)
#   AUTO_APPROVE        Auto-approve phases (default: false)
#   NUM_SHARDS          Number of shards (default: 7)
#

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default settings
START_K=${START_K:-2}
END_K=${END_K:-4}
AUTO_APPROVE=${AUTO_APPROVE:-false}
NUM_SHARDS=${NUM_SHARDS:-7}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --start-k)
            START_K="$2"
            shift 2
            ;;
        --end-k)
            END_K="$2"
            shift 2
            ;;
        --auto-approve)
            AUTO_APPROVE=true
            shift
            ;;
        --num-shards)
            NUM_SHARDS="$2"
            shift 2
            ;;
        --help)
            head -n 20 "$0" | tail -n +2 | sed 's/^# *//'
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

if [[ ${START_K} -lt 1 ]]; then
    echo "Error: START_K must be >= 1" >&2
    exit 1
fi

if [[ ${END_K} -lt ${START_K} ]]; then
    echo "Error: END_K must be >= START_K" >&2
    exit 1
fi

if [[ ${END_K} -gt 28 ]]; then
    echo "Warning: k values > 28 may generate extremely large datasets" >&2
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

echo "================================================================"
echo "Progressive K-Value Augmentation Generator"
echo "================================================================"
echo ""
echo "Configuration:"
echo "  k range:          ${START_K} to ${END_K}"
echo "  Num shards:       ${NUM_SHARDS}"
echo "  Auto-approve:     ${AUTO_APPROVE}"
echo "  Project root:     ${PROJECT_ROOT}"
echo ""

# ---------------------------------------------------------------------------
# Phase execution
# ---------------------------------------------------------------------------

TOTAL_PHASES=$((END_K - START_K + 1))
CURRENT_PHASE=0

for k in $(seq ${START_K} ${END_K}); do
    ((CURRENT_PHASE++))

    echo ""
    echo "================================================================"
    echo "Phase ${CURRENT_PHASE}/${TOTAL_PHASES}: k=${k} Combination Generation"
    echo "================================================================"
    echo ""

    # Export environment variables for child script
    export MAX_COMBO_SIZE=${k}
    export SAVE_FORMAT=csv

    # Run parallel augmentation generation
    echo "==> Step 1: Starting k=${k} generation..."
    echo "    Command: ./tools/run_parallel_augment.sh --num-shards ${NUM_SHARDS}"
    echo ""

    if ! "${SCRIPT_DIR}/run_parallel_augment.sh" --num-shards ${NUM_SHARDS}; then
        echo ""
        echo "ERROR: k=${k} generation failed!" >&2
        echo "Check logs in: ${PROJECT_ROOT}/logs/augment/" >&2
        exit 1
    fi

    echo ""
    echo "==> Step 2: Merging k=${k} manifests..."

    if ! "${SCRIPT_DIR}/merge_shard_manifests.sh"; then
        echo ""
        echo "ERROR: k=${k} manifest merge failed!" >&2
        exit 1
    fi

    echo ""
    echo "==> Step 3: Validating k=${k} outputs..."

    # Basic validation
    MANIFEST_PATH="${PROJECT_ROOT}/data/processed/augsets/manifest_final.csv"
    if [[ ! -f "${MANIFEST_PATH}" ]]; then
        echo "ERROR: Final manifest not found: ${MANIFEST_PATH}" >&2
        exit 1
    fi

    COMBO_COUNT=$(tail -n +2 "${MANIFEST_PATH}" | wc -l)
    echo "    Combos in manifest: ${COMBO_COUNT}"

    # Check for k=${k} entries
    K_COMBOS=$(tail -n +2 "${MANIFEST_PATH}" | awk -F',' -v k=${k} '$3 == k' | wc -l)
    echo "    k=${k} combos:      ${K_COMBOS}"

    if [[ ${K_COMBOS} -eq 0 ]]; then
        echo "WARNING: No k=${k} combos found in manifest!" >&2
    fi

    echo ""
    echo "==> Phase ${CURRENT_PHASE}/${TOTAL_PHASES} complete (k=${k})"

    # Decision gate (skip for last phase)
    if [[ ${k} -lt ${END_K} ]]; then
        echo ""
        echo "----------------------------------------------------------------"

        if [[ "${AUTO_APPROVE}" != "true" ]]; then
            echo "k=${k} phase complete. Review outputs before proceeding."
            echo ""
            read -p "Proceed to k=$((k+1))? [y/N] " -n 1 -r
            echo

            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo ""
                echo "==> Stopping at k=${k} (user request)"
                echo ""
                echo "To resume from k=$((k+1)), run:"
                echo "  START_K=$((k+1)) END_K=${END_K} ./tools/progressive_k_runner.sh"
                echo ""
                exit 0
            fi
        else
            echo "Auto-approve enabled. Proceeding to k=$((k+1))..."
        fi
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "All Phases Complete"
echo "================================================================"
echo ""
echo "Generated combinations for k=${START_K} through k=${END_K}"
echo ""
echo "Next steps:"
echo "  1. Review outputs in: ${PROJECT_ROOT}/data/processed/augsets/"
echo "  2. Check final manifest: ${MANIFEST_PATH}"
echo "  3. Train models using augmented datasets"
echo ""

# Display final statistics
if [[ -f "${MANIFEST_PATH}" ]]; then
    echo "Final Statistics:"
    echo "  Total combos: $(tail -n +2 "${MANIFEST_PATH}" | wc -l)"
    echo ""
    echo "  Breakdown by k:"
    tail -n +2 "${MANIFEST_PATH}" | awk -F',' '{print $3}' | sort -n | uniq -c | while read count k_val; do
        printf "    k=%-2s : %4d combos\n" "$k_val" "$count"
    done
    echo ""
fi

echo "==> Complete!"
echo ""
