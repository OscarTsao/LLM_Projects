#!/usr/bin/env bash
#
# Re-run only failed combo IDs from a previous augmentation run
#
# Usage:
#   ./tools/rerun_failed_combos.sh <combo_id_1> <combo_id_2> ...
# Or:
#   ./tools/rerun_failed_combos.sh --from-logs logs/augment/*.log
#

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
INPUT_CSV="${INPUT_CSV:-Data/ReDSM5/redsm5_annotations.csv}"
METHODS_YAML="${METHODS_YAML:-conf/augment_methods.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/processed/augsets}"
LOG_DIR="${LOG_DIR:-logs/augment}"
MAX_COMBO_SIZE="${MAX_COMBO_SIZE:-2}"
SAVE_FORMAT="${SAVE_FORMAT:-csv}"

mkdir -p "$LOG_DIR"

# Parse arguments
COMBO_IDS=()

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <combo_id_1> <combo_id_2> ..."
    echo "   Or: $0 --from-logs logs/augment/*.log"
    exit 1
fi

if [[ "$1" == "--from-logs" ]]; then
    shift
    echo "==> Extracting failed combo IDs from logs..."
    for log_file in "$@"; do
        grep -oP "Combo \K[a-f0-9]{10}(?= failed)" "$log_file" || true
    done | sort -u > /tmp/failed_combos.txt
    mapfile -t COMBO_IDS < /tmp/failed_combos.txt
    echo "Found ${#COMBO_IDS[@]} failed combos"
else
    COMBO_IDS=("$@")
fi

if [[ ${#COMBO_IDS[@]} -eq 0 ]]; then
    echo "No failed combos to process"
    exit 0
fi

echo "================================================================"
echo "Re-running ${#COMBO_IDS[@]} Failed Combos"
echo "================================================================"
echo ""
echo "Configuration:"
echo "  Combo IDs:        ${COMBO_IDS[*]}"
echo "  Save format:      $SAVE_FORMAT"
echo "  Input CSV:        $INPUT_CSV"
echo "  Methods YAML:     $METHODS_YAML"
echo "  Output root:      $OUTPUT_ROOT"
echo "  Log directory:    $LOG_DIR"
echo ""

# Create combo ID filter file
COMBO_FILTER="/tmp/failed_combo_ids.txt"
printf "%s\n" "${COMBO_IDS[@]}" > "$COMBO_FILTER"

# Run augmentation for failed combos only
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/rerun_failed_${TIMESTAMP}.log"

echo "==> Running augmentation..."
python -m tools.generate_augsets \
    --input-csv "$INPUT_CSV" \
    --methods-yaml "$METHODS_YAML" \
    --output-root "$OUTPUT_ROOT" \
    --combo-mode bounded_k \
    --max-combo-size "$MAX_COMBO_SIZE" \
    --combo-id-filter "$COMBO_FILTER" \
    --variants-per-sample 2 \
    --min-quality 0.55 \
    --max-quality 0.95 \
    --seed 42 \
    --save-format "$SAVE_FORMAT" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "================================================================"
echo "Re-run Complete"
echo "================================================================"
echo "Log: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Check for errors: grep -i error $LOG_FILE"
echo "  2. Merge manifests: ./tools/merge_shard_manifests.sh"
echo "  3. Validate results: wc -l $OUTPUT_ROOT/manifest_final.csv"
