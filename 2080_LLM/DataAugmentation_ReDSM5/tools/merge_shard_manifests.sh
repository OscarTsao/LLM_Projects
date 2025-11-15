#!/usr/bin/env bash
#
# merge_shard_manifests.sh
# Merges shard manifests into a single final manifest
#
# Usage:
#   ./tools/merge_shard_manifests.sh [OPTIONS]
#
# Options:
#   --output-root PATH    Path to augmented datasets (default: data/processed/augsets)
#   --output-name NAME    Output manifest name (default: manifest_final.csv)
#   --keep-shards         Keep individual shard manifests after merging
#   --help                Show this help message
#

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default settings
OUTPUT_ROOT="${PROJECT_ROOT}/data/processed/augsets"
OUTPUT_NAME="manifest_final.csv"
KEEP_SHARDS=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --output-name)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --keep-shards)
            KEEP_SHARDS=true
            shift
            ;;
        --help)
            head -n 16 "$0" | tail -n +2 | sed 's/^# *//'
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

if [[ ! -d "${OUTPUT_ROOT}" ]]; then
    echo "Error: Output directory not found: ${OUTPUT_ROOT}" >&2
    exit 1
fi

OUTPUT_MANIFEST="${OUTPUT_ROOT}/${OUTPUT_NAME}"

# ---------------------------------------------------------------------------
# Find shard manifests
# ---------------------------------------------------------------------------

echo "================================================================"
echo "Manifest Merger"
echo "================================================================"
echo ""
echo "Configuration:"
echo "  Output root:   ${OUTPUT_ROOT}"
echo "  Output file:   ${OUTPUT_MANIFEST}"
echo "  Keep shards:   ${KEEP_SHARDS}"
echo ""

# Find all shard manifest files
SHARD_MANIFESTS=($(find "${OUTPUT_ROOT}" -maxdepth 1 -name "manifest_shard*_of_*.csv" | sort))

if [[ ${#SHARD_MANIFESTS[@]} -eq 0 ]]; then
    echo "Error: No shard manifests found in ${OUTPUT_ROOT}" >&2
    echo "Expected pattern: manifest_shard*_of_*.csv" >&2
    exit 1
fi

echo "==> Found ${#SHARD_MANIFESTS[@]} shard manifests:"
for manifest in "${SHARD_MANIFESTS[@]}"; do
    echo "    - $(basename "${manifest}")"
done
echo ""

# ---------------------------------------------------------------------------
# Merge manifests
# ---------------------------------------------------------------------------

echo "==> Merging manifests..."

# Create temporary file
TEMP_FILE=$(mktemp)
trap "rm -f ${TEMP_FILE}" EXIT

# Process first manifest (including header)
FIRST_MANIFEST="${SHARD_MANIFESTS[0]}"
if [[ ! -f "${FIRST_MANIFEST}" ]]; then
    echo "Error: First manifest not found: ${FIRST_MANIFEST}" >&2
    exit 1
fi

cat "${FIRST_MANIFEST}" > "${TEMP_FILE}"
echo "    Added: $(basename "${FIRST_MANIFEST}") (with header)"

# Process remaining manifests (skip headers)
for ((i=1; i<${#SHARD_MANIFESTS[@]}; i++)); do
    manifest="${SHARD_MANIFESTS[$i]}"

    if [[ ! -f "${manifest}" ]]; then
        echo "    Warning: Manifest not found, skipping: ${manifest}" >&2
        continue
    fi

    # Skip header line (first line)
    tail -n +2 "${manifest}" >> "${TEMP_FILE}"
    echo "    Added: $(basename "${manifest}") (data only)"
done

# Move to final location
mv "${TEMP_FILE}" "${OUTPUT_MANIFEST}"

echo ""
echo "==> Merged manifest created: ${OUTPUT_MANIFEST}"

# ---------------------------------------------------------------------------
# Generate summary
# ---------------------------------------------------------------------------

echo ""
echo "==> Generating summary..."

# Count total lines (excluding header)
TOTAL_COMBOS=$(($(wc -l < "${OUTPUT_MANIFEST}") - 1))

# Calculate total rows across all datasets
TOTAL_ROWS=0
if command -v awk &> /dev/null; then
    # Sum the 'rows' column (assuming it's the 4th column based on generate_augsets.py)
    TOTAL_ROWS=$(awk -F',' 'NR>1 && $4 ~ /^[0-9]+$/ {sum+=$4} END {print sum}' "${OUTPUT_MANIFEST}")
fi

echo ""
echo "================================================================"
echo "Summary"
echo "================================================================"
echo ""
echo "  Shard manifests merged:  ${#SHARD_MANIFESTS[@]}"
echo "  Total combos:            ${TOTAL_COMBOS}"

if [[ ${TOTAL_ROWS} -gt 0 ]]; then
    echo "  Total augmented rows:    ${TOTAL_ROWS}"
fi

echo ""

# Show first few entries
if command -v head &> /dev/null && command -v column &> /dev/null; then
    echo "First 5 entries:"
    echo ""
    head -6 "${OUTPUT_MANIFEST}" | column -t -s','
    echo ""
fi

# ---------------------------------------------------------------------------
# Cleanup shard manifests
# ---------------------------------------------------------------------------

if [[ "${KEEP_SHARDS}" == "false" ]]; then
    echo "==> Cleaning up individual shard manifests..."

    for manifest in "${SHARD_MANIFESTS[@]}"; do
        rm -f "${manifest}"
        echo "    Removed: $(basename "${manifest}")"
    done

    echo ""
    echo "==> Cleanup complete"
else
    echo "==> Keeping individual shard manifests (--keep-shards specified)"
fi

# ---------------------------------------------------------------------------
# Detailed statistics
# ---------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "Detailed Statistics"
echo "================================================================"
echo ""

# Count combos by size (k=1, k=2, etc.)
if command -v awk &> /dev/null; then
    echo "Combos by size:"
    awk -F',' 'NR>1 {print $3}' "${OUTPUT_MANIFEST}" | sort | uniq -c | while read count size; do
        printf "  k=%-2s : %3d combos\n" "$size" "$count"
    done
    echo ""
fi

# Show combo statistics
if command -v awk &> /dev/null; then
    echo "Dataset statistics:"

    # Get min/max/avg rows per combo
    STATS=$(awk -F',' 'NR>1 && $4 ~ /^[0-9]+$/ {
        rows=$4;
        sum+=rows;
        count++;
        if(NR==2 || rows<min) min=rows;
        if(NR==2 || rows>max) max=rows;
    }
    END {
        if(count>0) printf "%.0f,%.0f,%.0f", min, max, sum/count;
        else print "0,0,0"
    }' "${OUTPUT_MANIFEST}")

    IFS=',' read -r MIN_ROWS MAX_ROWS AVG_ROWS <<< "$STATS"

    if [[ ${MIN_ROWS} -gt 0 ]]; then
        echo "  Min rows per combo:  ${MIN_ROWS}"
        echo "  Max rows per combo:  ${MAX_ROWS}"
        echo "  Avg rows per combo:  ${AVG_ROWS}"
    fi
fi

echo ""
echo "==> Complete! Final manifest: ${OUTPUT_MANIFEST}"
echo ""
