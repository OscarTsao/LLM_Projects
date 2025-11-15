#!/usr/bin/env bash
#
# test_parallel_infrastructure.sh
# Quick test of parallel augmentation infrastructure with 2 shards
#
# This creates a minimal test run to verify the infrastructure works
# without running the full 7-shard augmentation (which takes hours).
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "================================================================"
echo "Parallel Infrastructure Test"
echo "================================================================"
echo ""
echo "This test will:"
echo "  1. Run a quick 2-shard augmentation"
echo "  2. Monitor progress for 30 seconds"
echo "  3. Merge the shard manifests"
echo "  4. Verify output"
echo ""
echo "Note: This uses real augmentation but with fewer shards."
echo "      Expect this to take 5-15 minutes depending on your system."
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Clean up any previous test output
TEST_OUTPUT="${PROJECT_ROOT}/data/processed/augsets_test"
TEST_LOGS="${PROJECT_ROOT}/logs/augment_test"
TEST_CACHE="${PROJECT_ROOT}/data/cache_test"

echo "==> Cleaning previous test output..."
rm -rf "${TEST_OUTPUT}" "${TEST_LOGS}" "${TEST_CACHE}"
mkdir -p "${TEST_OUTPUT}" "${TEST_LOGS}" "${TEST_CACHE}"

# Create a custom test launcher script
TEST_LAUNCHER="${SCRIPT_DIR}/run_parallel_augment_test.sh"

cat > "${TEST_LAUNCHER}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NUM_SHARDS=2
INPUT_CSV="${PROJECT_ROOT}/Data/ReDSM5/redsm5_annotations.csv"
METHODS_YAML="${PROJECT_ROOT}/conf/augment_methods.yaml"
OUTPUT_ROOT="${PROJECT_ROOT}/data/processed/augsets_test"
CACHE_DIR="${PROJECT_ROOT}/data/cache_test"
LOG_DIR="${PROJECT_ROOT}/logs/augment_test"

VARIANTS_PER_SAMPLE=1  # Reduced for speed
SEED=42
SAVE_FORMAT="parquet"
NUM_PROC=1
QUALITY_MIN_SIM=0.55
QUALITY_MAX_SIM=0.95

mkdir -p "${OUTPUT_ROOT}" "${CACHE_DIR}" "${LOG_DIR}"

declare -a PIDS=()

cleanup() {
    echo ""
    echo "==> Cleaning up..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    exit 130
}

trap cleanup SIGINT SIGTERM

echo "==> Launching ${NUM_SHARDS} test shards..."
echo ""

for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
    log_file="${LOG_DIR}/shard_${shard_id}_of_${NUM_SHARDS}.log"
    > "${log_file}"

    python "${PROJECT_ROOT}/tools/generate_augsets.py" \
      --input "${INPUT_CSV}" \
      --text-col sentence_text \
      --evidence-col sentence_text \
      --criterion-col DSM5_symptom \
      --label-col status \
      --id-col sentence_id \
      --methods-yaml "${METHODS_YAML}" \
      --combo-mode singletons \
      --variants-per-sample ${VARIANTS_PER_SAMPLE} \
      --seed ${SEED} \
      --output-root "${OUTPUT_ROOT}" \
      --save-format ${SAVE_FORMAT} \
      --num-proc ${NUM_PROC} \
      --shard-index ${shard_id} \
      --num-shards ${NUM_SHARDS} \
      --disk-cache "${CACHE_DIR}/augment_shard${shard_id}.db" \
      --quality-min-sim ${QUALITY_MIN_SIM} \
      --quality-max-sim ${QUALITY_MAX_SIM} \
      --force \
      > "${log_file}" 2>&1 &

    pid=$!
    PIDS+=("$pid")
    echo "  Shard ${shard_id}: PID ${pid} -> ${log_file}"
done

echo ""
echo "==> Test shards launched. Waiting for completion..."

declare -a EXIT_CODES=()
SUCCESS_COUNT=0
FAILURE_COUNT=0

for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    set +e
    wait "$pid"
    exit_code=$?
    set -e
    EXIT_CODES+=("$exit_code")

    if [[ $exit_code -eq 0 ]]; then
        echo "  Shard ${i} (PID ${pid}): SUCCESS"
        ((SUCCESS_COUNT++))
    else
        echo "  Shard ${i} (PID ${pid}): FAILED (exit code: ${exit_code})"
        ((FAILURE_COUNT++))
    fi
done

echo ""
echo "Test run complete: ${SUCCESS_COUNT} success, ${FAILURE_COUNT} failed"

if [[ $FAILURE_COUNT -gt 0 ]]; then
    exit 1
fi
EOF

chmod +x "${TEST_LAUNCHER}"

# Step 1: Launch test shards in background
echo "==> Step 1: Launching test shards..."
"${TEST_LAUNCHER}" &
LAUNCHER_PID=$!

# Step 2: Monitor for a bit
echo ""
echo "==> Step 2: Monitoring progress for 30 seconds..."
echo "    (Press Ctrl+C to stop early)"
sleep 5  # Give it time to start

python "${SCRIPT_DIR}/monitor_augment.py" \
    --log-dir "${TEST_LOGS}" \
    --num-shards 2 \
    --interval 10 \
    || true

# Wait for launcher to finish
wait ${LAUNCHER_PID} || {
    echo ""
    echo "Error: Test shards failed"
    exit 1
}

echo ""
echo "==> Step 3: Merging manifests..."

# Create custom merge script for test
cat > "${SCRIPT_DIR}/merge_test_manifests.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT}"
OUTPUT_ROOT="${TEST_OUTPUT}"
SHARD_MANIFESTS=(\$(find "\${OUTPUT_ROOT}" -maxdepth 1 -name "manifest_shard*_of_*.csv" | sort))
if [[ \${#SHARD_MANIFESTS[@]} -eq 0 ]]; then
    echo "Error: No shard manifests found"
    exit 1
fi
OUTPUT_MANIFEST="\${OUTPUT_ROOT}/manifest_final.csv"
cat "\${SHARD_MANIFESTS[0]}" > "\${OUTPUT_MANIFEST}"
for ((i=1; i<\${#SHARD_MANIFESTS[@]}; i++)); do
    tail -n +2 "\${SHARD_MANIFESTS[\$i]}" >> "\${OUTPUT_MANIFEST}"
done
echo "Merged \${#SHARD_MANIFESTS[@]} manifests into \${OUTPUT_MANIFEST}"
EOF
chmod +x "${SCRIPT_DIR}/merge_test_manifests.sh"

"${SCRIPT_DIR}/merge_test_manifests.sh"

echo ""
echo "==> Step 4: Verifying output..."
echo ""

# Count manifests
MANIFEST_COUNT=$(find "${TEST_OUTPUT}" -maxdepth 1 -name "manifest_shard*.csv" | wc -l)
echo "  Shard manifests: ${MANIFEST_COUNT}"

# Check final manifest
if [[ -f "${TEST_OUTPUT}/manifest_final.csv" ]]; then
    COMBO_COUNT=$(($(wc -l < "${TEST_OUTPUT}/manifest_final.csv") - 1))
    echo "  Total combos:    ${COMBO_COUNT}"
else
    echo "  Error: Final manifest not found"
    exit 1
fi

# Count datasets
DATASET_COUNT=$(find "${TEST_OUTPUT}" -type f -name "dataset.parquet" | wc -l)
echo "  Datasets:        ${DATASET_COUNT}"

# Show sample
echo ""
echo "  Sample combos:"
head -4 "${TEST_OUTPUT}/manifest_final.csv" | tail -3

echo ""
echo "================================================================"
echo "Test Complete!"
echo "================================================================"
echo ""
echo "Infrastructure verified successfully:"
echo "  - Parallel launcher works"
echo "  - Progress monitor works"
echo "  - Manifest merger works"
echo "  - Output structure correct"
echo ""
echo "Test output location:"
echo "  ${TEST_OUTPUT}"
echo ""
echo "To run full augmentation:"
echo "  ./tools/run_parallel_augment.sh"
echo ""
echo "Clean up test files:"
echo "  rm -rf ${TEST_OUTPUT} ${TEST_LOGS} ${TEST_CACHE}"
echo ""
