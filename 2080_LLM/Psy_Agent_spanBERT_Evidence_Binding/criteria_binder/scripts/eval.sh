#!/bin/bash
# File: scripts/eval.sh

set -e

echo "Evaluating Criteria Binder model..."

# Default values
CHECKPOINT_PATH="${CHECKPOINT_PATH:-outputs/run1/best}"
SPLIT="${SPLIT:-test}"
OUTPUT_FILE="${OUTPUT_FILE:-$CHECKPOINT_PATH/${SPLIT}_predictions.jsonl}"
CONFIG_PATH="${CONFIG_PATH:-src/config}"
CONFIG_NAME="${CONFIG_NAME:-eval}"

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Please train a model first or specify correct checkpoint path."
    exit 1
fi

# Run evaluation
python -m src.cli eval \
    eval.checkpoint="$CHECKPOINT_PATH" \
    eval.split="$SPLIT" \
    eval.output_path="$OUTPUT_FILE" \
    --config-path "$CONFIG_PATH" \
    --config-name "$CONFIG_NAME"

echo "Evaluation completed. Predictions saved to: $OUTPUT_FILE"