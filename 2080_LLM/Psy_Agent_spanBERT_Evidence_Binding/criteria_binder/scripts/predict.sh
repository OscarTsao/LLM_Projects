#!/bin/bash
# File: scripts/predict.sh

set -e

echo "Running prediction with Criteria Binder model..."

# Default values
CHECKPOINT_PATH="${CHECKPOINT_PATH:-outputs/run1/best}"
INPUT_FILE="${INPUT_FILE:-data/examples/test.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-outputs/predictions.jsonl}"
CONFIG_PATH="${CONFIG_PATH:-src/config}"
CONFIG_NAME="${CONFIG_NAME:-predict}"

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Please train a model first or specify correct checkpoint path."
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
    echo "Please specify a valid input JSONL file."
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Run prediction
python -m src.cli predict \
    predict.checkpoint="$CHECKPOINT_PATH" \
    predict.input_path="$INPUT_FILE" \
    predict.output_path="$OUTPUT_FILE" \
    --config-path "$CONFIG_PATH" \
    --config-name "$CONFIG_NAME"

echo "Prediction completed. Results saved to: $OUTPUT_FILE"