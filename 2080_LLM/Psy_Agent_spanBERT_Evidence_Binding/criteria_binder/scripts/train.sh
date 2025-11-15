#!/bin/bash
# File: scripts/train.sh

set -e

echo "Training Criteria Binder model..."

# Default values
CONFIG_PATH="${CONFIG_PATH:-src/config}"
CONFIG_NAME="${CONFIG_NAME:-train}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/run1}"
MODEL_NAME="${MODEL_NAME:-SpanBERT/spanbert-base-cased}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
python -m src.cli train \
    --config-path "$CONFIG_PATH" \
    --config-name "$CONFIG_NAME" \
    model.name="$MODEL_NAME" \
    logging.output_dir="$OUTPUT_DIR"

echo "Training completed. Results saved to: $OUTPUT_DIR"
echo "Best checkpoint: $OUTPUT_DIR/best"