#!/usr/bin/env bash
# Launch TensorBoard for monitoring training runs

set -e

LOG_DIR="${1:-logs}"

echo "Starting TensorBoard..."
echo "Log directory: ${LOG_DIR}"
echo "Access TensorBoard at: http://localhost:6006"

tensorboard --logdir="${LOG_DIR}" --host=0.0.0.0 --port=6006
