#!/usr/bin/env bash
set -euo pipefail

# Attempt to install a CUDA-enabled PyTorch if NVIDIA GPU is available in the container.
detect_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  if [ -e /proc/driver/nvidia/version ] || [ -e /dev/nvidiactl ]; then
    return 0
  fi
  return 1
}

if detect_gpu; then
  echo "[install_torch] NVIDIA GPU detected; installing CUDA-enabled PyTorch (cu126)."
  python -m pip install --upgrade --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 || true
else
  echo "[install_torch] No GPU detected; installing CPU PyTorch."
  python -m pip install --upgrade --no-cache-dir torch torchvision torchaudio || true
fi
