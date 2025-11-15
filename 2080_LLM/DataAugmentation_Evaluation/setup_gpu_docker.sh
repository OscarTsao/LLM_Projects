#!/bin/bash
# Script to configure Docker for GPU access
# Run this on the HOST machine (not inside the container)

set -e

echo "=================================="
echo "Docker GPU Setup Script"
echo "=================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script needs to be run with sudo"
    echo "Usage: sudo bash setup_gpu_docker.sh"
    exit 1
fi

echo "1. Checking NVIDIA driver..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi
nvidia-smi --query-gpu=name --format=csv,noheader
echo "✓ NVIDIA driver found"
echo ""

echo "2. Checking nvidia-container-toolkit..."
if ! command -v nvidia-ctk &> /dev/null; then
    echo "ERROR: nvidia-container-toolkit not found."
    echo "Install it with:"
    echo "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo "  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y nvidia-container-toolkit"
    exit 1
fi
echo "✓ nvidia-container-toolkit version: $(nvidia-ctk --version | head -1)"
echo ""

echo "3. Configuring Docker daemon..."
# Backup existing config
if [ -f /etc/docker/daemon.json ]; then
    cp /etc/docker/daemon.json /etc/docker/daemon.json.backup
    echo "✓ Backed up existing daemon.json to daemon.json.backup"
fi

# Configure Docker daemon
nvidia-ctk runtime configure --runtime=docker
echo "✓ Docker daemon configured for NVIDIA runtime"
echo ""

echo "4. Restarting Docker..."
systemctl restart docker
sleep 2
echo "✓ Docker restarted"
echo ""

echo "5. Verifying configuration..."
if docker info | grep -q nvidia; then
    echo "✓ NVIDIA runtime is available"
else
    echo "⚠ Warning: NVIDIA runtime may not be properly configured"
fi
echo ""

echo "6. Testing GPU access in container..."
if docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "✓ GPU access working in Docker containers"
else
    echo "✗ GPU access test failed"
    echo "Try running: docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi"
    exit 1
fi
echo ""

echo "=================================="
echo "✅ Docker GPU setup complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Rebuild your dev container: Ctrl+Shift+P -> 'Dev Containers: Rebuild Container'"
echo "2. After rebuild, verify GPU access: python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
