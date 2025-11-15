# GPU Setup for Dev Container

## Problem
The dev container cannot access the GPU (`torch.cuda.is_available()` returns `False`).

## Root Cause
Docker daemon is not configured to use the NVIDIA runtime, even though:
- ✅ NVIDIA driver is installed (575.57.08, CUDA 12.9)
- ✅ nvidia-container-toolkit is installed (1.17.8)
- ✅ GPU is accessible on host (RTX 3090, 24GB)

## Solution

### Option 1: Automated Setup (Recommended)

Run the setup script **on the host** (exit the dev container first):

```bash
# Exit the container
exit

# Run the setup script
sudo bash setup_gpu_docker.sh
```

Then rebuild your dev container:
- Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
- Type "Dev Containers: Rebuild Container"
- Select it and wait for rebuild

### Option 2: Manual Setup

If you prefer to configure manually, run these commands **on the host**:

```bash
# 1. Configure Docker for NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# 2. Restart Docker
sudo systemctl restart docker

# 3. Verify configuration
docker info | grep -i runtime
# Should show "nvidia" in the list

# 4. Test GPU access
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

Then rebuild your dev container (same as Option 1).

## Verification

After rebuilding the container, verify GPU access:

```bash
# Inside the dev container
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
CUDA available: True
Device: NVIDIA GeForce RTX 3090
```

## Changes Made

1. **`.devcontainer/devcontainer.json`** - Added `"--runtime=nvidia"` to runArgs
2. **`setup_gpu_docker.sh`** - Created automated setup script

## Training Commands

Once GPU is accessible, both training commands will automatically use it:

```bash
make train-joint    # Joint training on GPU
make train-evidence # Evidence binding training on GPU
```

The training scripts automatically detect GPU and use it when available via:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Troubleshooting

### If GPU still not available after rebuild:

1. **Check Docker runtime configuration:**
   ```bash
   docker info | grep -i runtime
   ```
   Should show "nvidia" in the available runtimes.

2. **Check container GPU access:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```
   Should show your GPU (RTX 3090).

3. **Verify PyTorch installation in container:**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
   ```
   Should show PyTorch 2.6.0 with CUDA 12.4.

4. **Check devcontainer logs:**
   - Press `Ctrl+Shift+P` → "Dev Containers: Show Container Log"
   - Look for any GPU-related errors

### Common Issues:

- **"NVIDIA driver not found"**: Ensure NVIDIA drivers are installed on the host
- **"nvidia-container-toolkit not found"**: Install it following the manual setup instructions
- **"permission denied"**: Ensure your user is in the `docker` group: `sudo usermod -aG docker $USER`

## References

- [NVIDIA Container Toolkit Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
