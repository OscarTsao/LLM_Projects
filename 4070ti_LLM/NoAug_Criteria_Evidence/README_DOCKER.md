# Docker Setup

This project uses **VS Code Dev Containers** for all development and training.

## Why Dev Containers?

- ✅ Consistent environment across all developers
- ✅ CUDA GPU passthrough for training
- ✅ Pre-configured with Poetry, PyTorch, and all dependencies
- ✅ Integrated with VS Code for seamless development
- ✅ No manual Docker commands needed

## Setup

1. **Install Prerequisites:**
   - VS Code
   - Docker Desktop (with WSL2 on Windows)
   - "Dev Containers" extension for VS Code

2. **Open Project:**
   ```bash
   # Open folder in VS Code
   code /path/to/NoAug_Criteria_Evidence
   ```

3. **Start Dev Container:**
   - VS Code will detect `.devcontainer/devcontainer.json`
   - Click "Reopen in Container" when prompted
   - Container will build automatically (first time takes ~10 minutes)

4. **Verify Setup:**
   ```bash
   # Inside container
   python --version          # Python 3.10.x
   poetry --version          # Poetry 1.6.1
   nvidia-smi                # Check GPU access
   ```

## Container Configuration

Located in `.devcontainer/`:
- `devcontainer.json` - VS Code settings, extensions, GPU config
- `Dockerfile` - Image definition (CUDA 12.1 + Ubuntu 22.04)
- `docker-compose.yml` - Service composition

**Included:**
- CUDA 12.1.1
- Python 3.10
- Poetry 1.6.1
- PyTorch 2.1.2 (CUDA 12.1)
- All project dependencies

## Training with GPU

The Dev Container automatically configures GPU passthrough:

```bash
# Inside container - GPU is available
python -c "import torch; print(torch.cuda.is_available())"  # True

# Start training
make train TASK=criteria MODEL=roberta_base
```

## Rebuilding Container

If you modify `.devcontainer/Dockerfile`:

1. Command Palette (Ctrl+Shift+P)
2. "Dev Containers: Rebuild Container"

## Troubleshooting

### GPU Not Available
- Ensure Docker Desktop has GPU support enabled
- Check NVIDIA drivers on host: `nvidia-smi`
- Verify Docker GPU runtime: `docker run --rm --gpus all nvidia/cuda:12.1.1-base nvidia-smi`

### Container Build Failed
- Check internet connection
- Clear Docker cache: `docker system prune -a`
- Rebuild: "Dev Containers: Rebuild Container Without Cache"

### Poetry/Dependencies Issues
```bash
# Inside container
poetry install --no-interaction
poetry lock --no-update
```

## Alternative: Local Development

If you prefer not to use containers:

```bash
# On host machine
poetry install
make setup
```

Note: You'll need to manage CUDA, Python, and dependencies manually.

## Why No Root Docker Files?

Previously, this project had `Dockerfile` and `docker-compose.yml` in the root directory. These were redundant with `.devcontainer/` and have been removed to avoid confusion.

**Single Source of Truth:** All container configuration is in `.devcontainer/`

**For CI/CD:** If you need standalone Docker images for deployment, copy `.devcontainer/Dockerfile` to the root and modify as needed.
