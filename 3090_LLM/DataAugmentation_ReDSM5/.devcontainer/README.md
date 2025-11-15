# Dev Container Configuration

## Overview

This dev container provides a complete development environment for ReDSM-5 Classification with:
- **PyTorch** with CUDA 12.1 support
- **MLflow** tracking server with PostgreSQL backend
- **DVC** for data version control
- **Miniforge** (conda/mamba) for package management
- **TensorBoard** and **Optuna Dashboard** for monitoring
- All ML libraries (Transformers, NLPAug, TextAttack, etc.)

## Architecture

The dev container uses **docker-compose** with 3 services:

1. **postgres**: PostgreSQL database for MLflow backend storage
2. **mlflow**: MLflow tracking server
3. **app**: Main development container (your workspace)

## GPU Support

The dev container is configured for **NVIDIA GPU** support via the `nvidia` runtime.

### Requirements

1. **Install NVIDIA Container Toolkit** (one-time setup on host):
   ```bash
   .devcontainer/setup-nvidia-docker.sh
   ```

2. **Ensure Docker supports nvidia runtime**:
   ```bash
   docker info | grep -i runtime  # Should show nvidia
   nvidia-smi  # Verify GPU is visible on host
   ```

3. **Rebuild container**:
   - Command Palette → "Dev Containers: Rebuild Container"

### Verifying GPU Access

Inside the dev container:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## Forwarded Ports

The following services are accessible from your host machine:

| Port | Service | URL | Description |
|------|---------|-----|-------------|
| 5000 | MLflow UI | http://localhost:5000 | Track experiments, view metrics, download models |
| 6006 | TensorBoard | http://localhost:6006 | Real-time training visualization |
| 8080 | Optuna Dashboard | http://localhost:8080 | Hyperparameter optimization monitoring |
| 8888 | Jupyter | http://localhost:8888 | Notebook interface (if started) |
| 5432 | PostgreSQL | localhost:5432 | MLflow backend database |

## Container Features

- **Python 3.11** with PyTorch 2.1+ (CUDA 12.1 compatible)
- **Miniforge** (mamba/conda) for package management
- **MLflow 2.9+** with PostgreSQL backend
- **DVC 3.0+** for data version control (with S3 support)
- **Pre-installed libraries**: transformers, nlpaug, textattack, optuna, hydra-core, tensorboard
- **Development tools**: black, flake8, isort, pytest, ruff, mypy
- **VSCode extensions**: Python, Pylance, Jupyter, Black formatter, Ruff
- **Git and GitHub CLI** configured
- **Auto-formatting** on save with Black and isort

## Environment Variables

The dev container sets these environment variables:

```bash
MLFLOW_TRACKING_URI=http://mlflow:5000       # MLflow server URL
POSTGRES_HOST=postgres                        # PostgreSQL hostname
POSTGRES_PORT=5432                            # PostgreSQL port
POSTGRES_USER=mlflow                          # Database user
POSTGRES_PASSWORD=mlflow                      # Database password
POSTGRES_DB=mlflow                            # Database name
PYTHONDONTWRITEBYTECODE=1                    # Disable .pyc files
PYTHONUNBUFFERED=1                           # Disable output buffering
DVC_NO_ANALYTICS=1                           # Disable DVC analytics
```

## Quick Start

1. **Open in Dev Container**:
   - Open this folder in VS Code
   - Command Palette (Ctrl+Shift+P) → "Dev Containers: Reopen in Container"
   - Wait for all services to start (postgres, mlflow, app)

2. **Verify everything is working**:
   ```bash
   # Check GPU
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

   # Check MLflow connection
   python -c "import os; print(f'MLflow URI: {os.getenv(\"MLFLOW_TRACKING_URI\")}')"

   # View available commands
   make help
   ```

3. **Access monitoring dashboards**:
   - MLflow UI: http://localhost:5000
   - Run `make tensorboard` for TensorBoard
   - Run `make optuna-dashboard` for Optuna Dashboard

## Troubleshooting

### Container fails to start with GPU error

If you see `could not select device driver "" with capabilities: [[gpu]]` or `nvidia runtime not found`:
- NVIDIA Container Toolkit is not installed
- Run `.devcontainer/setup-nvidia-docker.sh` on the host
- Verify with: `docker info | grep -i runtime`

### PyTorch not detecting GPU

- Verify NVIDIA drivers on host: `nvidia-smi`
- Check Docker runtime: `docker info | grep -i runtime` (should show `nvidia`)
- Rebuild container after installing NVIDIA Container Toolkit
- Inside container, check: `python -c "import torch; print(torch.cuda.is_available())"`

### MLflow UI not accessible

- Ensure MLflow service is running: `docker ps | grep mlflow`
- Check logs: `docker logs redsm5-mlflow`
- Restart services: `docker-compose -f .devcontainer/docker-compose.yml restart mlflow`

### PostgreSQL connection errors

- Check PostgreSQL is healthy: `docker ps | grep postgres`
- View logs: `docker logs redsm5-postgres`
- Verify connection from app container:
  ```bash
  psql -h postgres -U mlflow -d mlflow -c "SELECT 1;"
  ```

### Port already in use

If ports 5000, 6006, 8080, or 5432 are already in use:
- Stop conflicting services
- Or modify ports in `.devcontainer/docker-compose.yml`

## Data Persistence

The following data persists across container rebuilds:

- **postgres-data**: PostgreSQL database (MLflow tracking data)
- **mlflow-artifacts**: MLflow artifact storage (models, configs, etc.)
- **Workspace**: Your code and Data/ directory are mounted from host

To reset everything:
```bash
docker-compose -f .devcontainer/docker-compose.yml down -v  # Remove all volumes
```