FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Install additional system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    nano \
    htop \
    tree \
    zip \
    unzip \
    sudo \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Ensure system Python is used (not host conda/mamba if mounted)
ENV PATH="/usr/local/bin:/usr/bin:/bin:${PATH}"

# Install Python development tools
RUN pip install --upgrade pip setuptools wheel

# Install core dependencies in batches to optimize layer caching
# Install PyTorch 2.6+ with CUDA 12.4 (compatible with driver 575.57, required for transformers security)
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir \
    transformers>=4.20.0 \
    pandas>=1.3.0 \
    numpy>=1.20.0 \
    scikit-learn>=0.24.0 \
    tqdm>=4.60.0

RUN pip install --no-cache-dir \
    matplotlib>=3.5.0 \
    seaborn>=0.11.0 \
    hydra-core>=1.3.2 \
    optuna>=3.4.0

RUN pip install --no-cache-dir \
    jupyter \
    black \
    flake8 \
    isort \
    pytest \
    ruff \
    nlpaug \
    textattack

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/workspaces"

# Verify PyTorch installation (GPU test will run at container start)
RUN python3 -c "import torch; print(f'PyTorch {torch.__version__} installed with CUDA {torch.version.cuda}')"

# Set the default shell
SHELL ["/bin/bash", "-c"]
