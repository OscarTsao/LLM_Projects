# Use NVIDIA CUDA base image with Ubuntu 22.04 for compatibility with both 20.04 and 22.04 LTS
# CUDA 12.1 base image with driver support, PyTorch nightly with CUDA 13 installed via pip
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get -y install --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    nano \
    htop \
    tree \
    zip \
    unzip \
    sudo \
    wget \
    ca-certificates \
    build-essential \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install Python development tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch nightly with CUDA 13.0 support
RUN pip3 install --pre --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130

# Install core ML dependencies in batches to optimize layer caching
RUN pip3 install --no-cache-dir \
    transformers>=4.20.0 \
    pandas>=1.3.0 \
    numpy>=1.20.0 \
    scikit-learn>=0.24.0 \
    tqdm>=4.60.0

RUN pip3 install --no-cache-dir \
    matplotlib>=3.5.0 \
    seaborn>=0.11.0 \
    hydra-core>=1.3.2 \
    optuna>=3.4.0

# Install development tools
RUN pip3 install --no-cache-dir \
    jupyter \
    black \
    flake8 \
    isort \
    pytest \
    ruff

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/workspaces"

# Create a non-root user for vscode
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Configure git to handle the workspace safely
RUN git config --system --add safe.directory '*'

# Switch to vscode user
USER $USERNAME

# Set the default shell
SHELL ["/bin/bash", "-c"]
