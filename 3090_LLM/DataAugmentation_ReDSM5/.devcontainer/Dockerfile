FROM nvcr.io/nvidia/pytorch:24.10-py3

# Install additional system dependencies (PyTorch and CUDA already included)
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    nano \
    htop \
    tree \
    zip \
    unzip \
    sudo \
    wget \
    curl \
    ca-certificates \
    postgresql-client \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda/mamba)
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/bin:$PATH

# Install Python development tools
RUN pip install --upgrade pip setuptools wheel

# Install core dependencies in batches to optimize layer caching
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

# Install MLflow, DVC, and monitoring tools
RUN pip install --no-cache-dir \
    mlflow>=2.9.0 \
    dvc>=3.0.0 \
    dvc-s3 \
    psycopg2-binary \
    tensorboard \
    optuna-dashboard

RUN pip install --no-cache-dir \
    jupyter \
    black \
    flake8 \
    isort \
    pytest \
    ruff \
    mypy

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
