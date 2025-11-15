FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VIRTUALENVS_IN_PROJECT=false

# System deps + Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    git make sudo \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create vscode user
RUN groupadd -g 1000 vscode || true && \
    useradd -m -u 1000 -g 1000 -s /bin/bash vscode || true && \
    echo "vscode ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/90-vscode

# Poetry and project deps (cached by metadata)
WORKDIR /tmp/install
COPY pyproject.toml poetry.lock ./
RUN pip install --upgrade pip && pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only=main --no-root --no-interaction --no-ansi

# CUDA-enabled PyTorch (>=2.6 to satisfy transformers security check; matches CUDA 12.1)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    'torch>=2.6,<3'

# Prepare VS Code server dirs
RUN mkdir -p /home/vscode/.vscode-server/{extensions,bin,data/Machine} && \
    chown -R vscode:vscode /home/vscode/.vscode-server

# Reset workdir to the mounted repository path
WORKDIR /workspaces/DataAug_DeBERTa_Criteria

USER vscode
CMD ["sleep", "infinity"]
