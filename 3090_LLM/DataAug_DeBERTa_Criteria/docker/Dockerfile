FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VENV_IN_PROJECT=1
ENV POETRY_CACHE_DIR=/tmp/poetry_cache

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Ensure python/pip are available under standard names
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install Poetry
RUN pip install --no-cache-dir poetry

# Set work directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml ./

# Configure Poetry and install dependencies (CPU/GPU-agnostic)
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-root --no-interaction --no-ansi \
    && rm -rf $POETRY_CACHE_DIR

# Copy project files
COPY . .

# Install the project in development mode
RUN poetry install --no-root --no-interaction --no-ansi

# Upgrade to CUDA-enabled PyTorch wheels (matches CUDA 12.1 base image)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# Create necessary directories
RUN mkdir -p experiments/logs experiments/mlflow_db

# Expose MLflow port
EXPOSE 5000

# Default command
CMD ["bash"]
