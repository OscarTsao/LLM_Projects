# syntax=docker/dockerfile:1

ARG CUDA_VERSION=12.1.1
ARG UBUNTU_VERSION=22.04
ARG TORCH_VERSION=2.1.2
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    POETRY_VERSION=1.6.1 \
    POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python3 -m pip install --upgrade pip

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
RUN poetry install --with dev --no-root

RUN /app/.venv/bin/pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch==${TORCH_VERSION}

COPY src ./src
COPY tests ./tests
COPY configs ./configs

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/poetry/bin:/workspace/.venv/bin:/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH" \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python3 -m pip install --upgrade pip

COPY --from=builder /opt/poetry /opt/poetry

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/tests ./tests
COPY --from=builder /app/configs ./configs
COPY pyproject.toml poetry.lock* ./

RUN /app/.venv/bin/pip install --no-deps -e .

RUN mkdir -p /app/data /app/outputs /app/mlruns

RUN useradd -ms /bin/bash vscode && chown -R vscode:vscode /app

USER vscode

WORKDIR /workspace

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 5000

CMD ["sleep", "infinity"]
