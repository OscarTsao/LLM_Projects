# syntax=docker/dockerfile:1

# Canonical build steps are shared with .devcontainer/Dockerfile.
# This file mirrors that configuration for users building from the project root.

FROM python:3.10-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    POETRY_VERSION=1.6.1 \
    POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
RUN poetry install --with dev --no-root

COPY src ./src
COPY tests ./tests
COPY configs ./configs

FROM python:3.10-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/poetry/bin:/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/poetry /opt/poetry

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/tests ./tests
COPY --from=builder /app/configs ./configs
COPY pyproject.toml poetry.lock* ./

RUN pip install --upgrade pip && pip install -e .

RUN mkdir -p /app/data /app/outputs /app/mlruns

RUN useradd -ms /bin/bash vscode && chown -R vscode:vscode /app

USER vscode

WORKDIR /workspace

EXPOSE 5000

CMD ["sleep", "infinity"]
