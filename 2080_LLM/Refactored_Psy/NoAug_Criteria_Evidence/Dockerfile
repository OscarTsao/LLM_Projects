# Multi-stage Dockerfile for PSY Agents NO-AUG

# Stage 1: Builder
FROM python:3.10-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.6.1
ENV POETRY_HOME=/opt/poetry
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=1
ENV POETRY_VIRTUALENVS_CREATE=1
RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="$POETRY_HOME/bin:$PATH"

# Copy project files
WORKDIR /app
COPY pyproject.toml poetry.lock ./
COPY src ./src
COPY tests ./tests
COPY configs ./configs

# Install dependencies
RUN poetry install --no-root --only main

# Stage 2: Runtime
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

WORKDIR /app

# Activate virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Install the package
RUN /app/.venv/bin/pip install -e .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/mlruns

# Expose MLflow port
EXPOSE 5000

# Default command
CMD ["python", "-m", "psy_agents_noaug.cli", "--help"]
