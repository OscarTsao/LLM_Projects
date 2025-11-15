FROM mcr.microsoft.com/devcontainers/python:3.10

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git make && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /workspaces/DataAug_Multi_Both

# Pre-install poetry and common dependencies
RUN pip install --upgrade pip && pip install poetry && poetry config virtualenvs.create false

# Copy pyproject.toml to workdir for dependency caching
COPY pyproject.toml ./
RUN poetry install --no-interaction --no-ansi --no-root || true

# Default command
CMD ["bash"]
