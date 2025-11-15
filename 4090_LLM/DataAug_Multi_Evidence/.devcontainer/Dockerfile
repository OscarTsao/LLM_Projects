FROM mcr.microsoft.com/devcontainers/python:3.10

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git make && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /workspaces/app

# Cache-friendly dependency install
COPY pyproject.toml /tmp/pyproject.toml
RUN pip install --upgrade pip && pip install poetry && poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi || true

# Default command
CMD ["bash"]
