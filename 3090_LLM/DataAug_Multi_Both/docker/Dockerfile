FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VENV_IN_PROJECT=1
ENV POETRY_CACHE_DIR=/tmp/poetry_cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Set work directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-root --no-interaction --no-ansi \
    && rm -rf $POETRY_CACHE_DIR

# Copy project files
COPY . .

# Install the project in development mode
RUN poetry install --no-root --no-interaction --no-ansi

# Create necessary directories
RUN mkdir -p experiments/logs experiments/mlflow_db

# Expose MLflow port
EXPOSE 5000

# Default command
CMD ["bash"]
