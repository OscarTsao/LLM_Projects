.PHONY: help install test lint format clean train evaluate docker-build docker-up docker-down

# Default target
help:
	@echo "Available targets:"
	@echo "  install       - Install dependencies with Poetry"
	@echo "  test          - Run all tests"
	@echo "  test-unit     - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  lint          - Run linters (ruff, mypy)"
	@echo "  format        - Format code with black"
	@echo "  clean         - Remove generated files"
	@echo "  train         - Run training (use ARGS for Hydra overrides)"
	@echo "  evaluate      - Run evaluation (use ARGS for options)"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-up     - Start Docker containers"
	@echo "  docker-down   - Stop Docker containers"

# Installation
install:
	poetry install

# Testing
test:
	poetry run pytest -v

test-unit:
	poetry run pytest -v -m unit

test-integration:
	poetry run pytest -v -m integration

test-coverage:
	poetry run pytest --cov=src/dataaug_multi_both --cov-report=html --cov-report=term

# Code quality
lint:
	poetry run ruff check .
	poetry run mypy src

format:
	poetry run black .
	poetry run ruff check --fix .

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage

# Training
train:
	poetry run python -m src.dataaug_multi_both.cli.train $(ARGS)

# Evaluation
evaluate:
	poetry run python -m src.dataaug_multi_both.cli.evaluate $(ARGS)

# Docker
docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

# MLflow
mlflow-ui:
	poetry run mlflow ui --backend-store-uri sqlite:///experiments/mlflow_db/mlflow.db --port 5000

