.PHONY: help install test lint format clean train evaluate docker-build docker-up docker-down stage1 stage2 hpo-report doctor export-best

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
	@echo "  stage1        - Run Stage-1 HPO (pass ARGS for CLI flags)"
	@echo "  stage2        - Run Stage-2 HPO (pass ARGS for CLI flags)"
	@echo "  hpo-report    - Summarise an Optuna study"
	@echo "  doctor        - Run environment diagnostics"
	@echo "  export-best   - Export best configuration snapshot"
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

# HPO
stage1:
	poetry run python -m dataaug_multi_both.cli.hpo stage1 $(ARGS)

stage2:
	poetry run python -m dataaug_multi_both.cli.hpo stage2 $(ARGS)

hpo-report:
	poetry run python -m dataaug_multi_both.cli.hpo report $(ARGS)

doctor:
	poetry run python scripts/doctor.py $(ARGS)

export-best:
	poetry run python scripts/export_best.py $(ARGS)

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
