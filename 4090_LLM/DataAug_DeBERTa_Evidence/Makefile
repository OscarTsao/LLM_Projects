.PHONY: help install test lint format clean train hpo evaluate docker-build docker-up docker-down

# Default target
help:
	@echo "Available targets:"
	@echo "  install          - Install dependencies with Poetry"
	@echo "  test             - Run all tests"
	@echo "  test-unit        - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-coverage    - Run tests with coverage report"
	@echo "  lint             - Run linters (ruff, mypy)"
	@echo "  format           - Format code with black and ruff"
	@echo "  clean            - Remove generated files"
	@echo "  clean-hpo        - Remove all HPO artifacts (trials, databases, checkpoints)"
	@echo "  train            - Run training (use ARGS for Hydra overrides)"
	@echo "  hpo              - Run hyperparameter optimization (use ARGS for options)"
	@echo "  hpo-best         - Run HPO with best configuration (epochs=100 for both stages)"
	@echo "  retrain-best     - Retrain best models with multiple seeds"
	@echo "  evaluate         - Run evaluation (use ARGS for options)"
	@echo "  docker-build     - Build Docker image"
	@echo "  docker-up        - Start Docker containers"
	@echo "  docker-down      - Stop Docker containers"
	@echo "  mlflow-ui        - Start MLflow UI"

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

clean-hpo:
	@echo "Cleaning HPO artifacts..."
	rm -rf experiments/trial_*
	rm -f experiments/checkpoint.pt
	rm -f optuna.db optuna.db-shm optuna.db-wal
	rm -f mlflow.db mlflow.db-shm mlflow.db-wal
	rm -rf artifacts/mlflow_buffer
	@echo "HPO history cleaned. Ready for fresh run."

# Training (epochs configured in configs/train.yaml: num_epochs=100)
train:
	poetry run python -m src.dataaug_multi_both.cli.train $(ARGS)

# Hyperparameter optimization (epochs configured in configs/hpo.yaml: stage_a=100, stage_b=100)
hpo:
	@poetry run python scripts/init_optuna_db.py optuna.db
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=src poetry run python src/dataaug_multi_both/console.py tune $(ARGS)

# HPO configuration variables
TRIALS_A ?= 380
EPOCHS_A ?= 100
TRIALS_B ?= 120
EPOCHS_B ?= 100
K_TOP    ?= 5
MODEL    ?= microsoft/deberta-v3-base
SEED     ?= 42
TIMEOUT  ?= 604800

.PHONY: hpo-best
hpo-best:
	@poetry run python scripts/init_optuna_db.py optuna.db
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=src poetry run python src/dataaug_multi_both/console.py tune \
		--model $(MODEL) \
		--trials-a $(TRIALS_A) \
		--epochs-a $(EPOCHS_A) \
		--trials-b $(TRIALS_B) \
		--epochs-b $(EPOCHS_B) \
		--k-top $(K_TOP) \
		--seed $(SEED) \
		--timeout $(TIMEOUT)

# Retrain configuration
RETRAIN_SEEDS ?= 3

.PHONY: retrain-best
retrain-best:
	PYTHONPATH=src poetry run python src/dataaug_multi_both/console.py retrain-best --seeds $(RETRAIN_SEEDS)

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
