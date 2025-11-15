.PHONY: help install dev-install clean clean-optuna train train-bert train-deberta hpo test format lint

# Set Python path
export PYTHONPATH := $(shell pwd):$(PYTHONPATH)

# Default target
help:
	@echo "Available targets:"
	@echo "  install        - Install production dependencies"
	@echo "  dev-install    - Install development dependencies"
	@echo "  clean          - Clean up generated files and directories"
	@echo "  clean-optuna   - Clean Optuna database files"
	@echo "  train          - Train model with default (RoBERTa) configuration"
	@echo "  train-bert     - Train model with BERT encoder"
	@echo "  train-deberta  - Train model with DeBERTa encoder"
	@echo "  hpo            - Run hyperparameter optimization"
	@echo "  test           - Run tests (when implemented)"
	@echo "  format         - Format code with black"
	@echo "  lint           - Lint code with ruff"
	@echo "  check          - Run format and lint checks"

# Installation targets
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

# Cleaning targets
clean:
	@echo "Cleaning up generated files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf outputs/ multirun/ artifacts/ mlruns/ 2>/dev/null || true
	@find . -name "*.db" -type f -delete 2>/dev/null || true
	@echo "Clean complete!"

clean-optuna:
	@echo "Cleaning Optuna database files..."
	@find . -name "*.db" -type f -delete 2>/dev/null || true
	@echo "Optuna databases cleaned!"

# Training targets
train:
	PYTHONPATH=. python src/train.py

train-bert:
	PYTHONPATH=. python src/train.py model=bert_base

train-deberta:
	PYTHONPATH=. python src/train.py model=deberta_base

train-roberta:
	PYTHONPATH=. python src/train.py model=roberta_base

# HPO target
hpo:
	PYTHONPATH=. python src/hpo.py

# Testing targets
test:
	@echo "Tests not yet implemented"
	# pytest tests/ -v

# Code quality targets
format:
	black src/ --line-length 100

lint:
	ruff check src/

check: format lint
	@echo "Format and lint checks complete!"

# MLflow server
mlflow-server:
	mlflow server --host 0.0.0.0 --port 5000

# Quick experiment with custom parameters
experiment:
	@echo "Usage: make experiment ARGS='key=value key2=value2'"
	@echo "Example: make experiment ARGS='training.batch_size=16 training.learning_rate=1e-5'"
	PYTHONPATH=. python src/train.py $(ARGS)

# Validate configuration
validate-config:
	@echo "Validating Hydra configuration..."
	PYTHONPATH=. python src/train.py --help > /dev/null && echo "✓ Training config is valid"
	PYTHONPATH=. python src/hpo.py --help > /dev/null && echo "✓ HPO config is valid"
