.PHONY: help install install-dev test clean lint format check-types run-demo
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package and dependencies
	pip install -e .

install-dev: ## Install the package with development dependencies
	pip install -e ".[dev]"

install-gpu: ## Install the package with GPU support
	pip install -e ".[gpu]"

install-all: ## Install the package with all optional dependencies
	pip install -e ".[dev,gpu,wandb,optuna]"

test: ## Run tests
	python -m pytest tests/ -v

test-cov: ## Run tests with coverage
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

clean: ## Clean up temporary files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage.*" -delete

lint: ## Run flake8 linting
	python -m flake8 src/ scripts/ tests/

format: ## Format code with black
	python -m black src/ scripts/ tests/

format-check: ## Check if code needs formatting
	python -m black --check src/ scripts/ tests/

check-types: ## Run type checking (if mypy is available)
	@which mypy > /dev/null && mypy src/ || echo "mypy not installed, skipping type check"

check: lint format-check ## Run all checks

run-demo: ## Run the demo example
	python examples/demo.py

run-main: ## Run main RAG pipeline
	python scripts/main.py --mode evaluate --num_posts 100

run-comparison: ## Run comparison analysis
	python scripts/comparison_main.py --num_negative 100 --num_positive 100

run-evaluation: ## Run evaluation system
	python scripts/evaluation_main.py --num_posts 100

build-index: ## Build and save FAISS index
	python scripts/main.py --mode build_index --save_index

setup-data: ## Setup data directories
	mkdir -p Data/indices logs results

docker-build: ## Build Docker image
	docker build -t psy-rag .

docker-run: ## Run Docker container
	docker run -it --rm -v $(PWD)/Data:/app/Data -v $(PWD)/results:/app/results psy-rag

# Development helpers
dev-setup: install-dev setup-data ## Complete development setup
	@echo "Development environment ready!"

prod-setup: install setup-data ## Production setup
	@echo "Production environment ready!"

# Data processing
process-data: ## Process and prepare data files
	@echo "Data processing commands would go here"

# Model management
download-models: ## Download required models
	@echo "Model download commands would go here"

# Documentation
docs: ## Generate documentation
	@echo "Documentation generation commands would go here"

# Cleanup and maintenance
deep-clean: clean ## Deep clean including data and model caches
	rm -rf Data/indices/*
	rm -rf logs/*
	rm -rf results/*
	rm -rf .cache/
	@echo "Deep clean completed"