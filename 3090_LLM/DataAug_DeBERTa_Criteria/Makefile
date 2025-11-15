.PHONY: help install test lint format clean train hpo hpo-best hpo-results retrain-best docker-build docker-up docker-down mlflow-ui smoke

# Ensure local package imports work without installing as site-package
export PYTHONPATH := src:$(PYTHONPATH)

help:
	@echo "DataAug DeBERTa Criteria"
	@echo ""
	@echo "Targets:"
	@echo "  make install        Install dependencies"
	@echo "  make test           Run full pytest suite"
	@echo "  make smoke          Run pytest -q"
	@echo "  make hpo            Launch Optuna sweep (20 trials)"
	@echo "  make hpo-best       Train using saved best params"
	@echo "  make hpo-results    Summarize Optuna study"
	@echo "  make mlflow-ui      Open local MLflow UI"
	@echo "  make docker-build   Build devcontainer image"
	@echo "  mlflow-ui        - Start MLflow UI (http://localhost:5000)"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  docker-build     - Build Docker image"
	@echo "  docker-up        - Start containers"
	@echo "  docker-down      - Stop containers"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  clean            - Remove caches"
	@echo "  clean-all        - Remove all generated files"
	@echo ""
	@echo "📖 Documentation: README.md"
	@echo "==============================================="

# Installation
install:
	poetry install --without dev,experiment

install-dev:
	poetry install --with dev,experiment

# Testing
PYTEST_ENV=TRANSFORMERS_NO_TORCHVISION_IMPORT=1 DISABLE_TORCHVISION_IMPORT_ERROR=1

test:
	$(PYTEST_ENV) poetry run pytest -v

test-unit:
	$(PYTEST_ENV) poetry run pytest -v -m unit

test-coverage:
	$(PYTEST_ENV) poetry run pytest --cov=src/dataaug_multi_both --cov-report=html --cov-report=term

test-hpo:
	$(PYTEST_ENV) poetry run python test_hpo_pipeline.py

# Code quality
lint:
	@echo "Running ruff..."
	poetry run ruff check .
	@echo "Running mypy..."
	poetry run mypy src

format:
	@echo "Formatting with black..."
	poetry run black .
	@echo "Auto-fixing with ruff..."
	poetry run ruff check --fix .

check: lint test
	@echo "All checks passed!"

# Cleanup
clean:
	@echo "Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage .hypothesis
	@echo "Cache cleaned!"

clean-all: clean
	@echo "Cleaning all generated files..."
	rm -rf mlruns outputs experiments debug_trial
	@echo "All generated files cleaned!"

# Training
train:
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python poetry run python -m src.dataaug_multi_both.cli.train $(ARGS)

# View HPO results
hpo-results:
	@poetry run python - <<-'PY'
	import optuna
	from pathlib import Path
	db_path = Path("experiments/criteria_hpo.db")
	if not db_path.exists():
	    print("⚠️  No study database found. Run `make hpo` first.")
	else:
	    study = optuna.load_study(study_name="criteria_hpo", storage=f"sqlite:///{db_path}")
	    print(f"\n📊 Study: {study.study_name}")
	    print(f"Total trials: {len(study.trials)}")
	    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
	    print(f"Completed: {len(complete)}")
	    if complete:
	        print(f"\n🏆 Best value: {study.best_value:.4f}")
	        print(f"Best params: {study.best_params}")
		PY
	@echo ""

## Hyperparameter Optimization
hpo:
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python poetry run python -m src.dataaug_multi_both.cli.train --hpo \
	  --experiments-dir experiments --experiment-name "criteria_hpo" \
	  --epochs-a 100 --epochs-b 100 $(ARGS)

hpo-best:
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python poetry run python -m src.dataaug_multi_both.cli.train --use-best \
	  --experiments-dir experiments --experiment-name "criteria_hpo" --study-name "criteria_hpo" $(ARGS)

retrain-best:
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python poetry run python -m src.dataaug_multi_both.cli.train --retrain-best --retrain-seeds 3 \
	  --experiments-dir experiments --experiment-name "criteria_best_multiseed" --study-name "criteria_hpo" $(ARGS)

smoke:
	$(PYTEST_ENV) poetry run pytest -q

# Docker
docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

# MLflow
mlflow-ui:  ## Open local MLflow UI for sqlite backend
	mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db --port 5000
