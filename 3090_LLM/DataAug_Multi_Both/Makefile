.PHONY: help install test lint format clean train evaluate hpo hpo-test hpo-production hpo-info hpo-results docker-build docker-up docker-down mlflow-ui

# Default target
help:
	@echo "==============================================="
	@echo "DataAug Multi Both - Mental Health NLP"
	@echo "==============================================="
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make install      - Install dependencies"
	@echo "  make hpo-test     - Test HPO (3 trials)"
	@echo "  make hpo          - Run HPO (50 trials)"
	@echo "  make mlflow-ui    - View results"
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "  install          - Install dependencies"
	@echo "  install-dev      - Install with dev dependencies"
	@echo ""
	@echo "✨ Code Quality:"
	@echo "  lint             - Run linters (ruff, mypy)"
	@echo "  format           - Format code with black and ruff"
	@echo "  check            - Run all code quality checks"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test             - Run all tests"
	@echo "  test-unit        - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-coverage    - Run tests with coverage report"
	@echo ""
	@echo "🎯 Hyperparameter Optimization:"
	@echo "  hpo-test         - Quick test (3 trials, ~15 min)"
	@echo "  hpo              - Default run (50 trials, ~4-8 hours)"
	@echo "  hpo-production   - Production run (500 trials, ~40-80 hours)"
	@echo "  hpo-info         - Show search space summary"
	@echo "  hpo-results      - View current HPO results"
	@echo ""
	@echo "📊 Evaluation & Results:"
	@echo "  evaluate         - Evaluate study (use ARGS)"
	@echo "  hpo-results      - View HPO study statistics"
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
test:
	poetry run pytest -v

test-unit:
	poetry run pytest -v -m unit

test-integration:
	poetry run pytest -v -m integration

test-coverage:
	poetry run pytest --cov=src/dataaug_multi_both --cov-report=html --cov-report=term

test-hpo:
	poetry run python test_hpo_pipeline.py

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
	@echo "Usage: make train requires subcommand. Available subcommands:"
	@echo "  - train: Train a single model (requires --trial-id)"
	@echo "  - hpo: Run hyperparameter optimization (use 'make hpo' instead)"
	@echo "  - resume: Resume training from checkpoint (requires --trial-dir)"
	@echo "  - status: Show status of experiments"
	@echo ""
	@echo "Example: make train ARGS='train --trial-id trial_001 --model-name mental-bert'"
	@echo "Or use: poetry run python -m dataaug_multi_both.cli.train [SUBCOMMAND] [OPTIONS]"
	@test -n "$(ARGS)" && poetry run python -m dataaug_multi_both.cli.train $(ARGS) || true

# Evaluation
evaluate:
	@echo "Evaluating study. Usage:"
	@echo "  make evaluate ARGS='--study-db experiments/hpo_production.db --study-name mental_health_hpo_production'"
	@test -n "$(ARGS)" && python -m dataaug_multi_both.cli.evaluate_study $(ARGS) || python -m dataaug_multi_both.cli.evaluate_study --help

# View HPO results
hpo-results:
	@echo "==============================================="
	@echo "HPO Study Results"
	@echo "==============================================="
	@if [ -f "experiments/hpo_production.db" ]; then \
		python -c "import optuna; study = optuna.load_study(study_name='mental_health_hpo_production', storage='sqlite:///experiments/hpo_production.db'); print(f'\n📊 Study: {study.study_name}'); print(f'Total trials: {len(study.trials)}'); print(f'COMPLETE: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}'); print(f'PRUNED: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}'); print(f'FAIL: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}'); complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]; print(f'\n🏆 Best trial:') if complete else None; print(f'  Value: {study.best_value:.4f}') if complete else None; print(f'  Backbone: {study.best_params.get(\"backbone\", \"N/A\")}') if complete else None; print(f'  Trial: {study.best_trial.number}') if complete else None"; \
	elif [ -f "experiments/hpo_default.db" ]; then \
		python -c "import optuna; study = optuna.load_study(study_name='hpo_default', storage='sqlite:///experiments/hpo_default.db'); print(f'\n📊 Study: {study.study_name}'); print(f'Total trials: {len(study.trials)}'); print(f'COMPLETE: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}'); print(f'PRUNED: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}'); print(f'FAIL: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}'); complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]; print(f'\n🏆 Best trial:') if complete else None; print(f'  Value: {study.best_value:.4f}') if complete else None; print(f'  Backbone: {study.best_params.get(\"backbone\", \"N/A\")}') if complete else None; print(f'  Trial: {study.best_trial.number}') if complete else None"; \
	else \
		echo "\n⚠️  No study database found. Run HPO first."; \
	fi
	@echo ""
	@echo "==============================================="

# Hyperparameter Optimization

# Default HPO: Moderate run for general exploration
hpo:
	@echo "Running HPO with updated search space (30+ models, 97-100 hyperparameters)"
	@echo "Study: hpo_default, Trials: 50, Dataset: irlab-udc/redsm5"
	@echo "Search space includes: BERT, DeBERTa, RoBERTa, ELECTRA, Longformer, BioBERT, Mental Health models"
	CUBLAS_WORKSPACE_CONFIG=:4096:8 poetry run python -m dataaug_multi_both.cli.train hpo \
		--study-name hpo_default \
		--n-trials 50 \
		--experiments-dir experiments \
		--keep-last-n 1 \
		--keep-best-k 3 \
		--study-db experiments/hpo_default.db \
		--mlflow-uri sqlite:///experiments/mlflow_db/mlflow.db \
		--dataset-id irlab-udc/redsm5 \
		$(ARGS)

# Quick HPO test: Fast validation (3 trials)
hpo-test:
	@echo "Running time-limited HPO smoke test (reduced timeouts)"
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUBLAS_WORKSPACE_CONFIG=:4096:8 poetry run python -m dataaug_multi_both.cli.train \
		--hpo \
		--experiment-name hpo_quick_test \
		--study-name hpo_quick_test \
		--experiments-dir experiments \
		--study-db experiments/hpo_quick_test.db \
		--dataset-id irlab-udc/redsm5 \
		--total-timeout 7200 \
		--timeout-a 3600 \
		--timeout-b 1200 \
		$(ARGS)

# Production HPO: Comprehensive search (500+ trials)
hpo-production:
	@echo "Running production HPO with 500 trials for comprehensive search"
	@echo "Retention: Keep last checkpoint + top 5 best models"
	@echo "Search space: 30+ backbone models, PEFT methods, advanced optimizers"
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUBLAS_WORKSPACE_CONFIG=:4096:8 poetry run python -m dataaug_multi_both.cli.train \
		--hpo \
		--experiment-name mental_health_hpo_production \
		--study-name mental_health_hpo_production \
		--experiments-dir experiments \
		--study-db experiments/hpo_production.db \
		--dataset-id irlab-udc/redsm5 \
		--top-k 5 \
		$(ARGS)

# Show HPO search space information
hpo-info:
	@echo "==============================================="
	@echo "HPO Search Space Summary"
	@echo "==============================================="
	@echo ""
	@echo "📊 Total Hyperparameters: ~97-100"
	@echo ""
	@echo "🧠 Backbone Models (15):"
	@echo "  - BERT family (3): bert-base, bert-large, bert-wwm-squad"
	@echo "  - DeBERTa (2): nvidia/quality-classifier, microsoft/deberta-v3-base"
	@echo "  - SpanBERT (2): spanbert-base, spanbert-large"
	@echo "  - XLM-RoBERTa (3): base, large, large-finetuned-conll03"
	@echo "  - ELECTRA (1): electra-base-discriminator"
	@echo "  - Longformer (2): longformer-base-4096, longformer-extra-pos"
	@echo "  - BioBERT (1): biobert-v1.1"
	@echo "  - ClinicalBERT (1): ClinicalBERT"
	@echo ""
	@echo "⚙️ Key Features:"
	@echo "  - PEFT: LoRA, LoRA+, AdaLoRA, Adapters, IA3"
	@echo "  - Optimizers: AdamW (torch), Adafactor, Lion, Adam"
	@echo "  - Schedulers: Linear, Cosine, Cosine Restart, One Cycle"
	@echo "  - Losses: CE, Focal, BCE, Weighted BCE, Adaptive Focal, Hybrid"
	@echo "  - Adversarial: FGM, PGD"
	@echo "  - Adaptation: DAPT, TAPT"
	@echo "  - Augmentation: NLPAug, TextAttack"
	@echo ""
	@echo "📁 Key Files:"
	@echo "  - Search space: src/dataaug_multi_both/hpo/search_space.py"
	@echo "  - Trial executor: src/dataaug_multi_both/hpo/trial_executor.py"
	@echo "  - CLI: src/dataaug_multi_both/cli/train.py"
	@echo ""
	@echo "==============================================="

# Docker
docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

# MLflow
mlflow-ui:
	@echo "Starting MLflow UI at http://localhost:5000"
	poetry run mlflow ui --backend-store-uri sqlite:///experiments/mlflow_db/mlflow.db --port 5000
