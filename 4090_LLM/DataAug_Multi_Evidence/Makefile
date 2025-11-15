.PHONY: help install lock test lint format type-check clean clean-logs clean-experiments clean-all
.PHONY: train-single train-hpo train-resume train-dry-run
.PHONY: evaluate-trial evaluate-study analyze-errors export-hpo-results
.PHONY: mlflow-ui data-config data-check
.PHONY: docker-build docker-run docker-shell
.PHONY: dev-setup hf-login
.PHONY: hpo-best retrain-best hpo-two-stage hpo-two-stage-ui hpo-two-stage-retrain

# Default shell
SHELL := /bin/bash

# Project paths
PROJECT_ROOT := $(shell pwd)
SRC_DIR := $(PROJECT_ROOT)/src
TESTS_DIR := $(PROJECT_ROOT)/tests
CONFIGS_DIR := $(PROJECT_ROOT)/configs
EXPERIMENTS_DIR := $(PROJECT_ROOT)/experiments
SPECS_DIR := $(PROJECT_ROOT)/specs/002-storage-optimized-training

# Configuration defaults
STUDY_DB ?= $(EXPERIMENTS_DIR)/hpo_study.db
MLFLOW_URI ?= file://$(EXPERIMENTS_DIR)/mlflow_db
MLFLOW_PORT ?= 5000
HPO_CONFIG ?= configs/hpo_study.yaml
SINGLE_CONFIG ?= configs/single_trial.yaml
HPO_TWO_STAGE_OUTPUT ?= $(EXPERIMENTS_DIR)/hpo_runs
HPO_TWO_STAGE_DATASET ?= $(CONFIGS_DIR)/data/dataset.yaml
HPO_TWO_STAGE_MODEL ?= microsoft/deberta-v3-base
HPO_TWO_STAGE_TRIALS_A ?= 380
HPO_TWO_STAGE_TRIALS_B ?= 120
HPO_TWO_STAGE_EPOCHS_A ?= 100
HPO_TWO_STAGE_EPOCHS_B ?= 100
HPO_TWO_STAGE_TOP_K ?= 5
HPO_TWO_STAGE_TIMEOUT ?= 604800
HPO_TWO_STAGE_SEED ?= 42
HPO_TWO_STAGE_MLFLOW_URI := sqlite:///$(HPO_TWO_STAGE_OUTPUT)/mlflow.db
HPO_TWO_STAGE_BEST ?= $(HPO_TWO_STAGE_OUTPUT)/artifacts/hpo/aggregate/best_trial.json

# Docker configuration
DOCKER_IMAGE ?= mental-health-hpo
DOCKER_TAG ?= latest
DOCKER_FULL_NAME := $(DOCKER_IMAGE):$(DOCKER_TAG)

# Color output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Execution runner
# By default, run tools directly to avoid sandboxed Poetry env creation.
# Set USE_POETRY=1 to force running inside Poetry's virtualenv.
USE_POETRY ?= 0
ifeq ($(USE_POETRY),1)
RUN := poetry run
PY := poetry run python
else
RUN :=
PY := python
endif

##@ General

help: ## Display this help message
	@echo "$(BLUE)DataAug_Multi_Both - Storage-Optimized Training & HPO Pipeline$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-25s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development Setup

install: ## Install all dependencies using Poetry
	@echo "$(BLUE)Installing dependencies...$(NC)"
	poetry install
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

lock: ## Update Poetry lock file
	@echo "$(BLUE)Updating Poetry lock file...$(NC)"
	poetry lock
	@echo "$(GREEN)✓ Lock file updated$(NC)"

dev-setup: install ## Full development setup (install + verify)
	@echo "$(BLUE)Running development setup...$(NC)"
	@$(MAKE) data-check
	@echo "$(GREEN)✓ Development environment ready$(NC)"

hf-login: ## Login to Hugging Face CLI (required for datasets/models)
	@echo "$(YELLOW)Logging into Hugging Face...$(NC)"
	huggingface-cli login
	@echo "$(GREEN)✓ Hugging Face authentication complete$(NC)"

##@ Code Quality

test: ## Run all tests with pytest
	@echo "$(BLUE)Running tests...$(NC)"
	$(RUN) pytest -v $(TESTS_DIR)

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(RUN) pytest -v $(TESTS_DIR)/unit

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(RUN) pytest -v $(TESTS_DIR)/integration

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@if $(RUN) pytest --help | grep -q -- "--cov"; then \
		$(RUN) pytest --cov=$(SRC_DIR) --cov-report=html --cov-report=term $(TESTS_DIR); \
		echo "$(GREEN)✓ Coverage report generated at htmlcov/index.html$(NC)"; \
	else \
		echo "$(YELLOW)pytest-cov not installed; running tests without coverage$(NC)"; \
		$(RUN) pytest -v $(TESTS_DIR); \
	fi

lint: ## Run linting with ruff
	@echo "$(BLUE)Running linter...$(NC)"
	$(RUN) ruff check . --no-cache

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	$(RUN) black .
	@echo "$(GREEN)✓ Code formatted$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checker...$(NC)"
	$(RUN) mypy $(SRC_DIR)

check-all: lint type-check test ## Run all checks (lint + type-check + test)
	@echo "$(GREEN)✓ All checks passed$(NC)"

##@ Training & HPO

train-single: ## Run single trial training (use SINGLE_CONFIG=path/to/config.yaml)
	@echo "$(BLUE)Running single trial training...$(NC)"
	@if [ ! -f "$(SINGLE_CONFIG)" ]; then \
		echo "$(RED)✗ Config file not found: $(SINGLE_CONFIG)$(NC)"; \
		echo "$(YELLOW)Set SINGLE_CONFIG=path/to/your/config.yaml$(NC)"; \
		exit 1; \
	fi
	$(PY) -m dataaug_multi_both.cli.train \
		--config $(SINGLE_CONFIG) \
		--mode single \
		--output-dir $(EXPERIMENTS_DIR)/trial_debug

train-hpo: ## Run full two-stage HPO (500 trials) with NO augmentations
	@echo "$(BLUE)Starting two-stage HPO (no augmentations)...$(NC)"
	@$(MAKE) hpo-best \
		MODEL=$(HPO_TWO_STAGE_MODEL) \
		TRIALS_A=$(HPO_TWO_STAGE_TRIALS_A) \
		EPOCHS_A=$(HPO_TWO_STAGE_EPOCHS_A) \
		TRIALS_B=$(HPO_TWO_STAGE_TRIALS_B) \
		EPOCHS_B=$(HPO_TWO_STAGE_EPOCHS_B) \
		K_TOP=$(HPO_TWO_STAGE_TOP_K) \
		TIMEOUT=$(HPO_TWO_STAGE_TIMEOUT) \
		SEED=$(HPO_TWO_STAGE_SEED) \
		OUTPUT_ROOT=$(HPO_TWO_STAGE_OUTPUT) \
		DATASET_CONFIG=$(HPO_TWO_STAGE_DATASET) \
		USE_REAL_DATA=1 \
		NO_AUG=1 \
		USE_POETRY=$(USE_POETRY)

train-resume: ## Resume interrupted HPO study
	@echo "$(BLUE)Resuming HPO study...$(NC)"
	@if [ ! -f "$(STUDY_DB)" ]; then \
		echo "$(RED)✗ Study database not found: $(STUDY_DB)$(NC)"; \
		exit 1; \
	fi
	$(PY) -m dataaug_multi_both.cli.train \
		--config $(HPO_CONFIG) \
		--mode hpo \
		--study-db $(STUDY_DB) \
		--mlflow-uri $(MLFLOW_URI) \
		--resume

train-dry-run: ## Validate configuration without training
	@echo "$(BLUE)Running dry-run validation...$(NC)"
	$(PY) -m dataaug_multi_both.cli.train \
		--config $(HPO_CONFIG) \
		--mode hpo \
		--dry-run

##@ Evaluation & Analysis

evaluate-trial: ## Evaluate specific trial (use TRIAL_ID=<uuid>)
	@if [ -z "$(TRIAL_ID)" ]; then \
		echo "$(RED)✗ TRIAL_ID not set$(NC)"; \
		echo "$(YELLOW)Usage: make evaluate-trial TRIAL_ID=<trial_uuid>$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Evaluating trial $(TRIAL_ID)...$(NC)"
	$(PY) -m dataaug_multi_both.cli.evaluate_trial --trial-id $(TRIAL_ID) --experiments-dir $(EXPERIMENTS_DIR)

evaluate-checkpoint: ## Evaluate checkpoint and export predictions (use TRIAL_ID=<id> CHECKPOINT=<file>)
	@if [ -z "$(TRIAL_ID)" ] || [ -z "$(CHECKPOINT)" ]; then \
		echo "$(RED)✗ TRIAL_ID or CHECKPOINT not set$(NC)"; \
		echo "$(YELLOW)Usage: make evaluate-checkpoint TRIAL_ID=<id> CHECKPOINT=<checkpoint_file.pt>$(NC)"; \
		exit 1; \
	fi
	@TRIAL_DIR="$(TRIAL_ID)"; \
	if [ ! -d "$$TRIAL_DIR" ] && [ -d "trial_$(TRIAL_ID)" ]; then \
		TRIAL_DIR="trial_$(TRIAL_ID)"; \
	fi; \
	CHECKPOINT_PATH=$${CHECKPOINT_PATH:-$(CHECKPOINT)}; \
	case "$$CHECKPOINT_PATH" in \
		/*) ;; \
		*) CHECKPOINT_PATH="$$TRIAL_DIR/checkpoints/$$CHECKPOINT_PATH";; \
	esac; \
	OUTPUT_DIR=$${OUTPUT_DIR:-outputs/prediction/$${TRIAL_DIR##*/}}; \
	MODEL_NAME=$${MODEL_NAME:-google-bert/bert-base-uncased}; \
	BATCH_SIZE=$${BATCH_SIZE:-16}; \
	echo "$(BLUE)Evaluating checkpoint $$CHECKPOINT_PATH → $$OUTPUT_DIR$(NC)"; \
	mkdir -p "$$OUTPUT_DIR"; \
	$(PY) -m dataaug_multi_both.cli.evaluate_checkpoint \
		--checkpoint "$$CHECKPOINT_PATH" \
		--model-name "$$MODEL_NAME" \
		--batch-size "$$BATCH_SIZE" \
		--output-dir "$$OUTPUT_DIR" \
		$(if $(strip $(ALLOW_DOWNLOAD)),--allow-download,)

evaluate-study: ## Evaluate full study (use STUDY_ID=<uuid> BEST_TRIAL_ID=<uuid>)
	@if [ -z "$(STUDY_ID)" ] || [ -z "$(BEST_TRIAL_ID)" ]; then \
		echo "$(RED)✗ STUDY_ID or BEST_TRIAL_ID not set$(NC)"; \
		echo "$(YELLOW)Usage: make evaluate-study STUDY_ID=<uuid> BEST_TRIAL_ID=<uuid>$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Evaluating study $(STUDY_ID)...$(NC)"
	$(PY) -m dataaug_multi_both.cli.evaluate_study \
		--study-id $(STUDY_ID) \
		--best-trial-id $(BEST_TRIAL_ID) \
		--experiments-dir $(EXPERIMENTS_DIR)

analyze-errors: ## Run error analysis on trial (use TRIAL_ID=<uuid> SPLIT=test)
	@if [ -z "$(TRIAL_ID)" ]; then \
		echo "$(RED)✗ TRIAL_ID not set$(NC)"; \
		echo "$(YELLOW)Usage: make analyze-errors TRIAL_ID=<uuid> [SPLIT=test]$(NC)"; \
		exit 1; \
	fi
	@SPLIT_NAME=$${SPLIT:-test}; \
	echo "$(BLUE)Running error analysis for trial $(TRIAL_ID) on $$SPLIT_NAME split...$(NC)"; \
	$(PY) -m dataaug_multi_both.analysis.error_analysis \
		--trial-id $(TRIAL_ID) \
		--split $$SPLIT_NAME \
		--output $(EXPERIMENTS_DIR)/errors_$(TRIAL_ID).csv

export-hpo-results: ## Export HPO results to CSV (use STUDY_NAME=<name>)
	@STUDY_NAME=$${STUDY_NAME:-mental_health_hpo_2025_10}; \
	echo "$(BLUE)Exporting HPO results for study: $$STUDY_NAME...$(NC)"; \
	$(PY) -c "import optuna; study = optuna.load_study(study_name='$$STUDY_NAME', storage='sqlite:///$(STUDY_DB)'); df = study.trials_dataframe(); output_file = '$(EXPERIMENTS_DIR)/hpo_results_$$STUDY_NAME.csv'; df.to_csv(output_file, index=False); print('Results exported to ' + output_file)" && echo "$(GREEN)✓ Export complete$(NC)" || (echo "$(RED)✗ Export failed$(NC)" && exit 1)

##@ MLflow & Monitoring

mlflow-ui: ## Start MLflow UI server (use MLFLOW_PORT=5000)
	@echo "$(BLUE)Starting MLflow UI on port $(MLFLOW_PORT)...$(NC)"
	@echo "$(YELLOW)Access UI at: http://localhost:$(MLFLOW_PORT)$(NC)"
	$(PY) -m mlflow ui --backend-store-uri $(MLFLOW_URI) --port $(MLFLOW_PORT)

hpo-two-stage-ui: ## Serve MLflow UI for two-stage HPO outputs
	@echo "$(BLUE)Starting MLflow UI for two-stage HPO on port $(MLFLOW_PORT)...$(NC)"
	@echo "$(YELLOW)Access UI at: http://localhost:$(MLFLOW_PORT)$(NC)"
	$(PY) -m mlflow ui --backend-store-uri $(HPO_TWO_STAGE_MLFLOW_URI) --port $(MLFLOW_PORT)

mlflow-experiments: ## List all MLflow experiments
	@echo "$(BLUE)Listing MLflow experiments...$(NC)"
	@$(PY) -c "import mlflow; mlflow.set_tracking_uri('$(MLFLOW_URI)'); experiments = mlflow.search_experiments(); [print(f'{exp.experiment_id}: {exp.name}') for exp in experiments]" || echo "$(YELLOW)No experiments found or MLflow not initialized$(NC)"

##@ Data Management

data-config: ## Validate dataset configuration
	@echo "$(BLUE)Validating dataset configuration...$(NC)"
	@if [ -f "$(CONFIGS_DIR)/data/dataset.yaml" ]; then \
		cat $(CONFIGS_DIR)/data/dataset.yaml; \
		echo "$(GREEN)✓ Dataset config found$(NC)"; \
	else \
		echo "$(RED)✗ Dataset config not found at $(CONFIGS_DIR)/data/dataset.yaml$(NC)"; \
		exit 1; \
	fi

data-check: ## Verify Hugging Face dataset accessibility
	@echo "$(BLUE)Checking Hugging Face dataset access...$(NC)"
	@$(PY) -c "from datasets import load_dataset; ds = load_dataset('irlab-udc/redsm5', split='train', streaming=True); first = next(iter(ds)); print('Dataset accessible')" && echo "$(GREEN)✓ Dataset accessible$(NC)" || (echo "$(RED)✗ Cannot access dataset$(NC)" && echo "$(YELLOW)Run: make hf-login$(NC)" && exit 1)

##@ Docker

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image $(DOCKER_FULL_NAME)...$(NC)"
	@if [ ! -f "docker/Dockerfile" ]; then \
		echo "$(RED)✗ Dockerfile not found at docker/Dockerfile$(NC)"; \
		exit 1; \
	fi
	docker build -t $(DOCKER_FULL_NAME) -f docker/Dockerfile .
	@echo "$(GREEN)✓ Docker image built: $(DOCKER_FULL_NAME)$(NC)"

docker-run: ## Run Docker container with GPU support
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run --gpus all \
		-v $(PROJECT_ROOT)/Data:/workspace/Data:ro \
		-v $(PROJECT_ROOT)/experiments:/workspace/experiments \
		-v $(HOME)/.cache/huggingface:/workspace/.cache/huggingface \
		-it $(DOCKER_FULL_NAME) bash

docker-run-cpu: ## Run Docker container without GPU
	@echo "$(BLUE)Running Docker container (CPU only)...$(NC)"
	docker run \
		-v $(PROJECT_ROOT)/Data:/workspace/Data:ro \
		-v $(PROJECT_ROOT)/experiments:/workspace/experiments \
		-v $(HOME)/.cache/huggingface:/workspace/.cache/huggingface \
		-it $(DOCKER_FULL_NAME) bash

docker-shell: docker-run ## Alias for docker-run

##@ Cleanup

clean-logs: ## Remove log files
	@echo "$(BLUE)Cleaning log files...$(NC)"
	find $(EXPERIMENTS_DIR) -type f -name "*.log" -delete
	find $(EXPERIMENTS_DIR) -type f -name "*.jsonl" -delete
	@echo "$(GREEN)✓ Log files removed$(NC)"

clean-experiments: ## Remove old experiment directories (interactive)
	@echo "$(YELLOW)⚠ This will list experiment directories for manual cleanup$(NC)"
	@echo "$(BLUE)Experiment directories:$(NC)"
	@ls -lh $(EXPERIMENTS_DIR)/trial_* 2>/dev/null || echo "No trial directories found"
	@echo ""
	@echo "$(YELLOW)To delete a specific trial: rm -rf $(EXPERIMENTS_DIR)/trial_<uuid>$(NC)"

clean-checkpoints: ## Remove all checkpoints except best and latest
	@echo "$(YELLOW)⚠ This will remove all checkpoints except best and latest$(NC)"
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		find $(EXPERIMENTS_DIR) -type d -name "checkpoints" -exec sh -c '\
			cd "{}" && ls -t | tail -n +4 | xargs rm -rf \
		' \;; \
		echo "$(GREEN)✓ Old checkpoints removed$(NC)"; \
	else \
		echo "$(YELLOW)Cancelled$(NC)"; \
	fi

clean-cache: ## Clear Python and Hugging Face caches
	@echo "$(BLUE)Cleaning caches...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Python cache cleared$(NC)"

clean-all: clean-logs clean-cache ## Full cleanup (logs + cache, prompts before experiments)
	@echo "$(RED)⚠ WARNING: This will remove all logs and caches$(NC)"
	@echo "$(YELLOW)Experiments and checkpoints will NOT be deleted automatically$(NC)"
	@echo "$(YELLOW)Use 'make clean-experiments' or 'make clean-checkpoints' separately$(NC)"

##@ Documentation

docs-spec: ## View project specification
	@if [ -f "$(SPECS_DIR)/spec.md" ]; then \
		cat $(SPECS_DIR)/spec.md; \
	else \
		echo "$(RED)✗ Spec not found$(NC)"; \
	fi

docs-quickstart: ## View quickstart guide
	@if [ -f "$(SPECS_DIR)/quickstart.md" ]; then \
		cat $(SPECS_DIR)/quickstart.md; \
	else \
		echo "$(RED)✗ Quickstart not found$(NC)"; \
	fi

docs-tasks: ## View implementation tasks
	@if [ -f "$(SPECS_DIR)/tasks.md" ]; then \
		cat $(SPECS_DIR)/tasks.md; \
	else \
		echo "$(RED)✗ Tasks not found$(NC)"; \
	fi

##@ Quick Workflows

quick-test: lint test-unit ## Quick validation (lint + unit tests)
	@echo "$(GREEN)✓ Quick validation complete$(NC)"

quick-hpo: data-check train-hpo ## Quick HPO launch (check data + start HPO)

quick-analysis: export-hpo-results ## Quick results analysis (export to CSV)
	@echo "$(GREEN)✓ Results exported$(NC)"
	@echo "$(YELLOW)View with: cat $(EXPERIMENTS_DIR)/hpo_results_*.csv | head$(NC)"

hpo-two-stage: ## Run full two-stage HPO (Stage A + Stage B, total 500 trials)
	@echo "$(BLUE)Launching two-stage HPO run...$(NC)"
	@$(MAKE) hpo-best \
		MODEL=$(HPO_TWO_STAGE_MODEL) \
		TRIALS_A=$(HPO_TWO_STAGE_TRIALS_A) \
		EPOCHS_A=$(HPO_TWO_STAGE_EPOCHS_A) \
		TRIALS_B=$(HPO_TWO_STAGE_TRIALS_B) \
		EPOCHS_B=$(HPO_TWO_STAGE_EPOCHS_B) \
		K_TOP=$(HPO_TWO_STAGE_TOP_K) \
		TIMEOUT=$(HPO_TWO_STAGE_TIMEOUT) \
		SEED=$(HPO_TWO_STAGE_SEED) \
		OUTPUT_ROOT=$(HPO_TWO_STAGE_OUTPUT) \
		DATASET_CONFIG=$(HPO_TWO_STAGE_DATASET) \
		USE_REAL_DATA=1 \
		USE_POETRY=$(USE_POETRY)

hpo-best: ## Run two-stage HPO (Stage A + Stage B)
	@$(PY) -m dataaug_multi_both.hpo.driver \
		--trials-a $(TRIALS_A) \
		--epochs-a $(EPOCHS_A) \
		--trials-b $(TRIALS_B) \
		--epochs-b $(EPOCHS_B) \
		--k-top $(K_TOP) \
		--timeout $(TIMEOUT) \
		--model $(MODEL) \
		--seed $(SEED) \
		--output-root $(OUTPUT_ROOT) \
		--dataset-config $(DATASET_CONFIG) \
		$(if $(strip $(USE_REAL_DATA)),--use-real-data,) \
		$(if $(strip $(NO_AUG)),--no-augment,)

retrain-best: ## Retrain best configuration across multiple seeds
	@$(PY) -m dataaug_multi_both.hpo.retrain \
		--best-path $(BEST_PATH) \
		--seeds $(RETRAIN_SEEDS) \
		--epochs $(RETRAIN_EPOCHS) \
		--output-root $(OUTPUT_ROOT)/retrain \
		--model $(MODEL) \
		--dataset-config $(DATASET_CONFIG) \
		$(if $(strip $(USE_REAL_DATA)),--use-real-data,)

hpo-two-stage-retrain: ## Retrain best two-stage HPO configuration across seeds
	@if [ ! -f "$(HPO_TWO_STAGE_BEST)" ]; then \
		echo "$(RED)✗ Best trial file not found: $(HPO_TWO_STAGE_BEST)$(NC)"; \
		echo "$(YELLOW)Run: make hpo-two-stage$(NC)"; \
		exit 1; \
	fi
	@$(PY) -m dataaug_multi_both.hpo.retrain \
		--best-path $(HPO_TWO_STAGE_BEST) \
		--seeds $(RETRAIN_SEEDS) \
		--epochs $(RETRAIN_EPOCHS) \
		--output-root $(HPO_TWO_STAGE_OUTPUT)/retrain \
		--model $(HPO_TWO_STAGE_MODEL) \
		--dataset-config $(HPO_TWO_STAGE_DATASET) \
		--use-real-data

# Development helpers
.PHONY: watch-logs show-disk-usage list-trials

watch-logs: ## Watch training logs in real-time (use TRIAL_ID=<uuid>)
	@if [ -z "$(TRIAL_ID)" ]; then \
		echo "$(RED)✗ TRIAL_ID not set$(NC)"; \
		echo "$(YELLOW)Usage: make watch-logs TRIAL_ID=<uuid>$(NC)"; \
		exit 1; \
	fi
	tail -f $(EXPERIMENTS_DIR)/trial_$(TRIAL_ID)/logs/stdout.log

show-disk-usage: ## Show disk usage of experiments directory
	@echo "$(BLUE)Disk usage summary:$(NC)"
	@du -sh $(EXPERIMENTS_DIR)
	@echo ""
	@echo "$(BLUE)Top 10 largest directories:$(NC)"
	@du -sh $(EXPERIMENTS_DIR)/*/ 2>/dev/null | sort -rh | head -n 10

list-trials: ## List all trial directories
	@echo "$(BLUE)Trial directories:$(NC)"
	@ls -lh $(EXPERIMENTS_DIR)/trial_* 2>/dev/null | grep '^d' || echo "No trial directories found"

# Info target
info: ## Display project information
	@echo "$(BLUE)Project Information$(NC)"
	@echo "===================="
	@echo "Project Root:      $(PROJECT_ROOT)"
	@echo "Source Dir:        $(SRC_DIR)"
	@echo "Tests Dir:         $(TESTS_DIR)"
	@echo "Experiments Dir:   $(EXPERIMENTS_DIR)"
	@echo "Study DB:          $(STUDY_DB)"
	@echo "MLflow URI:        $(MLFLOW_URI)"
	@echo "Docker Image:      $(DOCKER_FULL_NAME)"
	@echo ""
	@echo "$(BLUE)Python Environment$(NC)"
	@echo "===================="
	@poetry --version || echo "Poetry not installed"
	@python --version || true
TRIALS_A ?= 380
TRIALS_B ?= 120
EPOCHS_A ?= 5
EPOCHS_B ?= 30
K_TOP ?= 5
MODEL ?= microsoft/deberta-v3-base
SEED ?= 42
OUTPUT_ROOT ?= experiments/hpo_default
TIMEOUT ?= 604800
RETRAIN_SEEDS ?= 3
RETRAIN_EPOCHS ?= 10
BEST_PATH ?= $(OUTPUT_ROOT)/artifacts/hpo/aggregate/best_trial.json
DATASET_CONFIG ?= configs/data/dataset.yaml
USE_REAL_DATA ?=
