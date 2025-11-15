.PHONY: help setup install clean clean-all
.PHONY: groundtruth train train-evidence train-aug
.PHONY: hpo-s0 hpo-s1 hpo-s2 refit eval export
.PHONY: lint format test test-cov test-groundtruth
.PHONY: test-aug test-contract test-pipelines test-no-leak verify-aug
.PHONY: pre-commit-install pre-commit-run

# Default target
.DEFAULT_GOAL := help

# Color codes for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

#==============================================================================
# Help
#==============================================================================

## help: Show this help message
help:
	@echo "$(BLUE)PSY Agents AUG - Available Targets$(NC)"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  make setup              - Full setup (install + pre-commit + sanity check)"
	@echo "  make install            - Install dependencies with poetry"
	@echo "  make install-dev        - Install with development dependencies"
	@echo ""
	@echo "$(GREEN)Data:$(NC)"
	@echo "  make groundtruth        - Generate ground truth files (HuggingFace default)"
	@echo "  make groundtruth-local  - Generate ground truth from local CSV"
	@echo ""
	@echo "$(GREEN)Training:$(NC)"
	@echo "  make train              - Train default model (criteria, roberta_base)"
	@echo "  make train-evidence     - Train evidence task"
	@echo "  make train-aug          - Train with augmentation enabled"
	@echo "  make train TASK=<task> MODEL=<model> - Train with custom task/model"
	@echo ""
	@echo "$(GREEN)Hyperparameter Optimization (HPO):$(NC)"
	@echo "  make hpo-s0             - Stage 0: Sanity check (2 trials)"
	@echo "  make hpo-s1             - Stage 1: Coarse search (20 trials)"
	@echo "  make hpo-s2             - Stage 2: Fine search (50 trials)"
	@echo "  make hpo-s0-aug         - Stage 0 with augmentation"
	@echo "  make hpo-s1-aug         - Stage 1 with augmentation"
	@echo "  make hpo-s2-aug         - Stage 2 with augmentation"
	@echo "  make refit              - Stage 3: Refit best model on train+val"
	@echo ""
	@echo "$(GREEN)Evaluation:$(NC)"
	@echo "  make eval               - Evaluate best model on test set"
	@echo "  make export             - Export metrics from MLflow"
	@echo ""
	@echo "$(GREEN)Augmentation Testing:$(NC)"
	@echo "  make test-aug           - Run all augmentation tests"
	@echo "  make test-contract      - Test augmentation contracts (determinism, train-only)"
	@echo "  make test-pipelines     - Test augmentation pipelines"
	@echo "  make test-no-leak       - Test no data leakage in val/test"
	@echo "  make verify-aug         - Verify augmentation setup with examples"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  make lint               - Run linters (ruff + black --check)"
	@echo "  make format             - Format code (ruff --fix + black)"
	@echo "  make test               - Run all tests"
	@echo "  make test-cov           - Run tests with coverage report"
	@echo "  make test-groundtruth   - Run ground truth validation tests"
	@echo ""
	@echo "$(GREEN)Pre-commit:$(NC)"
	@echo "  make pre-commit-install - Install pre-commit hooks"
	@echo "  make pre-commit-run     - Run pre-commit on all files"
	@echo ""
	@echo "$(GREEN)Cleaning:$(NC)"
	@echo "  make clean              - Remove caches and temp files"
	@echo "  make clean-all          - Clean everything (including data/mlruns)"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make setup"
	@echo "  make groundtruth"
	@echo "  make train-aug"
	@echo "  make hpo-s1-aug TASK=criteria"
	@echo "  make verify-aug"
	@echo ""

#==============================================================================
# Setup
#==============================================================================

## setup: Complete setup (install + pre-commit + sanity checks)
setup: install pre-commit-install sanity-check
	@echo "$(GREEN)✓ Setup complete!$(NC)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Generate ground truth: make groundtruth"
	@echo "  2. Test augmentation: make verify-aug"
	@echo "  3. Train with augmentation: make train-aug"
	@echo "  4. Run HPO: make hpo-s0-aug"

## install: Install dependencies using Poetry
install:
	@echo "$(BLUE)Installing dependencies...$(NC)"
	poetry install

## install-dev: Install with development dependencies
install-dev:
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	poetry install --with dev

## sanity-check: Run sanity checks
sanity-check:
	@echo "$(BLUE)Running sanity checks...$(NC)"
	@echo "Checking Python version..."
	@poetry run python --version
	@echo "Checking imports..."
	@poetry run python -c "import psy_agents_aug; print('✓ Package imports successfully')"
	@poetry run python -c "import psy_agents_aug.augment; print('✓ Augment module imports successfully')"
	@echo "Checking configs..."
	@test -f configs/config.yaml && echo "✓ Main config exists" || echo "✗ Missing config.yaml"
	@test -d configs/augmentation && echo "✓ Augment configs exist" || echo "⚠ Missing augment configs"
	@echo "$(GREEN)✓ Sanity checks passed$(NC)"

#==============================================================================
# Data Generation
#==============================================================================

## groundtruth: Generate ground truth files from HuggingFace
groundtruth:
	@echo "$(BLUE)Generating ground truth from HuggingFace...$(NC)"
	poetry run python -m psy_agents_aug.cli make_groundtruth data=hf_redsm5

## groundtruth-local: Generate ground truth from local CSV files
groundtruth-local:
	@echo "$(BLUE)Generating ground truth from local CSV...$(NC)"
	poetry run python -m psy_agents_aug.cli make_groundtruth data=local_csv

#==============================================================================
# Training
#==============================================================================

# Default values for training
TASK ?= criteria
MODEL ?= roberta_base
AUG_PIPELINE ?= nlpaug_pipeline

## train: Train a model with specified task and model
train:
	@echo "$(BLUE)Training model...$(NC)"
	@echo "Task: $(TASK)"
	@echo "Model: $(MODEL)"
	@echo "Augmentation: DISABLED"
	poetry run python -m psy_agents_aug.cli train task=$(TASK) model=$(MODEL)

## train-evidence: Train evidence task with default model
train-evidence:
	@echo "$(BLUE)Training evidence model...$(NC)"
	poetry run python -m psy_agents_aug.cli train task=evidence model=roberta_base

## train-aug: Train with augmentation enabled
train-aug:
	@echo "$(BLUE)Training model with augmentation...$(NC)"
	@echo "Task: $(TASK)"
	@echo "Model: $(MODEL)"
	@echo "Augmentation: ENABLED (pipeline=$(AUG_PIPELINE))"
	poetry run python -m psy_agents_aug.cli train task=$(TASK) model=$(MODEL) \
		augmentation.enabled=true augmentation.pipeline=$(AUG_PIPELINE)

#==============================================================================
# Hyperparameter Optimization (HPO)
#==============================================================================

HPO_TASK ?= criteria
HPO_MODEL ?= roberta_base

## hpo-s0: Run HPO stage 0 (sanity check with 2 trials)
hpo-s0:
	@echo "$(BLUE)Running HPO Stage 0: Sanity Check$(NC)"
	poetry run python -m psy_agents_aug.cli hpo hpo=stage0_sanity task=$(HPO_TASK) model=$(HPO_MODEL)

## hpo-s1: Run HPO stage 1 (coarse search with 20 trials)
hpo-s1:
	@echo "$(BLUE)Running HPO Stage 1: Coarse Search$(NC)"
	poetry run python -m psy_agents_aug.cli hpo hpo=stage1_coarse task=$(HPO_TASK) model=$(HPO_MODEL)

## hpo-s2: Run HPO stage 2 (fine search with 50 trials)
hpo-s2:
	@echo "$(BLUE)Running HPO Stage 2: Fine Search$(NC)"
	poetry run python -m psy_agents_aug.cli hpo hpo=stage2_fine task=$(HPO_TASK) model=$(HPO_MODEL)

## hpo-s0-aug: Run HPO stage 0 with augmentation
hpo-s0-aug:
	@echo "$(BLUE)Running HPO Stage 0 with Augmentation$(NC)"
	poetry run python -m psy_agents_aug.cli hpo hpo=stage0_sanity task=$(HPO_TASK) model=$(HPO_MODEL) \
		augmentation.enabled=true augmentation.pipeline=$(AUG_PIPELINE)

## hpo-s1-aug: Run HPO stage 1 with augmentation
hpo-s1-aug:
	@echo "$(BLUE)Running HPO Stage 1 with Augmentation$(NC)"
	poetry run python -m psy_agents_aug.cli hpo hpo=stage1_coarse task=$(HPO_TASK) model=$(HPO_MODEL) \
		augmentation.enabled=true augmentation.pipeline=$(AUG_PIPELINE)

## hpo-s2-aug: Run HPO stage 2 with augmentation
hpo-s2-aug:
	@echo "$(BLUE)Running HPO Stage 2 with Augmentation$(NC)"
	poetry run python -m psy_agents_aug.cli hpo hpo=stage2_fine task=$(HPO_TASK) model=$(HPO_MODEL) \
		augmentation.enabled=true augmentation.pipeline=$(AUG_PIPELINE)

## refit: Run HPO stage 3 (refit best model on train+val)
refit:
	@echo "$(BLUE)Running HPO Stage 3: Refit Best Model$(NC)"
	@if [ ! -f "outputs/hpo_stage2/best_config.yaml" ]; then \
		echo "$(RED)✗ No best config found. Run hpo-s2 first.$(NC)"; \
		exit 1; \
	fi
	poetry run python -m psy_agents_aug.cli refit task=$(HPO_TASK) best_config=outputs/hpo_stage2/best_config.yaml

#==============================================================================
# Evaluation
#==============================================================================

CHECKPOINT ?= outputs/checkpoints/best_checkpoint.pt

## eval: Evaluate best model on test set
eval:
	@echo "$(BLUE)Evaluating best model...$(NC)"
	@if [ ! -f "$(CHECKPOINT)" ]; then \
		echo "$(YELLOW)⚠ Checkpoint not found: $(CHECKPOINT)$(NC)"; \
		echo "Use: make eval CHECKPOINT=path/to/checkpoint.pt"; \
	fi
	poetry run python -m psy_agents_aug.cli evaluate_best checkpoint=$(CHECKPOINT)

## export: Export metrics from MLflow
export:
	@echo "$(BLUE)Exporting metrics...$(NC)"
	poetry run python -m psy_agents_aug.cli export_metrics

#==============================================================================
# Augmentation Testing
#==============================================================================

## test-aug: Run all augmentation tests
test-aug:
	@echo "$(BLUE)Running augmentation tests...$(NC)"
	poetry run pytest tests/test_augment_*.py -v

## test-contract: Test augmentation contracts (determinism, train-only)
test-contract:
	@echo "$(BLUE)Testing augmentation contracts...$(NC)"
	poetry run pytest tests/test_augment_contract.py -v

## test-pipelines: Test augmentation pipelines
test-pipelines:
	@echo "$(BLUE)Testing augmentation pipelines...$(NC)"
	poetry run pytest tests/test_augment_pipelines.py -v

## test-no-leak: Test no data leakage in val/test
test-no-leak:
	@echo "$(BLUE)Testing no data leakage...$(NC)"
	poetry run pytest tests/test_augment_no_leak.py -v

## verify-aug: Verify augmentation setup with examples
verify-aug:
	@echo "$(BLUE)Verifying augmentation setup...$(NC)"
	@echo ""
	@echo "Testing NLPAug pipeline..."
	poetry run python -m psy_agents_aug.cli test_augmentation --pipeline nlpaug
	@echo ""
	@echo "Testing TextAttack pipeline (if available)..."
	poetry run python -m psy_agents_aug.cli test_augmentation --pipeline textattack || \
		echo "$(YELLOW)⚠ TextAttack not available$(NC)"
	@echo ""
	@echo "Testing Hybrid pipeline..."
	poetry run python -m psy_agents_aug.cli test_augmentation --pipeline hybrid || \
		echo "$(YELLOW)⚠ Hybrid pipeline not available$(NC)"
	@echo ""
	@echo "$(GREEN)✓ Augmentation verification complete$(NC)"

#==============================================================================
# Development
#==============================================================================

## lint: Run linters (ruff + black check)
lint:
	@echo "$(BLUE)Running linters...$(NC)"
	@echo "Running ruff..."
	poetry run ruff check src/ tests/ scripts/
	@echo "Running black..."
	poetry run black --check src/ tests/ scripts/
	@echo "$(GREEN)✓ Linting complete$(NC)"

## format: Format code (ruff --fix + black)
format:
	@echo "$(BLUE)Formatting code...$(NC)"
	poetry run ruff check --fix src/ tests/ scripts/
	poetry run black src/ tests/ scripts/
	@echo "$(GREEN)✓ Formatting complete$(NC)"

## test: Run all tests
test:
	@echo "$(BLUE)Running tests...$(NC)"
	poetry run pytest tests/ -v

## test-cov: Run tests with coverage report
test-cov:
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	poetry run pytest tests/ -v --cov=src/psy_agents_aug --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report: htmlcov/index.html$(NC)"

## test-groundtruth: Run ground truth validation tests
test-groundtruth:
	@echo "$(BLUE)Running ground truth tests...$(NC)"
	poetry run pytest tests/test_groundtruth.py -v

#==============================================================================
# Pre-commit
#==============================================================================

## pre-commit-install: Install pre-commit hooks
pre-commit-install:
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	poetry run pre-commit install
	@echo "$(GREEN)✓ Pre-commit hooks installed$(NC)"

## pre-commit-run: Run pre-commit on all files
pre-commit-run:
	@echo "$(BLUE)Running pre-commit on all files...$(NC)"
	poetry run pre-commit run --all-files

#==============================================================================
# Cleaning
#==============================================================================

## clean: Remove generated files and caches
clean:
	@echo "$(BLUE)Cleaning caches and temp files...$(NC)"
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf outputs/
	rm -rf multirun/
	@echo "$(GREEN)✓ Cleaned$(NC)"

## clean-all: Clean everything including data and MLflow runs
clean-all: clean
	@echo "$(BLUE)Cleaning data and MLflow runs...$(NC)"
	rm -rf mlruns/
	rm -rf data/processed/
	rm -rf *.db
	@echo "$(YELLOW)⚠ Removed processed data and MLflow runs$(NC)"

#==============================================================================
# Quick Workflows
#==============================================================================

## quick-start: Quick start workflow (setup + groundtruth + verify augmentation)
quick-start: setup groundtruth verify-aug
	@echo "$(GREEN)✓ Quick start complete!$(NC)"
	@echo ""
	@echo "Next steps:"
	@echo "  - Train with augmentation: make train-aug"
	@echo "  - Run HPO with augmentation: make hpo-s1-aug"
	@echo "  - Test augmentation: make test-aug"

## full-hpo: Run complete HPO pipeline (stages 0-3)
full-hpo: hpo-s0 hpo-s1 hpo-s2 refit
	@echo "$(GREEN)✓ Full HPO pipeline complete!$(NC)"

## full-hpo-aug: Run complete HPO pipeline with augmentation
full-hpo-aug: hpo-s0-aug hpo-s1-aug hpo-s2-aug refit
	@echo "$(GREEN)✓ Full HPO pipeline with augmentation complete!$(NC)"

#==============================================================================
# Comparison Experiments
#==============================================================================

## compare-aug: Run experiments comparing with/without augmentation
compare-aug:
	@echo "$(BLUE)Running comparison experiments...$(NC)"
	@echo ""
	@echo "1. Training baseline (no augmentation)..."
	$(MAKE) train TASK=$(TASK) MODEL=$(MODEL)
	@echo ""
	@echo "2. Training with NLPAug..."
	$(MAKE) train-aug TASK=$(TASK) MODEL=$(MODEL) AUG_PIPELINE=nlpaug_pipeline
	@echo ""
	@echo "3. Exporting metrics..."
	$(MAKE) export
	@echo ""
	@echo "$(GREEN)✓ Comparison complete! Check exported metrics.$(NC)"

#==============================================================================
# Info
#==============================================================================

## info: Show project information
info:
	@echo "$(BLUE)Project Information$(NC)"
	@echo "  Name: PSY Agents AUG"
	@echo "  Path: $(shell pwd)"
	@echo "  Python: $(shell poetry run python --version)"
	@echo "  Poetry: $(shell poetry --version)"
	@echo ""
	@echo "$(BLUE)Directories:$(NC)"
	@echo "  Configs: ./configs"
	@echo "  Data: ./data"
	@echo "  Source: ./src/psy_agents_aug"
	@echo "  Augment: ./src/psy_agents_aug/augment"
	@echo "  Tests: ./tests"
	@echo "  Scripts: ./scripts"
	@echo ""
	@echo "$(BLUE)Status:$(NC)"
	@test -d .git && echo "  Git: ✓ Initialized" || echo "  Git: ✗ Not initialized"
	@test -f poetry.lock && echo "  Poetry: ✓ Dependencies locked" || echo "  Poetry: ✗ No lock file"
	@test -d data/processed && echo "  Data: ✓ Processed data exists" || echo "  Data: ✗ Run 'make groundtruth'"
	@test -d mlruns && echo "  MLflow: ✓ Runs directory exists" || echo "  MLflow: ✗ No runs yet"
	@test -d src/psy_agents_aug/augment && echo "  Augment: ✓ Module exists" || echo "  Augment: ✗ Missing module"
