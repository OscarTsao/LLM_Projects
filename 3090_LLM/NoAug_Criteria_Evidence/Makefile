.PHONY: help setup install clean clean-all
.PHONY: groundtruth train train-evidence
.PHONY: hpo-s0 hpo-s1 hpo-s2 refit eval export
.PHONY: lint format test test-cov test-groundtruth
.PHONY: pre-commit-install pre-commit-run
.PHONY: tune-criteria-max tune-evidence-max tune-share-max tune-joint-max
.PHONY: tune-criteria-supermax tune-evidence-supermax tune-share-supermax tune-joint-supermax tune-all-supermax

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
	@echo "$(BLUE)PSY Agents NO-AUG - Available Targets$(NC)"
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
	@echo "  make train TASK=<task> MODEL=<model> - Train with custom task/model"
	@echo ""
	@echo "$(GREEN)Hyperparameter Optimization (HPO):$(NC)"
	@echo "  make hpo-s0             - Stage 0: Sanity check (8 trials)"
	@echo "  make hpo-s1             - Stage 1: Coarse search (20 trials)"
	@echo "  make hpo-s2             - Stage 2: Fine search (50 trials)"
	@echo "  make refit              - Stage 3: Refit best model on train+val"
	@echo "  make full-hpo-all       - Multi-stage HPO for ALL architectures"
	@echo "  make maximal-hpo-all    - Maximal HPO for ALL architectures"
	@echo "  make tune-criteria-supermax  - Super-max HPO: criteria (5000 trials, 100 epochs)"
	@echo "  make tune-evidence-supermax  - Super-max HPO: evidence (8000 trials, 100 epochs)"
	@echo "  make tune-share-supermax     - Super-max HPO: share (3000 trials, 100 epochs)"
	@echo "  make tune-joint-supermax     - Super-max HPO: joint (3000 trials, 100 epochs)"
	@echo "  make tune-all-supermax       - Super-max HPO: ALL architectures sequentially (~19K trials)"
	@echo ""
	@echo "$(GREEN)Evaluation:$(NC)"
	@echo "  make eval               - Evaluate best model on test set"
	@echo "  make export             - Export metrics from MLflow"
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
	@echo "  make train TASK=criteria MODEL=roberta_base"
	@echo "  make hpo-s1 TASK=criteria"
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
	@echo "  2. Train a model: make train"
	@echo "  3. Run HPO: make hpo-s0"

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
	@poetry run python -c "import psy_agents_noaug; print('✓ Package imports successfully')"
	@echo "Checking configs..."
	@test -f configs/config.yaml && echo "✓ Main config exists" || echo "✗ Missing config.yaml"
	@echo "$(GREEN)✓ Sanity checks passed$(NC)"

#==============================================================================
# Data Generation
#==============================================================================

## groundtruth: Generate ground truth files from HuggingFace
groundtruth:
	@echo "$(BLUE)Generating ground truth from HuggingFace...$(NC)"
	poetry run python -m psy_agents_noaug.cli make_groundtruth data=hf_redsm5

## groundtruth-local: Generate ground truth from local CSV files
groundtruth-local:
	@echo "$(BLUE)Generating ground truth from local CSV...$(NC)"
	poetry run python -m psy_agents_noaug.cli make_groundtruth data=local_csv

#==============================================================================
# Training
#==============================================================================

# Default values for training
TASK ?= criteria
MODEL ?= roberta_base

## train: Train a model with specified task and model
train:
	@echo "$(BLUE)Training model...$(NC)"
	@echo "Task: $(TASK)"
	@echo "Model: $(MODEL)"
	poetry run python -m psy_agents_noaug.cli train task=$(TASK) model=$(MODEL)

## train-evidence: Train evidence task with default model
train-evidence:
	@echo "$(BLUE)Training evidence model...$(NC)"
	poetry run python -m psy_agents_noaug.cli train task=evidence model=roberta_base

#==============================================================================
# Hyperparameter Optimization (HPO)
#==============================================================================

HPO_TASK ?= criteria
HPO_MODEL ?= roberta_base

## hpo-s0: Run HPO stage 0 (sanity check with 2 trials)
hpo-s0:
	@echo "$(BLUE)Running HPO Stage 0: Sanity Check$(NC)"
	poetry run python -m psy_agents_noaug.cli hpo hpo=stage0_sanity task=$(HPO_TASK) model=$(HPO_MODEL)

## hpo-s1: Run HPO stage 1 (coarse search with 20 trials)
hpo-s1:
	@echo "$(BLUE)Running HPO Stage 1: Coarse Search$(NC)"
	poetry run python -m psy_agents_noaug.cli hpo hpo=stage1_coarse task=$(HPO_TASK) model=$(HPO_MODEL)

## hpo-s2: Run HPO stage 2 (fine search with 50 trials)
hpo-s2:
	@echo "$(BLUE)Running HPO Stage 2: Fine Search$(NC)"
	poetry run python -m psy_agents_noaug.cli hpo hpo=stage2_fine task=$(HPO_TASK) model=$(HPO_MODEL)

## refit: Run HPO stage 3 (refit best model on train+val)
refit:
	@echo "$(BLUE)Running HPO Stage 3: Refit Best Model$(NC)"
	@if [ ! -f "outputs/hpo_stage2/best_config.yaml" ]; then \
		echo "$(RED)✗ No best config found. Run hpo-s2 first.$(NC)"; \
		exit 1; \
	fi
	poetry run python -m psy_agents_noaug.cli refit task=$(HPO_TASK) best_config=outputs/hpo_stage2/best_config.yaml

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
	poetry run python -m psy_agents_noaug.cli evaluate_best checkpoint=$(CHECKPOINT)

## export: Export metrics from MLflow
export:
	@echo "$(BLUE)Exporting metrics...$(NC)"
	poetry run python -m psy_agents_noaug.cli export_metrics

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
	poetry run pytest tests/ -v --cov=src/psy_agents_noaug --cov-report=html --cov-report=term
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

## quick-start: Quick start workflow (setup + groundtruth + sanity HPO)
quick-start: setup groundtruth hpo-s0
	@echo "$(GREEN)✓ Quick start complete!$(NC)"
	@echo ""
	@echo "Next steps:"
	@echo "  - Run full HPO: make hpo-s1 hpo-s2 refit"
	@echo "  - Train model: make train"
	@echo "  - Evaluate: make eval"

## full-hpo: Run complete HPO pipeline (stages 0-3) for one architecture
full-hpo: hpo-s0 hpo-s1 hpo-s2 refit
	@echo "$(GREEN)✓ Full HPO pipeline complete!$(NC)"

## full-hpo-all: Run complete HPO pipeline for ALL architectures sequentially
full-hpo-all:
	@echo "$(BLUE)Running multi-stage HPO for all architectures...$(NC)"
	poetry run python scripts/run_all_hpo.py --mode multistage

## maximal-hpo-all: Run maximal HPO for ALL architectures sequentially
maximal-hpo-all:
	@echo "$(BLUE)Running maximal HPO for all architectures...$(NC)"
	poetry run python scripts/run_all_hpo.py --mode maximal

#==============================================================================
# Info
#==============================================================================

## info: Show project information
info:
	@echo "$(BLUE)Project Information$(NC)"
	@echo "  Name: PSY Agents NO-AUG"
	@echo "  Path: $(shell pwd)"
	@echo "  Python: $(shell poetry run python --version)"
	@echo "  Poetry: $(shell poetry --version)"
	@echo ""
	@echo "$(BLUE)Directories:$(NC)"
	@echo "  Configs: ./configs"
	@echo "  Data: ./data"
	@echo "  Source: ./src/psy_agents_noaug"
	@echo "  Tests: ./tests"
	@echo "  Scripts: ./scripts"
	@echo ""
	@echo "$(BLUE)Status:$(NC)"
	@test -d .git && echo "  Git: ✓ Initialized" || echo "  Git: ✗ Not initialized"
	@test -f poetry.lock && echo "  Poetry: ✓ Dependencies locked" || echo "  Poetry: ✗ No lock file"
	@test -d data/processed && echo "  Data: ✓ Processed data exists" || echo "  Data: ✗ Run 'make groundtruth'"
	@test -d mlruns && echo "  MLflow: ✓ Runs directory exists" || echo "  MLflow: ✗ No runs yet"

tune-criteria-max:
	@HPO_EPOCHS?=6; \
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python scripts/tune_max.py --agent criteria --study noaug-criteria-max --n-trials 800 --parallel 4 --outdir $${HPO_OUTDIR:-./_runs}

tune-evidence-max:
	@HPO_EPOCHS?=6; \
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python scripts/tune_max.py --agent evidence --study noaug-evidence-max --n-trials 1200 --parallel 4 --outdir $${HPO_OUTDIR:-./_runs}

tune-share-max:
	@HPO_EPOCHS?=6; \
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python scripts/tune_max.py --agent share --study noaug-share-max --n-trials 600 --parallel 4 --outdir $${HPO_OUTDIR:-./_runs}

tune-joint-max:
	@HPO_EPOCHS?=6; \
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python scripts/tune_max.py --agent joint --study noaug-joint-max --n-trials 600 --parallel 4 --outdir $${HPO_OUTDIR:-./_runs}

#==============================================================================
# Super-Max HPO (100 epochs + EarlyStopping, very high trial counts)
#==============================================================================

# Override trial counts and parallelism via environment:
#   N_TRIALS_CRITERIA=6000 PAR=4 make tune-criteria-supermax
# OPTIMIZATION: Reduced PAR from 4 → 3 → 2 to prevent GPU memory pressure
# Previous: 4 parallel (4 × ~6GB = ~24GB) → GPU at limit, OOM crashes
# Attempted: 3 parallel (3 × ~6GB = ~18GB) → Still hit 97% usage (23.3GB), crashes persisted
# Current: 2 parallel (2 × ~6-8GB = 12-16GB) → Safer, leaves 8-12GB margin
# Trade-off: -33% throughput, but dramatically improved stability (worth it!)
PAR ?= 2
N_TRIALS_CRITERIA ?= 5000
N_TRIALS_EVIDENCE ?= 8000
N_TRIALS_SHARE ?= 3000
N_TRIALS_JOINT ?= 3000
HPO_OUTDIR ?= ./_runs

## tune-criteria-supermax: Run super-max HPO for criteria (5000 trials, 100 epochs, ES patience=20)
tune-criteria-supermax:
	@echo "$(BLUE)Running SUPER-MAX HPO for Criteria...$(NC)"
	@echo "Trials: $(N_TRIALS_CRITERIA) | Parallel: $(PAR) (reduced 4→3→2 for stability) | Epochs: 100 | Patience: 20"
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HPO_EPOCHS=100 HPO_PATIENCE=20 poetry run python scripts/tune_max.py \
		--agent criteria --study noaug-criteria-supermax \
		--n-trials $(N_TRIALS_CRITERIA) --parallel $(PAR) \
		--outdir $(HPO_OUTDIR)

## tune-evidence-supermax: Run super-max HPO for evidence (8000 trials, 100 epochs, ES patience=20)
tune-evidence-supermax:
	@echo "$(BLUE)Running SUPER-MAX HPO for Evidence...$(NC)"
	@echo "Trials: $(N_TRIALS_EVIDENCE) | Parallel: $(PAR) | Epochs: 100 | Patience: 20"
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HPO_EPOCHS=100 HPO_PATIENCE=20 poetry run python scripts/tune_max.py \
		--agent evidence --study noaug-evidence-supermax \
		--n-trials $(N_TRIALS_EVIDENCE) --parallel $(PAR) \
		--outdir $(HPO_OUTDIR)

## tune-share-supermax: Run super-max HPO for share (3000 trials, 100 epochs, ES patience=20)
tune-share-supermax:
	@echo "$(BLUE)Running SUPER-MAX HPO for Share...$(NC)"
	@echo "Trials: $(N_TRIALS_SHARE) | Parallel: $(PAR) | Epochs: 100 | Patience: 20"
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HPO_EPOCHS=100 HPO_PATIENCE=20 poetry run python scripts/tune_max.py \
		--agent share --study noaug-share-supermax \
		--n-trials $(N_TRIALS_SHARE) --parallel $(PAR) \
		--outdir $(HPO_OUTDIR)

## tune-joint-supermax: Run super-max HPO for joint (3000 trials, 100 epochs, ES patience=20)
tune-joint-supermax:
	@echo "$(BLUE)Running SUPER-MAX HPO for Joint...$(NC)"
	@echo "Trials: $(N_TRIALS_JOINT) | Parallel: $(PAR) | Epochs: 100 | Patience: 20"
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HPO_EPOCHS=100 HPO_PATIENCE=20 poetry run python scripts/tune_max.py \
		--agent joint --study noaug-joint-supermax \
		--n-trials $(N_TRIALS_JOINT) --parallel $(PAR) \
		--outdir $(HPO_OUTDIR)

## tune-all-supermax: Run super-max HPO for ALL architectures sequentially (criteria→evidence→share→joint)
tune-all-supermax:
	@echo "$(BLUE)======================================================================"
	@echo "  Running Super-Max HPO for ALL Architectures Sequentially"
	@echo "======================================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Sequence: Criteria → Evidence → Share → Joint$(NC)"
	@echo "$(YELLOW)Total trials: ~19,000 (5000+8000+3000+3000)$(NC)"
	@echo "$(YELLOW)Estimated time: ~120-180 hours with PAR=2 (reduced for GPU stability)$(NC)"
	@echo ""
	@echo "$(GREEN)Starting...$(NC)"
	@echo ""
	@echo "$(BLUE)[1/4] Running Criteria (5000 trials)...$(NC)"
	@$(MAKE) tune-criteria-supermax
	@echo ""
	@echo "$(GREEN)✓ Criteria complete!$(NC)"
	@echo ""
	@echo "$(BLUE)[2/4] Running Evidence (8000 trials)...$(NC)"
	@$(MAKE) tune-evidence-supermax
	@echo ""
	@echo "$(GREEN)✓ Evidence complete!$(NC)"
	@echo ""
	@echo "$(BLUE)[3/4] Running Share (3000 trials)...$(NC)"
	@$(MAKE) tune-share-supermax
	@echo ""
	@echo "$(GREEN)✓ Share complete!$(NC)"
	@echo ""
	@echo "$(BLUE)[4/4] Running Joint (3000 trials)...$(NC)"
	@$(MAKE) tune-joint-supermax
	@echo ""
	@echo "$(GREEN)✓ Joint complete!$(NC)"
	@echo ""
	@echo "$(GREEN)======================================================================"
	@echo "  ✓ ALL SUPER-MAX HPO RUNS COMPLETE!"
	@echo "======================================================================$(NC)"
	@echo ""
	@echo "Results saved to: $(HPO_OUTDIR)"
	@echo "View with: mlflow ui --backend-store-uri sqlite:///mlflow.db"
	@echo ""
