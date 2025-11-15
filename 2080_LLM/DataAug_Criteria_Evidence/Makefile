.PHONY: help setup install clean clean-all
.PHONY: groundtruth train train-evidence
.PHONY: hpo-s0 hpo-s1 hpo-s2 refit eval export
.PHONY: test-deployment test-experiment test-cicd test-monitoring test-api test-security test-registry test-ab test-interpretability test-data-quality test-feature-store test-governance
.PHONY: lint format test test-cov test-groundtruth
.PHONY: pre-commit-install pre-commit-run
.PHONY: tune-criteria-max tune-evidence-max tune-evidence-aug tune-evidence-joint tune-share-max tune-joint-max
.PHONY: tune-evidence-3stage-full tune-evidence-3stage-smoke tune-criteria-3stage-full
.PHONY: tune-criteria-supermax tune-evidence-supermax tune-share-supermax tune-joint-supermax tune-all-supermax
.PHONY: full-hpo-all maximal-hpo-all show-best-%
.PHONY: stage-a stage-a-test stage-a-all stage-b stage-b-test stage-c stage-c-test full-supermax build-ensemble build-ensemble-test monitor-hpo list-hpo-studies check-hpo-health
.PHONY: compare-pruners compare-pruners-test show-pruner-budget show-pruner-recommendations
.PHONY: compare-samplers compare-samplers-test show-sampler-recommendations
.PHONY: audit audit-strict sbom licenses compliance bench verify-determinism train-with-json-logs
.PHONY: retrain-evidence-aug retrain-evidence-noaug
.PHONY: docker-build docker-test docker-run docker-shell docker-clean docker-mlflow

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
	@echo "  make tune-evidence-3stage-full   - 3-Stage Evidence HPO: A→B→C (1200+600+240 trials)"
	@echo "  make tune-evidence-3stage-smoke  - 3-Stage Evidence Smoke Test (10+12+8 trials, 3 epochs)"
	@echo "  make tune-criteria-3stage-full   - 3-Stage Criteria HPO: A→B→C (800+400+160 trials)"
	@echo "  make tune-criteria-supermax  - Super-max HPO: criteria (5000 trials, 100 epochs)"
	@echo "  make tune-evidence-supermax  - Super-max HPO: evidence (8000 trials, 100 epochs)"
	@echo "  make tune-share-supermax     - Super-max HPO: share (3000 trials, 100 epochs)"
	@echo "  make tune-joint-supermax     - Super-max HPO: joint (3000 trials, 100 epochs)"
	@echo "  make tune-all-supermax       - Super-max HPO: ALL architectures sequentially (~19K trials)"
	@echo ""
	@echo "$(GREEN)SUPERMAX Multi-Stage HPO (Phase 5):$(NC)"
	@echo "  make stage-a AGENT=<agent>   - Stage-A baseline exploration (default: 1000 trials, 6 epochs)"
	@echo "  make stage-a-test AGENT=<agent> - Stage-A smoke test (20 trials, 2 epochs)"
	@echo "  make stage-a-all             - Stage-A for all 4 agents"
	@echo "  make stage-b AGENT=<agent> STAGE_A_JSON=<path> - Stage-B multi-objective (1500 trials, 10 epochs)"
	@echo "  make stage-b-test AGENT=<agent> STAGE_A_JSON=<path> - Stage-B smoke test (20 trials, 3 epochs)"
	@echo "  make stage-c AGENT=<agent> STAGE_B_JSON=<path> - Stage-C K-fold CV (5-fold, 15 epochs)"
	@echo "  make stage-c-test AGENT=<agent> STAGE_B_JSON=<path> - Stage-C smoke test (2-fold, 3 epochs)"
	@echo "  make full-supermax AGENT=<agent> - Complete Stage-A→B→C pipeline"
	@echo ""
	@echo "$(GREEN)SUPERMAX Ensemble Building (Phase 6):$(NC)"
	@echo "  make build-ensemble AGENT=<agent> ENSEMBLE_INPUT=<path> ENSEMBLE_TYPE=<pareto|cv> - Build ensemble from results"
	@echo "  make build-ensemble-test AGENT=<agent> ENSEMBLE_INPUT=<path> ENSEMBLE_TYPE=<type> - Test with simulated predictions"
	@echo ""
	@echo "$(GREEN)SUPERMAX Monitoring & Management (Phase 7):$(NC)"
	@echo "  make list-hpo-studies HPO_STORAGE=<storage> - List all studies in database"
	@echo "  make monitor-hpo HPO_STUDY_NAME=<study> HPO_STORAGE=<storage> - Monitor study in real-time"
	@echo "  make check-hpo-health HPO_STUDY_NAME=<study> HPO_STORAGE=<storage> - Run health check"
	@echo ""
	@echo "$(GREEN)SUPERMAX Advanced Pruners & Search Strategies (Phase 8):$(NC)"
	@echo "  make compare-pruners           - Compare pruners on ML-style benchmark (50 trials)"
	@echo "  make compare-pruners-test      - Quick pruner comparison test (20 trials)"
	@echo "  make show-pruner-budget        - Show resource budget estimation examples"
	@echo "  make show-pruner-recommendations - Show pruner recommendations for common scenarios"
	@echo ""
	@echo "$(GREEN)SUPERMAX BOHB & Advanced Samplers (Phase 9):$(NC)"
	@echo "  make compare-samplers          - Compare samplers on benchmark (100 trials)"
	@echo "  make compare-samplers-test     - Quick sampler comparison test (50 trials)"
	@echo "  make show-sampler-recommendations - Show sampler recommendations for common scenarios"
	@echo ""
	@echo "$(GREEN)Evaluation:$(NC)"
	@echo "  make eval               - Evaluate best model on test set"
	@echo "  make export             - Export metrics from MLflow"
	@echo ""
	@echo "$(GREEN)Deployment (Phase 14):$(NC)"
	@echo "  make test-deployment    - Test deployment functionality (registry, packaging, deployment)"
	@echo ""
	@echo "$(GREEN)Experiment Tracking (Phase 15):$(NC)"
	@echo "  make test-experiment    - Test experiment tracking functionality (tracking, versioning, reproducibility, comparison)"
	@echo ""
	@echo "$(GREEN)CI/CD Integration (Phase 16):$(NC)"
	@echo "  make test-cicd          - Test CI/CD workflows (workflow manager, quality gates, pipeline orchestration)"
	@echo ""
	@echo "$(GREEN)Model Monitoring (Phase 17):$(NC)"
	@echo "  make test-monitoring    - Test monitoring system (performance, drift detection, health checks, alerts)"
	@echo ""
	@echo "$(GREEN)API Serving (Phase 18):$(NC)"
	@echo "  make test-api           - Test REST API endpoints (predictions, health, metrics, batch processing)"
	@echo ""
	@echo "$(GREEN)Security & Authentication (Phase 19):$(NC)"
	@echo "  make test-security      - Test security features (API keys, rate limiting, auth, headers)"
	@echo ""
	@echo "$(GREEN)Model Registry (Phase 20):$(NC)"
	@echo "  make test-registry      - Test model versioning, metadata, promotion workflows, lineage tracking"
	@echo ""
	@echo "$(GREEN)A/B Testing & Experimentation (Phase 21):$(NC)"
	@echo "  make test-ab            - Test A/B testing framework (traffic splitting, experiments, stats, tracking)"
	@echo ""
	@echo "$(GREEN)Model Interpretability & Explainability (Phase 22):$(NC)"
	@echo "  make test-interpretability - Test SHAP, attention viz, feature importance, explanations, counterfactuals"
	@echo ""
	@echo "$(GREEN)Data Quality & Drift Detection (Phase 23):$(NC)"
	@echo "  make test-data-quality  - Test drift detection, validation, quality metrics, anomaly detection"
	@echo ""
	@echo "$(GREEN)Feature Store & Engineering (Phase 24):$(NC)"
	@echo "  make test-feature-store - Test feature registry, versioning, computation, serving"
	@echo ""
	@echo "$(GREEN)Model Governance & Compliance (Phase 25):$(NC)"
	@echo "  make test-governance    - Test model cards, bias detection, compliance, audit trails"
	@echo ""
	@echo "$(GREEN)Model Monitoring & Observability (Phase 26):$(NC)"
	@echo "  make test-monitoring    - Test performance monitoring, drift detection, health checks, alerts"
	@echo ""
	@echo "$(GREEN)Model Explainability & Interpretability (Phase 27):$(NC)"
	@echo "  make test-explainability - Test feature importance, attention visualization, explanations"
	@echo ""
	@echo "$(GREEN)Model Registry & Versioning (Phase 28):$(NC)"
	@echo "  make test-registry       - Test model registration, versioning, lifecycle management"
	@echo ""
	@echo "$(GREEN)Model Serving & Deployment (Phase 29):$(NC)"
	@echo "  make test-serving        - Test model loading, prediction API, batch inference, monitoring"
	@echo ""
	@echo "$(GREEN)Deployment Automation & CI/CD (Phase 30):$(NC)"
	@echo "  make test-deployment     - Test deployment strategies, health checks, rollback, orchestration"
	@echo ""
	@echo "$(GREEN)Integration & CI/CD (Phase 31):$(NC)"
	@echo "  make test-integration    - Test end-to-end workflows across all phases"
	@echo "  make docker-build        - Build Docker image"
	@echo "  make docker-up           - Start services with docker-compose"
	@echo "  make docker-down         - Stop docker-compose services"
	@echo "  make docker-test         - Run tests in Docker container"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  make lint               - Run linters (ruff + black --check)"
	@echo "  make typecheck          - Run mypy type checking"
	@echo "  make format             - Format code (ruff --fix + black)"
	@echo "  make test               - Run all tests"
	@echo "  make test-cov           - Run tests with coverage report"
	@echo "  make test-groundtruth   - Run ground truth validation tests"
	@echo "  make lock               - Lock dependencies to requirements-lock.txt"
	@echo ""
	@echo "$(GREEN)Pre-commit:$(NC)"
	@echo "  make pre-commit-install - Install pre-commit hooks"
	@echo "  make pre-commit-run     - Run pre-commit on all files"
	@echo "  make pre-commit-update  - Update pre-commit hook versions"
	@echo ""
	@echo "$(GREEN)Security & Compliance:$(NC)"
	@echo "  make audit              - Run security vulnerability scan"
	@echo "  make audit-strict       - Run audit and fail on critical vulnerabilities"
	@echo "  make sbom               - Generate Software Bill of Materials"
	@echo "  make licenses           - Generate third-party license report"
	@echo "  make compliance         - Run all compliance checks"
	@echo ""
	@echo "$(GREEN)Performance & Benchmarking:$(NC)"
	@echo "  make bench              - Run DataLoader performance benchmarks"
	@echo "  make verify-determinism - Verify deterministic behavior"
	@echo ""
	@echo "$(GREEN)Docker:$(NC)"
	@echo "  make docker-build       - Build Docker images (multi-stage)"
	@echo "  make docker-test        - Run tests in Docker container"
	@echo "  make docker-run         - Start production container"
	@echo "  make docker-shell       - Open shell in production container"
	@echo "  make docker-mlflow      - Start MLflow UI in Docker"
	@echo "  make docker-clean       - Clean Docker images and containers"
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

## lock: Lock dependencies to requirements-lock.txt
lock:
	@echo "$(BLUE)Locking dependencies...$(NC)"
	poetry export -f requirements.txt --without-hashes --with dev > requirements-lock.txt
	@echo "$(GREEN)✓ Dependencies locked to requirements-lock.txt$(NC)"

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

## prepare-tfidf: Pre-fit TF-IDF cache for augmentation (criteria)
prepare-tfidf:
	@echo "$(BLUE)Pre-fitting TF-IDF cache for augmentation (task=$(TASK))...$(NC)"
	poetry run python scripts/prepare_tfidf_cache.py --task $(TASK)

## prepare-tfidf-all: Pre-fit TF-IDF cache for all tasks
prepare-tfidf-all:
	@echo "$(BLUE)Pre-fitting TF-IDF cache for all tasks...$(NC)"
	$(MAKE) prepare-tfidf TASK=criteria
	$(MAKE) prepare-tfidf TASK=evidence

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

## hpo-s0: Run HPO stage 0 (sanity check with 8 trials)
hpo-s0:
	@echo "$(BLUE)Running HPO Stage 0: Sanity Check (8 trials, 3 epochs)$(NC)"
	HPO_EPOCHS=3 HPO_PATIENCE=5 poetry run python scripts/tune_max.py \
		--agent $(HPO_TASK) \
		--study $(HPO_TASK)-stage0-sanity \
		--n-trials 8 \
		--parallel 1 \
		--outdir outputs/hpo_stage0

## hpo-s1: Run HPO stage 1 (coarse search with 20 trials)
hpo-s1:
	@echo "$(BLUE)Running HPO Stage 1: Coarse Search (20 trials)$(NC)"
	HPO_EPOCHS=$${HPO_EPOCHS:-10} HPO_PATIENCE=$${HPO_PATIENCE:-10} poetry run python scripts/tune_max.py \
		--agent $(HPO_TASK) \
		--study $(HPO_TASK)-stage1-coarse \
		--n-trials 20 \
		--parallel 1 \
		--outdir outputs/hpo_stage1

## hpo-s2: Run HPO stage 2 (fine search with 50 trials)
hpo-s2:
	@echo "$(BLUE)Running HPO Stage 2: Fine Search (50 trials)$(NC)"
	HPO_EPOCHS=$${HPO_EPOCHS:-15} HPO_PATIENCE=$${HPO_PATIENCE:-15} poetry run python scripts/tune_max.py \
		--agent $(HPO_TASK) \
		--study $(HPO_TASK)-stage2-fine \
		--n-trials 50 \
		--parallel 1 \
		--outdir outputs/hpo_stage2

## refit: Run HPO stage 3 (refit best model on train+val)
refit:
	@echo "$(BLUE)Running HPO Stage 3: Refit Best Model$(NC)"
	@echo "$(YELLOW)Note: Use best config from stage 2 (outputs/hpo_stage2/$(HPO_TASK)_*/best_trial.json)$(NC)"
	@echo "$(YELLOW)Refit training should be done via: make train or scripts/train_*.py$(NC)"
	@echo "$(RED)Automated refit not yet implemented - manual training required$(NC)"

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
# Deployment & Model Registry (Phase 14)
#==============================================================================

## test-deployment: Test deployment functionality
test-deployment:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)Phase 14: Testing Deployment & Model Registry$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  - Model Registry (MLflow integration)"
	@echo "  - Deployment Packager (self-contained packages)"
	@echo "  - Model Deployer (deployment manifests)"
	@echo "  - Production model utilities"
	@echo ""
	poetry run python scripts/test_deployment.py
	@echo ""
	@echo "$(GREEN)✓ Deployment tests completed!$(NC)"

#==============================================================================
# Experiment Tracking & Reproducibility (Phase 15)
#==============================================================================

## test-experiment: Test experiment tracking functionality
test-experiment:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)Phase 15: Testing Experiment Tracking & Reproducibility$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  - ExperimentTracker (comprehensive experiment tracking)"
	@echo "  - ConfigVersioner (configuration version control)"
	@echo "  - ReproducibilityManager (reproducibility guarantees)"
	@echo "  - ExperimentComparator (experiment comparison)"
	@echo ""
	poetry run python scripts/test_experiment.py
	@echo ""
	@echo "$(GREEN)✓ Experiment tracking tests completed!$(NC)"

#==============================================================================
# CI/CD Integration & Automated Workflows (Phase 16)
#==============================================================================

## test-cicd: Test CI/CD workflows and automation
test-cicd:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)Phase 16: Testing CI/CD Integration & Workflows$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  - WorkflowManager (workflow orchestration with dependencies)"
	@echo "  - QualityGateValidator (quality gate validation with thresholds)"
	@echo "  - Pipeline (multi-stage pipeline execution with artifacts)"
	@echo "  - Failure handling and error scenarios"
	@echo ""
	poetry run python scripts/test_cicd.py
	@echo ""
	@echo "$(GREEN)✓ CI/CD integration tests completed!$(NC)"

#==============================================================================
# Model Monitoring & Observability (Phase 17)
#==============================================================================

## test-monitoring: Test model monitoring and observability
test-monitoring:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)Phase 17: Testing Model Monitoring & Observability$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  - PerformanceMonitor (latency, throughput, resource usage)"
	@echo "  - DriftDetector (data drift, prediction drift detection)"
	@echo "  - HealthMonitor (health checks, status monitoring)"
	@echo "  - AlertManager (alerting rules, notifications)"
	@echo "  - Integrated monitoring scenario"
	@echo ""
	poetry run python scripts/test_monitoring.py
	@echo ""
	@echo "$(GREEN)✓ Model monitoring tests completed!$(NC)"

#==============================================================================
# API Serving & REST Endpoints (Phase 18)
#==============================================================================

## test-api: Test REST API endpoints and serving
test-api:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)Phase 18: Testing API Serving & REST Endpoints$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  - Root and info endpoints"
	@echo "  - Health check and readiness endpoints"
	@echo "  - Metrics endpoint (performance, requests)"
	@echo "  - Single prediction endpoint"
	@echo "  - Batch prediction endpoint"
	@echo "  - Request validation and error handling"
	@echo ""
	poetry run python scripts/test_api.py
	@echo ""
	@echo "$(GREEN)✓ API serving tests completed!$(NC)"

#==============================================================================
# Security & Authentication (Phase 19)
#==============================================================================

## test-security: Test security and authentication features
test-security:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)Phase 19: Testing Security & Authentication$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  - API key creation and validation"
	@echo "  - Key expiration and revocation"
	@echo "  - Rate limiting and throttling"
	@echo "  - Endpoint-specific limits"
	@echo "  - Authentication manager"
	@echo "  - Security headers"
	@echo ""
	poetry run python scripts/test_security.py
	@echo ""
	@echo "$(GREEN)✓ Security tests completed!$(NC)"

#==============================================================================
# Model Versioning & Registry (Phase 20)
#==============================================================================

## test-registry: Test model versioning and registry features
test-registry:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)Phase 20: Testing Model Versioning & Registry$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  - Semantic versioning"
	@echo "  - Model version registration and tagging"
	@echo "  - Model metadata tracking and comparison"
	@echo "  - Promotion workflows (dev → staging → production)"
	@echo "  - Approval workflows"
	@echo "  - Model lineage and provenance tracking"
	@echo "  - Lineage graph traversal"
	@echo ""
	poetry run python scripts/test_registry.py
	@echo ""
	@echo "$(GREEN)✓ Model registry tests completed!$(NC)"

test-serving:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)Phase 29: Testing Model Serving & Deployment$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  - ModelLoader (PyTorch loading, device management, caching)"
	@echo "  - Predictor (prediction API, batch inference, confidence extraction)"
	@echo "  - PredictionRequest/Response contracts"
	@echo "  - Monitor integration (performance and prediction monitors)"
	@echo "  - Registry-based model loading"
	@echo ""
	poetry run python scripts/test_serving.py
	@echo ""
	@echo "$(GREEN)✓ Model serving tests completed!$(NC)"

test-deployment:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)Phase 30: Testing Deployment Automation & CI/CD$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  - Deployment configuration and management"
	@echo "  - Deployment strategies (direct, blue-green, canary, rolling)"
	@echo "  - Health check validation"
	@echo "  - Metrics validation (error rates)"
	@echo "  - Rollback mechanisms (manual and automatic)"
	@echo "  - Auto-rollback monitoring"
	@echo "  - Deployment orchestration"
	@echo ""
	poetry run python scripts/test_deployment.py
	@echo ""
	@echo "$(GREEN)✓ Deployment automation tests completed!$(NC)"

test-integration:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)Phase 31: Testing End-to-End Integration$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  - Complete end-to-end workflows"
	@echo "  - Model lifecycle (register → version → deploy)"
	@echo "  - Monitoring integration"
	@echo "  - Registry + Serving + Deployment integration"
	@echo "  - Multi-phase workflows"
	@echo ""
	poetry run python scripts/test_integration.py
	@echo ""
	@echo "$(GREEN)✓ Integration tests completed!$(NC)"

#==============================================================================
# Docker & CI/CD Targets (Phase 31)
#==============================================================================

## docker-build: Build Docker image
docker-build:
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t psy-agents-noaug:latest .
	@echo "$(GREEN)✓ Docker image built successfully$(NC)"

## docker-up: Start services with docker-compose
docker-up:
	@echo "$(BLUE)Starting services with docker-compose...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo ""
	@echo "Access MLflow UI at: http://localhost:5000"

## docker-down: Stop docker-compose services
docker-down:
	@echo "$(BLUE)Stopping docker-compose services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

## docker-test: Run tests in Docker container
docker-test:
	@echo "$(BLUE)Running tests in Docker container...$(NC)"
	docker-compose run --rm test
	@echo "$(GREEN)✓ Docker tests completed$(NC)"

## docker-clean: Clean Docker images and containers
docker-clean:
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	docker-compose down -v
	docker rmi psy-agents-noaug:latest || true
	@echo "$(GREEN)✓ Docker cleanup completed$(NC)"

#==============================================================================
# A/B Testing & Experimentation (Phase 21)
#==============================================================================

## test-ab: Test A/B testing and experimentation framework
test-ab:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)Phase 21: Testing A/B Testing & Experimentation$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  - Traffic splitting strategies (uniform, weighted, sticky)"
	@echo "  - Experiment lifecycle management"
	@echo "  - Statistical significance testing (t-test)"
	@echo "  - Bayesian analysis"
	@echo "  - Experiment tracking and metrics"
	@echo "  - Metric aggregation and conversion rates"
	@echo "  - Sample size calculation"
	@echo ""
	poetry run python scripts/test_ab_testing.py
	@echo ""
	@echo "$(GREEN)✓ A/B testing tests completed!$(NC)"

## test-interpretability: Test model interpretability and explainability framework
test-interpretability:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)Phase 22: Testing Model Interpretability & Explainability$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  - SHAP explanations (gradient-based approximation)"
	@echo "  - Attention visualization (transformer attention weights)"
	@echo "  - Feature importance (gradient, integrated gradients, ablation)"
	@echo "  - Importance tracking and aggregation"
	@echo "  - Model explainer (unified interface)"
	@echo "  - Counterfactual generation"
	@echo "  - Batch explanations"
	@echo ""
	poetry run python scripts/test_interpretability.py
	@echo ""
	@echo "$(GREEN)✓ Interpretability tests completed!$(NC)"

## test-data-quality: Test data quality and drift detection framework
test-data-quality:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)Phase 23: Testing Data Quality & Drift Detection$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  - Drift detection (KS test, PSI, Jensen-Shannon)"
	@echo "  - Data validation (type, range, not-null, unique)"
	@echo "  - Quality metrics (completeness, validity, consistency)"
	@echo "  - Quality reporting (multi-feature analysis)"
	@echo "  - Anomaly detection (IQR, Z-score, Isolation Forest)"
	@echo ""
	poetry run python scripts/test_data_quality.py
	@echo ""
	@echo "$(GREEN)✓ Data quality tests completed!$(NC)"

## test-feature-store: Test feature store and engineering framework
test-feature-store:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)Phase 24: Testing Feature Store & Engineering$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  - Feature registry (registration, groups, search)"
	@echo "  - Feature versioning (lifecycle, lineage)"
	@echo "  - Feature computation (caching, dependencies)"
	@echo "  - Feature serving (online, batch, vectors)"
	@echo ""
	poetry run python scripts/test_feature_store.py
	@echo ""
	@echo "$(GREEN)✓ Feature store tests completed!$(NC)"

## test-governance: Test model governance and compliance framework
test-governance:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)Phase 25: Testing Model Governance & Compliance$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  - Model cards (creation, documentation, export)"
	@echo "  - Bias detection (demographic parity, equal opportunity, disparate impact)"
	@echo "  - Compliance tracking (GDPR, HIPAA, CCPA)"
	@echo "  - Audit trails (logging, lineage, reporting)"
	@echo ""
	poetry run python scripts/test_governance.py
	@echo ""
	@echo "$(GREEN)✓ Governance tests completed!$(NC)"

#==============================================================================
# SUPERMAX Multi-Stage HPO (Stage-A/B/C)
#==============================================================================

## stage-a: SUPERMAX Stage-A baseline exploration (default: 1000 trials)
stage-a:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)SUPERMAX Stage-A: Baseline Exploration$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(YELLOW)Agent: $(AGENT)$(NC)"
	@echo "$(YELLOW)Trials: $${STAGE_A_TRIALS:-1000}$(NC)"
	@echo "$(YELLOW)Epochs: $${STAGE_A_EPOCHS:-6}$(NC)"
	@echo "$(YELLOW)Top-K: $${STAGE_A_TOPK:-50}$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	poetry run python scripts/run_stage_a.py \
		--agent $(AGENT) \
		--trials $${STAGE_A_TRIALS:-1000} \
		--epochs $${STAGE_A_EPOCHS:-6} \
		--patience $${HPO_PATIENCE:-2} \
		--max-samples $${HPO_MAX_SAMPLES:-512} \
		--topk $${STAGE_A_TOPK:-50} \
		--outdir outputs/supermax/stage_a/$(AGENT)

## stage-a-test: SUPERMAX Stage-A smoke test (20 trials, 2 epochs)
stage-a-test:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)SUPERMAX Stage-A: Smoke Test (20 trials)$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	STAGE_A_TRIALS=20 STAGE_A_EPOCHS=2 $(MAKE) stage-a AGENT=$(AGENT)

## stage-a-all: Run Stage-A for all 4 agents
stage-a-all:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)SUPERMAX Stage-A: ALL AGENTS$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	$(MAKE) stage-a AGENT=criteria
	$(MAKE) stage-a AGENT=evidence
	$(MAKE) stage-a AGENT=share
	$(MAKE) stage-a AGENT=joint
	@echo "$(GREEN)✓ Stage-A complete for all agents$(NC)"

## stage-b: SUPERMAX Stage-B multi-objective optimization (default: 1500 trials)
stage-b:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)SUPERMAX Stage-B: Multi-objective Optimization (NSGA-II)$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(YELLOW)Agent: $(AGENT)$(NC)"
	@echo "$(YELLOW)Trials: $${STAGE_B_TRIALS:-1500}$(NC)"
	@echo "$(YELLOW)Epochs: $${STAGE_B_EPOCHS:-10}$(NC)"
	@echo "$(YELLOW)Stage-A input: $${STAGE_A_JSON}$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@if [ -z "$$STAGE_A_JSON" ]; then \
		echo "$(RED)Error: STAGE_A_JSON not set$(NC)"; \
		echo "$(YELLOW)Usage: make stage-b AGENT=criteria STAGE_A_JSON=path/to/stage_a_top50.json$(NC)"; \
		exit 1; \
	fi
	poetry run python scripts/run_stage_b.py \
		--agent $(AGENT) \
		--stage-a-results $$STAGE_A_JSON \
		--trials $${STAGE_B_TRIALS:-1500} \
		--epochs $${STAGE_B_EPOCHS:-10} \
		--patience $${HPO_PATIENCE:-3} \
		--max-samples $${HPO_MAX_SAMPLES:-512} \
		--outdir outputs/supermax/stage_b/$(AGENT)

## stage-b-test: SUPERMAX Stage-B smoke test (20 trials, 3 epochs)
stage-b-test:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)SUPERMAX Stage-B: Smoke Test (20 trials)$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	STAGE_B_TRIALS=20 STAGE_B_EPOCHS=3 $(MAKE) stage-b AGENT=$(AGENT) STAGE_A_JSON=$(STAGE_A_JSON)

## stage-c: SUPERMAX Stage-C K-fold cross-validation (default: 5-fold)
stage-c:
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(BLUE)SUPERMAX Stage-C: K-fold Cross-Validation Refinement$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@echo "$(YELLOW)Agent: $(AGENT)$(NC)"
	@echo "$(YELLOW)K-folds: $${STAGE_C_KFOLDS:-5}$(NC)"
	@echo "$(YELLOW)Epochs: $${STAGE_C_EPOCHS:-15}$(NC)"
	@echo "$(YELLOW)Stage-B input: $${STAGE_B_JSON}$(NC)"
	@echo "$(BLUE)===========================================================$(NC)"
	@if [ -z "$$STAGE_B_JSON" ]; then \
		echo "$(RED)Error: STAGE_B_JSON not set$(NC)"; \
		echo "$(YELLOW)Usage: make stage-c AGENT=criteria STAGE_B_JSON=path/to/stage_b_pareto.json$(NC)"; \
		exit 1; \
	fi
	poetry run python scripts/run_stage_c.py \
		--agent $(AGENT) \
		--stage-b-results $$STAGE_B_JSON \
		--k-folds $${STAGE_C_KFOLDS:-5} \
		--epochs $${STAGE_C_EPOCHS:-15} \
		--patience $${HPO_PATIENCE:-4} \
		--max-samples $${HPO_MAX_SAMPLES:-0} \
		--outdir outputs/supermax/stage_c/$(AGENT)

## stage-c-test: SUPERMAX Stage-C smoke test (2-fold, 3 epochs)
stage-c-test:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Stage-C: Smoke Test (2-fold)$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	STAGE_C_KFOLDS=2 STAGE_C_EPOCHS=3 $(MAKE) stage-c AGENT=$(AGENT) STAGE_B_JSON=$(STAGE_B_JSON)

## full-supermax: Run complete Stage-A→B→C pipeline
full-supermax:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Full Pipeline: Stage-A → Stage-B → Stage-C$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(YELLOW)Running Stage-A (baseline exploration)...$(NC)"
	$(MAKE) stage-a AGENT=$(AGENT)
	@echo "$(YELLOW)Running Stage-B (multi-objective)...$(NC)"
	STAGE_A_JSON=outputs/supermax/stage_a/$(AGENT)/$(AGENT)_stage_a_top50.json \
		$(MAKE) stage-b AGENT=$(AGENT)
	@echo "$(YELLOW)Running Stage-C (K-fold CV)...$(NC)"
	STAGE_B_JSON=outputs/supermax/stage_b/$(AGENT)/$(AGENT)_stage_b_pareto.json \
		$(MAKE) stage-c AGENT=$(AGENT)
	@echo "$(GREEN)✓ SUPERMAX pipeline complete for $(AGENT)!$(NC)"

## build-ensemble: Build ensemble from Stage-B/C results (Phase 6)
build-ensemble:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 6: Ensemble Selection$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(YELLOW)Agent: $(AGENT)$(NC)"
	@echo "$(YELLOW)Input: $${ENSEMBLE_INPUT}$(NC)"
	@echo "$(YELLOW)Type: $${ENSEMBLE_TYPE:-pareto}$(NC)"
	@echo "$(YELLOW)N models: $${ENSEMBLE_N_MODELS:-5}$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	@if [ -z "$$ENSEMBLE_INPUT" ]; then \
		echo "$(RED)Error: ENSEMBLE_INPUT not set$(NC)"; \
		echo "$(YELLOW)Usage: make build-ensemble AGENT=<agent> ENSEMBLE_INPUT=<path> [ENSEMBLE_TYPE=pareto|cv]$(NC)"; \
		exit 1; \
	fi
	poetry run python scripts/build_ensemble.py \
		--agent $(AGENT) \
		--input-file $$ENSEMBLE_INPUT \
		--input-type $${ENSEMBLE_TYPE:-pareto} \
		--n-models $${ENSEMBLE_N_MODELS:-5} \
		--selection-strategy $${ENSEMBLE_STRATEGY:-hybrid} \
		--diversity-weight $${ENSEMBLE_DIV_WEIGHT:-0.3} \
		$${ENSEMBLE_SIMULATE:+--simulate} \
		--outdir outputs/supermax/ensemble/$(AGENT)

## build-ensemble-test: Test ensemble building with simulated predictions
build-ensemble-test:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 6: Ensemble Test (Simulated)$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	ENSEMBLE_SIMULATE=1 ENSEMBLE_N_MODELS=2 $(MAKE) build-ensemble AGENT=$(AGENT) ENSEMBLE_INPUT=$(ENSEMBLE_INPUT) ENSEMBLE_TYPE=$(ENSEMBLE_TYPE)

## monitor-hpo: Monitor running HPO study in real-time (Phase 7)
monitor-hpo:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 7: HPO Monitoring$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	@if [ -z "$$HPO_STUDY_NAME" ]; then \
		echo "$(RED)Error: HPO_STUDY_NAME not set$(NC)"; \
		echo "$(YELLOW)Usage: make monitor-hpo HPO_STUDY_NAME=<study> HPO_STORAGE=<storage>$(NC)"; \
		echo "$(YELLOW)Or: make list-hpo-studies HPO_STORAGE=<storage>$(NC)"; \
		exit 1; \
	fi
	@if [ -z "$$HPO_STORAGE" ]; then \
		echo "$(RED)Error: HPO_STORAGE not set$(NC)"; \
		exit 1; \
	fi
	poetry run python scripts/monitor_hpo.py \
		--study-name $$HPO_STUDY_NAME \
		--storage $$HPO_STORAGE \
		--update-interval $${HPO_UPDATE_INTERVAL:-10} \
		--checkpoint-dir $${HPO_CHECKPOINT_DIR:-outputs/supermax/checkpoints} \
		--log-dir $${HPO_LOG_DIR:-outputs/supermax/logs}

## list-hpo-studies: List all HPO studies in database
list-hpo-studies:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX: List HPO Studies$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	@if [ -z "$$HPO_STORAGE" ]; then \
		echo "$(RED)Error: HPO_STORAGE not set$(NC)"; \
		echo "$(YELLOW)Usage: make list-hpo-studies HPO_STORAGE=<storage>$(NC)"; \
		exit 1; \
	fi
	poetry run python scripts/monitor_hpo.py \
		--storage $$HPO_STORAGE \
		--list-studies

## check-hpo-health: Check health of HPO study
check-hpo-health:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX: HPO Health Check$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	@if [ -z "$$HPO_STUDY_NAME" ] || [ -z "$$HPO_STORAGE" ]; then \
		echo "$(RED)Error: HPO_STUDY_NAME and HPO_STORAGE required$(NC)"; \
		exit 1; \
	fi
	poetry run python scripts/monitor_hpo.py \
		--study-name $$HPO_STUDY_NAME \
		--storage $$HPO_STORAGE \
		--check-health

#==============================================================================
# SUPERMAX Phase 8: Advanced Pruners & Search Strategies
#==============================================================================

## compare-pruners: Compare pruning strategies on benchmark (Phase 8)
compare-pruners:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 8: Pruner Comparison$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	poetry run python scripts/compare_pruners.py \
		--benchmark $${PRUNER_BENCHMARK:-ml_convergence} \
		--n-trials $${PRUNER_N_TRIALS:-50} \
		--n-steps $${PRUNER_N_STEPS:-20} \
		--output $${PRUNER_OUTPUT:-outputs/pruner_comparison.json}
	@echo "$(GREEN)✓ Pruner comparison complete!$(NC)"

## compare-pruners-test: Quick pruner comparison test
compare-pruners-test:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 8: Quick Pruner Test$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	PRUNER_N_TRIALS=20 PRUNER_N_STEPS=10 PRUNER_OUTPUT=outputs/pruner_test.json $(MAKE) compare-pruners

## show-pruner-budget: Show resource budget estimation
show-pruner-budget:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 8: Resource Budget Estimation$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	poetry run python scripts/compare_pruners.py --show-budget

## show-pruner-recommendations: Show pruner recommendations
show-pruner-recommendations:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 8: Pruner Recommendations$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	poetry run python scripts/compare_pruners.py --show-recommendation

#==============================================================================
# SUPERMAX Phase 9: BOHB & Advanced Samplers
#==============================================================================

## compare-samplers: Compare sampling strategies on benchmark (Phase 9)
compare-samplers:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 9: Sampler Comparison$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	poetry run python scripts/compare_samplers.py \
		--benchmark $${SAMPLER_BENCHMARK:-sphere} \
		--n-trials $${SAMPLER_N_TRIALS:-100} \
		--output $${SAMPLER_OUTPUT:-outputs/sampler_comparison.json}
	@echo "$(GREEN)✓ Sampler comparison complete!$(NC)"

## compare-samplers-test: Quick sampler comparison test
compare-samplers-test:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 9: Quick Sampler Test$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	SAMPLER_N_TRIALS=50 SAMPLER_OUTPUT=outputs/sampler_test.json $(MAKE) compare-samplers

## show-sampler-recommendations: Show sampler recommendations
show-sampler-recommendations:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 9: Sampler Recommendations$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	poetry run python scripts/compare_samplers.py --show-recommendations

#==============================================================================
# SUPERMAX Phase 10: Meta-Learning & Warm-Starting
#==============================================================================

## test-meta-learning: Test meta-learning functionality
test-meta-learning:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 10: Testing Meta-Learning$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	poetry run python scripts/test_meta_learning.py

## analyze-hpo: Analyze a completed HPO study
analyze-hpo:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 10: HPO Study Analysis$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	@if [ -z "$(STUDY)" ]; then \
		echo "$(RED)Error: STUDY not specified$(NC)"; \
		echo "Usage: make analyze-hpo STUDY=criteria-maximal-hpo"; \
		exit 1; \
	fi
	poetry run python scripts/analyze_hpo.py \
		--study $(STUDY) \
		--storage $(OPTUNA_STORAGE) \
		--detailed

## compare-hpo-studies: Compare multiple HPO studies
compare-hpo-studies:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 10: Compare HPO Studies$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	@if [ -z "$(STUDIES)" ]; then \
		echo "$(RED)Error: STUDIES not specified$(NC)"; \
		echo "Usage: make compare-hpo-studies STUDIES='criteria-max evidence-max'"; \
		exit 1; \
	fi
	poetry run python scripts/analyze_hpo.py \
		--studies $(STUDIES) \
		--storage $(OPTUNA_STORAGE) \
		--compare \
		--analyze-convergence

## show-transfer-recommendations: Show transfer learning recommendations
show-transfer-recommendations:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 10: Transfer Learning Recommendations$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	@if [ -z "$(TARGET_TASK)" ]; then \
		echo "$(RED)Error: TARGET_TASK not specified$(NC)"; \
		echo "Usage: make show-transfer-recommendations TARGET_TASK=joint AVAILABLE='criteria-max evidence-max'"; \
		exit 1; \
	fi
	poetry run python scripts/analyze_hpo.py \
		--transfer-to $(TARGET_TASK) \
		--available-studies $(AVAILABLE)

## warm-start-hpo: Create warm-started HPO study
warm-start-hpo:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 10: Warm-Start HPO Study$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	@if [ -z "$(NEW_STUDY)" ] || [ -z "$(SOURCE_STUDY)" ]; then \
		echo "$(RED)Error: NEW_STUDY or SOURCE_STUDY not specified$(NC)"; \
		echo "Usage: make warm-start-hpo NEW_STUDY=evidence-ws SOURCE_STUDY=criteria-max"; \
		exit 1; \
	fi
	poetry run python scripts/warm_start_hpo.py \
		--new-study $(NEW_STUDY) \
		--source-study $(SOURCE_STUDY) \
		--storage $(OPTUNA_STORAGE) \
		--n-configs $(or $(N_CONFIGS),10) \
		--strategy $(or $(STRATEGY),best)

## transfer-hpo: Create study with transfer learning
transfer-hpo:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 10: Transfer Learning HPO$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	@if [ -z "$(NEW_STUDY)" ] || [ -z "$(SOURCE_STUDY)" ] || [ -z "$(SOURCE_TASK)" ] || [ -z "$(TARGET_TASK)" ]; then \
		echo "$(RED)Error: Required parameters not specified$(NC)"; \
		echo "Usage: make transfer-hpo NEW_STUDY=evidence-tl SOURCE_STUDY=criteria-max SOURCE_TASK=criteria TARGET_TASK=evidence"; \
		exit 1; \
	fi
	poetry run python scripts/warm_start_hpo.py \
		--new-study $(NEW_STUDY) \
		--source-study $(SOURCE_STUDY) \
		--source-task $(SOURCE_TASK) \
		--target-task $(TARGET_TASK) \
		--storage $(OPTUNA_STORAGE) \
		--n-configs $(or $(N_CONFIGS),10)

#==============================================================================
# SUPERMAX Phase 11: Ensemble Methods & Model Selection
#==============================================================================

## test-ensemble: Test ensemble methods functionality
test-ensemble:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 11: Testing Ensemble Methods$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	poetry run python scripts/test_ensemble.py

#==============================================================================
# SUPERMAX Phase 12: Distributed HPO & Parallel Execution
#==============================================================================

## test-distributed: Test distributed HPO functionality
test-distributed:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 12: Testing Distributed HPO$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	poetry run python scripts/test_distributed.py

## run-distributed-hpo: Run distributed HPO with parallel execution
## Usage: make run-distributed-hpo AGENT=criteria N_TRIALS=100 N_WORKERS=4 GPU_IDS="0,1,2,3"
run-distributed-hpo:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 12: Distributed HPO$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "Agent: $(AGENT)"
	@echo "Trials: $(N_TRIALS)"
	@echo "Workers: $(N_WORKERS)"
	@if [ -n "$(GPU_IDS)" ]; then \
		echo "GPUs: $(GPU_IDS)"; \
		poetry run python scripts/run_distributed_hpo.py \
			--agent $(AGENT) \
			--n-trials $(N_TRIALS) \
			--n-workers $(N_WORKERS) \
			--gpu-ids $(GPU_IDS); \
	else \
		echo "GPUs: CPU only"; \
		poetry run python scripts/run_distributed_hpo.py \
			--agent $(AGENT) \
			--n-trials $(N_TRIALS) \
			--n-workers $(N_WORKERS); \
	fi

## check-gpu-availability: Check available GPUs
check-gpu-availability:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 12: GPU Availability Check$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	@poetry run python -c "from psy_agents_noaug.hpo.distributed import check_gpu_availability; import json; info = check_gpu_availability(); print(json.dumps(info, indent=2))"

#==============================================================================
# SUPERMAX Phase 13: Results Analysis & Visualization
#==============================================================================

## test-visualization: Test visualization functionality
test-visualization:
	@echo "$(BLUE)===========================================================$ $(NC)"
	@echo "$(BLUE)SUPERMAX Phase 13: Testing Visualization$(NC)"
	@echo "$(BLUE)===========================================================$ $(NC)"
	poetry run python scripts/test_visualization.py

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

## typecheck: Run mypy type checking
typecheck:
	@echo "$(BLUE)Running mypy type checker...$(NC)"
	poetry run mypy src/psy_agents_noaug/
	@echo "$(GREEN)✓ Type checking complete$(NC)"

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
	poetry run pre-commit install --hook-type commit-msg
	@echo "$(GREEN)✓ Pre-commit hooks installed$(NC)"

## pre-commit-run: Run pre-commit on all files
pre-commit-run:
	@echo "$(BLUE)Running pre-commit on all files...$(NC)"
	poetry run pre-commit run --all-files

## pre-commit-update: Update pre-commit hook versions
pre-commit-update:
	@echo "$(BLUE)Updating pre-commit hook versions...$(NC)"
	poetry run pre-commit autoupdate
	@echo "$(GREEN)✓ Pre-commit hooks updated$(NC)"

#==============================================================================
# Security & Compliance
#==============================================================================

## audit: Run security vulnerability scan
audit:
	@echo "$(BLUE)Running security audit...$(NC)"
	poetry run python scripts/audit_security.py --severity medium

## audit-strict: Run security audit and fail on critical vulnerabilities
audit-strict:
	@echo "$(BLUE)Running strict security audit...$(NC)"
	poetry run python scripts/audit_security.py --severity critical --fail-on-critical

## sbom: Generate Software Bill of Materials
sbom:
	@echo "$(BLUE)Generating SBOM...$(NC)"
	poetry run python scripts/generate_sbom.py --format json --output sbom.json --with-metadata

## licenses: Generate third-party license report
licenses:
	@echo "$(BLUE)Generating license report...$(NC)"
	poetry run python scripts/generate_licenses.py --format markdown --output THIRD_PARTY_LICENSES.md

## compliance: Run all compliance checks (audit + sbom + licenses)
compliance: audit sbom licenses
	@echo "$(GREEN)✓ All compliance checks complete$(NC)"

#==============================================================================
# Performance & Benchmarking
#==============================================================================

## bench: Run performance benchmarks
bench:
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	poetry run python scripts/bench_dataloader.py --num-batches 50 --output benchmark_results.json

## verify-determinism: Verify deterministic behavior
verify-determinism:
	@echo "$(BLUE)Verifying determinism...$(NC)"
	poetry run python scripts/verify_determinism.py

## train-with-json-logs: Train with JSON-structured logging
train-with-json-logs:
	@echo "$(BLUE)Training with JSON-structured logging...$(NC)"
	LOG_JSON=true poetry run python -m psy_agents_noaug.cli train

#==============================================================================
# Retrain Best Evidence (from Optuna DB)
#==============================================================================

## retrain-evidence-aug: Retrain best evidence trial and save config/checkpoint/metrics
retrain-evidence-aug:
	@echo "$(BLUE)Retraining best Evidence model (with augmentation settings)...$(NC)"
	RETRAIN_EPOCHS=$${RETRAIN_EPOCHS:-6} NUM_WORKERS=$${NUM_WORKERS:-8} \
	poetry run python scripts/retrain_best_evidence.py \
		--study $${EVIDENCE_STUDY:-aug-evidence-production-2025-10-27} \
		--storage $${OPTUNA_STORAGE:-sqlite:///_optuna/dataaug.db} \
		--outdir outputs/retrain/evidence_aug

## retrain-evidence-noaug: Retrain best evidence trial with augmentation disabled
retrain-evidence-noaug:
	@echo "$(BLUE)Retraining best Evidence model (augmentation disabled)...$(NC)"
	RETRAIN_EPOCHS=$${RETRAIN_EPOCHS:-6} NUM_WORKERS=$${NUM_WORKERS:-8} \
	poetry run python scripts/retrain_best_evidence.py \
		--study $${EVIDENCE_STUDY:-aug-evidence-production-2025-10-27} \
		--storage $${OPTUNA_STORAGE:-sqlite:///_optuna/dataaug.db} \
		--outdir outputs/retrain/evidence_noaug \
		--disable-augmentation

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

## full-hpo-all: Run complete multi-stage HPO pipeline for selected architectures
full-hpo-all:
	@agents="$(if $(strip $(AGENTS)),$(AGENTS),criteria evidence share joint)"; \
	for agent in $$agents; do \
		echo "$(BLUE)> Multi-stage HPO $$agent$(NC)"; \
		poetry run psy-agents hpo-stage --agent $$agent --seeds "$(if $(strip $(HPO_SEEDS)),$(HPO_SEEDS),1)"; \
	done

## maximal-hpo-all: Run maximal single-stage HPO for selected architectures
maximal-hpo-all:
	@agents="$(if $(strip $(AGENTS)),$(AGENTS),criteria evidence share joint)"; \
	for agent in $$agents; do \
		echo "$(BLUE)> Maximal HPO $$agent$(NC)"; \
		poetry run psy-agents hpo-max --agent $$agent --seeds "$(if $(strip $(HPO_SEEDS)),$(HPO_SEEDS),1)" --trials $(if $(strip $(HPO_TRIALS)),$(HPO_TRIALS),100) --epochs $(if $(strip $(HPO_EPOCHS)),$(HPO_EPOCHS),6); \
	done

## show-best-%: Display top trials for the given agent
show-best-%:
	@poetry run psy-agents show-best --agent $* --study $(if $(strip $(HPO_PROFILE)),$(HPO_PROFILE),noaug)-$*-max --topk $(if $(strip $(TOPK)),$(TOPK),5)

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
	@HPO_EPOCHS?=100; HPO_PATIENCE?=20; \
	HPO_EPOCHS=$$HPO_EPOCHS HPO_PATIENCE=$$HPO_PATIENCE \
	python scripts/tune_max.py --agent criteria --study noaug-criteria-max --n-trials 800 --parallel 4 --outdir $${HPO_OUTDIR:-./_runs}

tune-evidence-max:
	@HPO_EPOCHS?=100; HPO_PATIENCE?=20; \
	HPO_EPOCHS=$$HPO_EPOCHS HPO_PATIENCE=$$HPO_PATIENCE \
	python scripts/tune_max.py --agent evidence --stage A --study noaug-evidence-max --n-trials 1200 --parallel 4 --outdir $${HPO_OUTDIR:-./_runs}

tune-evidence-aug:
	@if [ -z "$${FROM_STUDY}" ]; then \
		echo "FROM_STUDY must point to the Stage-A study (e.g., noaug-evidence-max)"; \
		exit 1; \
	fi
	@HPO_EPOCHS?=60; HPO_PATIENCE?=15; \
	HPO_EPOCHS=$$HPO_EPOCHS HPO_PATIENCE=$$HPO_PATIENCE \
	python scripts/tune_max.py --agent evidence --stage B \
		--study $${STAGE_B_STUDY:-aug-evidence-ext} \
		--from-study $${FROM_STUDY} \
		--n-trials $${N_TRIALS_STAGE_B:-600} \
		--parallel $${PAR_STAGE_B:-4} \
		--outdir $${HPO_OUTDIR:-./_runs}

tune-evidence-joint:
	@if [ -z "$${FROM_STUDY}" ]; then \
		echo "FROM_STUDY must point to the Stage-B study (e.g., aug-evidence-ext)"; \
		exit 1; \
	fi
	@HPO_EPOCHS?=80; HPO_PATIENCE?=20; \
	HPO_EPOCHS=$$HPO_EPOCHS HPO_PATIENCE=$$HPO_PATIENCE \
	python scripts/tune_max.py --agent evidence --stage C \
		--study $${STAGE_C_STUDY:-aug-evidence-joint} \
		--from-study $${FROM_STUDY} \
		--pareto-limit $${PARETO_LIMIT:-5} \
		--n-trials $${N_TRIALS_STAGE_C:-240} \
		--parallel $${PAR_STAGE_C:-2} \
		--outdir $${HPO_OUTDIR:-./_runs}

tune-share-max:
	@HPO_EPOCHS?=100; HPO_PATIENCE?=20; \
	HPO_EPOCHS=$$HPO_EPOCHS HPO_PATIENCE=$$HPO_PATIENCE \
	python scripts/tune_max.py --agent share --study noaug-share-max --n-trials 600 --parallel 4 --outdir $${HPO_OUTDIR:-./_runs}

tune-joint-max:
	@HPO_EPOCHS?=100; HPO_PATIENCE?=20; \
	HPO_EPOCHS=$$HPO_EPOCHS HPO_PATIENCE=$$HPO_PATIENCE \
	python scripts/tune_max.py --agent joint --study noaug-joint-max --n-trials 600 --parallel 4 --outdir $${HPO_OUTDIR:-./_runs}

#==============================================================================
# 3-Stage HPO Workflows (Convenience Targets)
#==============================================================================

## tune-evidence-3stage-full: Run full 3-stage Evidence HPO (A→B→C) automatically
tune-evidence-3stage-full:
	@echo "$(BLUE)=====================================================================$(NC)"
	@echo "$(BLUE)Starting 3-Stage Evidence HPO Workflow$(NC)"
	@echo "$(BLUE)Stage A: Baseline (1200 trials) → Stage B: Aug (600 trials) → Stage C: Joint (240 trials)$(NC)"
	@echo "$(BLUE)=====================================================================$(NC)"
	@echo ""
	@echo "$(GREEN)[Stage A] Running baseline HPO (no augmentation)...$(NC)"
	@$(MAKE) tune-evidence-max
	@echo ""
	@echo "$(GREEN)[Stage B] Running augmentation search...$(NC)"
	@FROM_STUDY=noaug-evidence-max $(MAKE) tune-evidence-aug
	@echo ""
	@echo "$(GREEN)[Stage C] Running joint refinement...$(NC)"
	@FROM_STUDY=aug-evidence-ext $(MAKE) tune-evidence-joint
	@echo ""
	@echo "$(GREEN)✓ 3-Stage Evidence HPO Complete!$(NC)"
	@echo "$(YELLOW)Results:$(NC)"
	@echo "  Stage A (Baseline): noaug-evidence-max"
	@echo "  Stage B (Aug): aug-evidence-ext"
	@echo "  Stage C (Joint): aug-evidence-joint"

## tune-evidence-3stage-smoke: Run smoke test 3-stage HPO (reduced trials for testing)
tune-evidence-3stage-smoke:
	@echo "$(BLUE)=====================================================================$(NC)"
	@echo "$(BLUE)Starting 3-Stage Evidence HPO Smoke Test$(NC)"
	@echo "$(BLUE)Stage A: 10 trials → Stage B: 12 trials → Stage C: 8 trials$(NC)"
	@echo "$(BLUE)=====================================================================$(NC)"
	@echo ""
	@echo "$(GREEN)[Stage A] Smoke test: baseline (10 trials, 3 epochs)...$(NC)"
	@HPO_EPOCHS=3 python scripts/tune_max.py --agent evidence --stage A \
		--study smoke-evidence-baseline --n-trials 10 --parallel 2 \
		--outdir $${HPO_OUTDIR:-./_runs}
	@echo ""
	@echo "$(GREEN)[Stage B] Smoke test: augmentation search (12 trials, 3 epochs)...$(NC)"
	@HPO_EPOCHS=3 python scripts/tune_max.py --agent evidence --stage B \
		--study smoke-evidence-aug --from-study smoke-evidence-baseline \
		--n-trials 12 --parallel 2 --outdir $${HPO_OUTDIR:-./_runs}
	@echo ""
	@echo "$(GREEN)[Stage C] Smoke test: joint refinement (8 trials, 3 epochs)...$(NC)"
	@HPO_EPOCHS=3 python scripts/tune_max.py --agent evidence --stage C \
		--study smoke-evidence-joint --from-study smoke-evidence-aug \
		--pareto-limit 3 --n-trials 8 --parallel 2 --outdir $${HPO_OUTDIR:-./_runs}
	@echo ""
	@echo "$(GREEN)✓ 3-Stage Evidence Smoke Test Complete!$(NC)"
	@echo "$(YELLOW)Results:$(NC)"
	@echo "  Stage A: smoke-evidence-baseline (10 trials)"
	@echo "  Stage B: smoke-evidence-aug (12 trials)"
	@echo "  Stage C: smoke-evidence-joint (8 trials)"

## tune-criteria-3stage-full: Run full 3-stage Criteria HPO (A→B→C) automatically
tune-criteria-3stage-full:
	@echo "$(BLUE)=====================================================================$(NC)"
	@echo "$(BLUE)Starting 3-Stage Criteria HPO Workflow$(NC)"
	@echo "$(BLUE)Stage A: Baseline (800 trials) → Stage B: Aug (400 trials) → Stage C: Joint (160 trials)$(NC)"
	@echo "$(BLUE)=====================================================================$(NC)"
	@echo ""
	@echo "$(GREEN)[Stage A] Running baseline HPO (no augmentation)...$(NC)"
	@$(MAKE) tune-criteria-max
	@echo ""
	@echo "$(GREEN)[Stage B] Running augmentation search...$(NC)"
	@FROM_STUDY=noaug-criteria-max HPO_EPOCHS=60 N_TRIALS_STAGE_B=400 \
		python scripts/tune_max.py --agent criteria --stage B \
		--study aug-criteria-ext --from-study noaug-criteria-max \
		--n-trials 400 --parallel 4 --outdir $${HPO_OUTDIR:-./_runs}
	@echo ""
	@echo "$(GREEN)[Stage C] Running joint refinement...$(NC)"
	@FROM_STUDY=aug-criteria-ext HPO_EPOCHS=80 N_TRIALS_STAGE_C=160 \
		python scripts/tune_max.py --agent criteria --stage C \
		--study aug-criteria-joint --from-study aug-criteria-ext \
		--pareto-limit 5 --n-trials 160 --parallel 2 --outdir $${HPO_OUTDIR:-./_runs}
	@echo ""
	@echo "$(GREEN)✓ 3-Stage Criteria HPO Complete!$(NC)"

#==============================================================================
# Super-Max HPO (100 epochs + EarlyStopping, very high trial counts)
#==============================================================================

# Override trial counts and parallelism via environment:
#   N_TRIALS_CRITERIA=6000 PAR=6 make tune-criteria-supermax
PAR ?= 4
N_TRIALS_CRITERIA ?= 5000
N_TRIALS_EVIDENCE ?= 8000
N_TRIALS_SHARE ?= 3000
N_TRIALS_JOINT ?= 3000
HPO_OUTDIR ?= ./_runs

## tune-criteria-supermax: Run super-max HPO for criteria (5000 trials, 100 epochs, ES patience=20)
tune-criteria-supermax:
	@echo "$(BLUE)Running SUPER-MAX HPO for Criteria...$(NC)"
	@echo "Trials: $(N_TRIALS_CRITERIA) | Parallel: $(PAR) | Epochs: 100 | Patience: 20"
	HPO_EPOCHS=100 HPO_PATIENCE=20 poetry run python scripts/tune_max.py \
		--agent criteria --study noaug-criteria-supermax \
		--n-trials $(N_TRIALS_CRITERIA) --parallel 1 \
		--outdir $(HPO_OUTDIR)

## tune-evidence-supermax: Run super-max HPO for evidence (8000 trials, 100 epochs, ES patience=20)
tune-evidence-supermax:
	@echo "$(BLUE)Running SUPER-MAX HPO for Evidence...$(NC)"
	@echo "Trials: $(N_TRIALS_EVIDENCE) | Parallel: $(PAR) | Epochs: 100 | Patience: 20"
	HPO_EPOCHS=100 HPO_PATIENCE=20 poetry run python scripts/tune_max.py \
		--agent evidence --study noaug-evidence-supermax \
		--n-trials $(N_TRIALS_EVIDENCE) --parallel 1 \
		--outdir $(HPO_OUTDIR)

## tune-share-supermax: Run super-max HPO for share (3000 trials, 100 epochs, ES patience=20)
tune-share-supermax:
	@echo "$(BLUE)Running SUPER-MAX HPO for Share...$(NC)"
	@echo "Trials: $(N_TRIALS_SHARE) | Parallel: $(PAR) | Epochs: 100 | Patience: 20"
	HPO_EPOCHS=100 HPO_PATIENCE=20 poetry run python scripts/tune_max.py \
		--agent share --study noaug-share-supermax \
		--n-trials $(N_TRIALS_SHARE) --parallel 1 \
		--outdir $(HPO_OUTDIR)

## tune-joint-supermax: Run super-max HPO for joint (3000 trials, 100 epochs, ES patience=20)
tune-joint-supermax:
	@echo "$(BLUE)Running SUPER-MAX HPO for Joint...$(NC)"
	@echo "Trials: $(N_TRIALS_JOINT) | Parallel: $(PAR) | Epochs: 100 | Patience: 20"
	HPO_EPOCHS=100 HPO_PATIENCE=20 poetry run python scripts/tune_max.py \
		--agent joint --study noaug-joint-supermax \
		--n-trials $(N_TRIALS_JOINT) --parallel 1 \
		--outdir $(HPO_OUTDIR)

## tune-all-supermax: Run super-max HPO for ALL architectures sequentially (criteria→evidence→share→joint)
tune-all-supermax:
	@echo "$(BLUE)======================================================================"
	@echo "  Running Super-Max HPO for ALL Architectures Sequentially"
	@echo "======================================================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Sequence: Criteria → Evidence → Share → Joint$(NC)"
	@echo "$(YELLOW)Total trials: ~19,000 (5000+8000+3000+3000)$(NC)"
	@echo "$(YELLOW)Estimated time: ~80-120 hours with PAR=4$(NC)"
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

#==============================================================================
# Docker Targets
#==============================================================================

## docker-build: Build Docker images (multi-stage)
docker-build:
	@echo "$(BLUE)Building Docker images...$(NC)"
	@bash scripts/build_docker.sh

## docker-test: Run tests in Docker container
docker-test:
	@echo "$(BLUE)Running tests in Docker...$(NC)"
	@docker-compose -f docker-compose.test.yml run --rm test

## docker-run: Run production container
docker-run:
	@echo "$(BLUE)Starting production container...$(NC)"
	@docker-compose -f docker-compose.test.yml up runtime

## docker-shell: Open shell in production container
docker-shell:
	@echo "$(BLUE)Opening shell in production container...$(NC)"
	@docker run -it --rm psy-agents-noaug:latest /bin/bash

## docker-mlflow: Start MLflow UI in Docker
docker-mlflow:
	@echo "$(BLUE)Starting MLflow UI...$(NC)"
	@docker-compose -f docker-compose.test.yml --profile mlflow up mlflow

## docker-clean: Clean Docker images and containers
docker-clean:
	@echo "$(BLUE)Cleaning Docker artifacts...$(NC)"
	@docker-compose -f docker-compose.test.yml down -v || true
	@docker rmi psy-agents-noaug:builder psy-agents-noaug:latest psy-agents-noaug:0.1.0 || true
	@echo "$(GREEN)✓ Docker artifacts cleaned$(NC)"
