.PHONY: env-create env-update pip-sync augment-nlpaug augment-textattack augment-hybrid augment-all regenerate-synonyms train train-optuna evaluate train-criteria train-evidence train-joint train-criteria-optuna train-evidence-optuna train-joint-optuna evaluate-criteria evaluate-evidence evaluate-joint data-preview lint format test clean mlflow-ui tensorboard optuna-dashboard dvc-init dvc-status docker-clean docker-up docker-down docker-exec docker-logs docker-train docker-train-optuna docker-test gpu-test help

# Auto-detect environment (mamba or direct python)
PYTHON ?= python
ENV_NAME ?= redsm5
MLFLOW_URI ?= $(if $(MLFLOW_TRACKING_URI),$(MLFLOW_TRACKING_URI),http://localhost:5000)

# Check if mamba is available, otherwise use direct python
MAMBA_EXISTS := $(shell command -v mamba 2> /dev/null)
ifdef MAMBA_EXISTS
    RUN_CMD = mamba run -n $(ENV_NAME)
else
    RUN_CMD = 
endif

# Docker settings
DOCKER_COMPOSE_FILE = .devcontainer/docker-compose.yml
CONTAINER_NAME = redsm5-dev
DOCKER_EXEC = docker exec -i $(CONTAINER_NAME)

env-create:
	mamba env create -f environment.yml

env-update:
	mamba env update -f environment.yml --prune

pip-sync:
	$(RUN_CMD) python -m pip install -r requirements.txt

pip-install:
	pip install -r requirements.txt

augment-nlpaug:
	$(RUN_CMD) $(PYTHON) scripts/generate_nlpaug_dataset.py

augment-textattack:
	$(RUN_CMD) $(PYTHON) scripts/generate_textattack_dataset.py

augment-hybrid:
	$(RUN_CMD) $(PYTHON) scripts/generate_hybrid_dataset.py

augment-all: augment-nlpaug augment-textattack augment-hybrid

regenerate-synonyms:
	$(RUN_CMD) $(PYTHON) scripts/regenerate_augmentation.py

train:
	$(RUN_CMD) $(PYTHON) -m src.training.train

train-optuna:
	$(RUN_CMD) $(PYTHON) -m src.training.train_optuna

evaluate:
	$(RUN_CMD) $(PYTHON) -m src.training.evaluate

# Multi-Agent Training Modes
train-criteria:
	$(RUN_CMD) $(PYTHON) -m src.training.train_criteria training_mode=criteria

train-evidence:
	$(RUN_CMD) $(PYTHON) -m src.training.train_evidence training_mode=evidence

train-joint:
	$(RUN_CMD) $(PYTHON) -m src.training.train_joint training_mode=joint

# Multi-Agent HPO
train-criteria-optuna:
	$(RUN_CMD) $(PYTHON) -m src.training.train_criteria_optuna training_mode=criteria

train-evidence-optuna:
	$(RUN_CMD) $(PYTHON) -m src.training.train_evidence_optuna training_mode=evidence

train-joint-optuna:
	$(RUN_CMD) $(PYTHON) -m src.training.train_joint_optuna training_mode=joint

# Multi-Agent Evaluation
evaluate-criteria:
	$(RUN_CMD) $(PYTHON) -m src.training.evaluate_criteria training_mode=criteria

evaluate-evidence:
	$(RUN_CMD) $(PYTHON) -m src.training.evaluate_evidence training_mode=evidence

evaluate-joint:
	$(RUN_CMD) $(PYTHON) -m src.training.evaluate_joint training_mode=joint

data-preview:
	$(RUN_CMD) $(PYTHON) -c "import pandas as pd; df = pd.read_csv('Data/Augmentation/augmented_positive_pairs.csv'); print(df.head())"

lint:
	$(RUN_CMD) ruff check src tests scripts

format:
	$(RUN_CMD) black src tests scripts

test:
	$(RUN_CMD) pytest --maxfail=1 --disable-warnings -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

# MLflow and Monitoring
mlflow-ui:
	@echo "Opening MLflow UI at http://localhost:5000"
	@echo "Ensure the host MLflow service is running (this project no longer launches it automatically)."

tensorboard:
	@echo "Starting TensorBoard on http://localhost:6006"
	$(RUN_CMD) tensorboard --logdir=logs --host=0.0.0.0 --port=6006

optuna-dashboard:
	@echo "Starting Optuna Dashboard on http://localhost:8080"
	@bash scripts/launch_optuna_dashboard.sh

# DVC Commands
dvc-init:
	$(RUN_CMD) dvc init --no-scm

dvc-status:
	$(RUN_CMD) dvc status

dvc-push:
	$(RUN_CMD) dvc push

dvc-pull:
	$(RUN_CMD) dvc pull

# Docker Commands
docker-clean:
	@echo "Cleaning up existing containers and networks..."
	@docker compose -f $(DOCKER_COMPOSE_FILE) down --remove-orphans 2>/dev/null || true
	@docker rm -f redsm5-dev 2>/dev/null || true
	@echo "✅ Cleanup complete"

docker-up:
	@echo "Starting Docker containers..."
	@echo "Checking for port conflicts..."
	@conflicts=0; \
	for port in 6006 8080 8888; do \
		if lsof -Pi :$$port -sTCP:LISTEN -t >/dev/null 2>&1 || ss -tlnp 2>&1 | grep -q :$$port; then \
			echo "⚠️  Port $$port is already in use!"; \
			conflicts=$$((conflicts + 1)); \
		fi; \
	done; \
	if [ $$conflicts -gt 0 ]; then \
		echo ""; \
		echo "Ports in use: Check with 'docker ps' or 'ss -tlnp | grep <port>'"; \
		echo ""; \
		echo "Solutions:"; \
		echo "  1. Stop conflicting services/containers"; \
		echo "  2. Run 'make docker-clean' to clean up old containers"; \
		echo "  3. Check for Jupyter notebooks: 'jupyter notebook list'"; \
		echo ""; \
		exit 1; \
	fi
	docker compose -f $(DOCKER_COMPOSE_FILE) up -d
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo "✅ Containers started! Access:"
	@echo "   - Dev container: make docker-exec"
	@echo "   - Ensure MLflow server is reachable at ${MLFLOW_URI}"

docker-down:
	@echo "Stopping Docker containers..."
	docker compose -f $(DOCKER_COMPOSE_FILE) down

docker-restart:
	@echo "Restarting Docker containers..."
	docker compose -f $(DOCKER_COMPOSE_FILE) restart

docker-exec:
	@echo "Entering development container..."
	$(DOCKER_EXEC) bash

docker-logs:
	docker compose -f $(DOCKER_COMPOSE_FILE) logs -f

docker-logs-app:
	docker logs -f $(CONTAINER_NAME)

# Docker Training Commands (run inside container from host)
docker-train:
	$(DOCKER_EXEC) bash -c "cd /workspaces/DataAugmentation_ReDSM5 && python -m src.training.train"

docker-train-optuna:
	$(DOCKER_EXEC) bash -c "cd /workspaces/DataAugmentation_ReDSM5 && python -m src.training.train_optuna"

docker-test:
	$(DOCKER_EXEC) bash -c "cd /workspaces/DataAugmentation_ReDSM5 && pytest --maxfail=1 --disable-warnings -v"

# GPU Test
gpu-test:
	$(RUN_CMD) $(PYTHON) test_gpu.py

# Help
help:
	@echo "🚀 DataAugmentation ReDSM-5 - Available Commands"
	@echo ""
	@echo "📦 Environment Setup:"
	@echo "  env-create        - Create mamba environment"
	@echo "  env-update        - Update mamba environment"
	@echo "  pip-sync          - Install Python dependencies (with mamba)"
	@echo "  pip-install       - Install Python dependencies (direct pip)"
	@echo ""
	@echo "🐳 Docker Commands (Recommended for Training):"
	@echo "  docker-clean      - Clean up existing containers (use if docker-up fails)"
	@echo "  docker-up         - Start the development container (MLflow runs externally)"
	@echo "  docker-down       - Stop all containers"
	@echo "  docker-restart    - Restart all containers"
	@echo "  docker-exec       - Enter development container shell"
	@echo "  docker-logs       - View all container logs"
	@echo "  docker-logs-app   - View app container logs only"
	@echo ""
	@echo "🏋️  Training (from Host - runs in Docker):"
	@echo "  docker-train      - Run standard training in container"
	@echo "  docker-train-optuna - Run Optuna HPO in container"
	@echo "  docker-test       - Run tests in container"
	@echo ""
	@echo "🔬 Training (Direct - requires environment):"
	@echo "  train             - Standard training"
	@echo "  train-optuna      - Hyperparameter optimization with Optuna"
	@echo "  evaluate          - Evaluate trained model"
	@echo "  gpu-test          - Test GPU availability"
	@echo ""
	@echo "📊 Data Augmentation:"
	@echo "  augment-nlpaug    - Generate NLPAug dataset"
	@echo "  augment-textattack - Generate TextAttack dataset"
	@echo "  augment-hybrid    - Generate hybrid dataset"
	@echo "  augment-all       - Run all augmentation pipelines"
	@echo ""
	@echo "🤖 Multi-Agent Training:"
	@echo "  train-criteria    - Train criteria matching agent (Mode 1)"
	@echo "  train-evidence    - Train evidence binding agent (Mode 2)"
	@echo "  train-joint       - Train both agents jointly (Mode 3)"
	@echo ""
	@echo "🔍 Multi-Agent HPO:"
	@echo "  train-criteria-optuna - HPO for criteria matching agent"
	@echo "  train-evidence-optuna - HPO for evidence binding agent"
	@echo "  train-joint-optuna    - HPO for joint training"
	@echo ""
	@echo "📈 Multi-Agent Evaluation:"
	@echo "  evaluate-criteria - Evaluate criteria matching agent"
	@echo "  evaluate-evidence - Evaluate evidence binding agent"
	@echo "  evaluate-joint    - Evaluate joint model"
	@echo ""
	@echo "📡 Monitoring:"
	@echo "  mlflow-ui         - Info about MLflow UI (http://localhost:5000)"
	@echo "  tensorboard       - Start TensorBoard"
	@echo "  optuna-dashboard  - Start Optuna Dashboard"
	@echo ""
	@echo "💾 Data Version Control:"
	@echo "  dvc-init          - Initialize DVC"
	@echo "  dvc-status        - Check DVC status"
	@echo "  dvc-push          - Push data to remote"
	@echo "  dvc-pull          - Pull data from remote"
	@echo ""
	@echo "🛠️  Development:"
	@echo "  lint              - Run ruff linter"
	@echo "  format            - Format code with black"
	@echo "  test              - Run pytest tests"
	@echo "  clean             - Remove cache files"
	@echo ""
	@echo "💡 Quick Start:"
	@echo "  1. make docker-up          # Start containers"
	@echo "  2. make docker-train       # Run training from host"
	@echo "     OR"
	@echo "  1. make docker-up          # Start containers"
	@echo "  2. make docker-exec        # Enter container"
	@echo "  3. make train              # Run training inside container"
