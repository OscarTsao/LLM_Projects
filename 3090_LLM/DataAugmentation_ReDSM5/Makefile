.PHONY: env-create env-update pip-sync augment-nlpaug augment-textattack augment-hybrid augment-all regenerate-synonyms train train-optuna evaluate train-criteria train-evidence train-joint train-criteria-optuna train-evidence-optuna train-joint-optuna evaluate-criteria evaluate-evidence evaluate-joint data-preview lint format test clean mlflow-ui tensorboard optuna-dashboard dvc-init dvc-status help

PYTHON ?= python
ENV_NAME ?= redsm5
MAMBA_RUN ?= mamba run -n $(ENV_NAME)
MLFLOW_URI ?= http://mlflow:5000

env-create:
	mamba env create -f environment.yml

env-update:
	mamba env update -f environment.yml --prune

pip-sync:
	$(MAMBA_RUN) python -m pip install -r requirements.txt

augment-nlpaug:
	$(MAMBA_RUN) $(PYTHON) scripts/generate_nlpaug_dataset.py

augment-textattack:
	$(MAMBA_RUN) $(PYTHON) scripts/generate_textattack_dataset.py

augment-hybrid:
	$(MAMBA_RUN) $(PYTHON) scripts/generate_hybrid_dataset.py

augment-all: augment-nlpaug augment-textattack augment-hybrid

regenerate-synonyms:
	$(MAMBA_RUN) $(PYTHON) scripts/regenerate_augmentation.py

train:
	$(MAMBA_RUN) $(PYTHON) -m src.training.train

train-optuna:
	$(MAMBA_RUN) $(PYTHON) -m src.training.train_optuna

evaluate:
	$(MAMBA_RUN) $(PYTHON) -m src.training.evaluate

# Multi-Agent Training Modes
train-criteria:
	$(MAMBA_RUN) $(PYTHON) -m src.training.train_criteria training_mode=criteria

train-evidence:
	$(MAMBA_RUN) $(PYTHON) -m src.training.train_evidence training_mode=evidence

train-joint:
	$(MAMBA_RUN) $(PYTHON) -m src.training.train_joint training_mode=joint

# Multi-Agent HPO
train-criteria-optuna:
	$(MAMBA_RUN) $(PYTHON) -m src.training.train_criteria_optuna training_mode=criteria

train-evidence-optuna:
	$(MAMBA_RUN) $(PYTHON) -m src.training.train_evidence_optuna training_mode=evidence

train-joint-optuna:
	$(MAMBA_RUN) $(PYTHON) -m src.training.train_joint_optuna training_mode=joint

# Multi-Agent Evaluation
evaluate-criteria:
	$(MAMBA_RUN) $(PYTHON) -m src.training.evaluate_criteria training_mode=criteria

evaluate-evidence:
	$(MAMBA_RUN) $(PYTHON) -m src.training.evaluate_evidence training_mode=evidence

evaluate-joint:
	$(MAMBA_RUN) $(PYTHON) -m src.training.evaluate_joint training_mode=joint

data-preview:
	$(MAMBA_RUN) $(PYTHON) -c "import pandas as pd; df = pd.read_csv('Data/Augmentation/augmented_positive_pairs.csv'); print(df.head())"

lint:
	$(MAMBA_RUN) ruff check src tests scripts

format:
	$(MAMBA_RUN) black src tests scripts

test:
	$(MAMBA_RUN) pytest --maxfail=1 --disable-warnings -v

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
	@echo "Note: MLflow server should already be running via docker-compose"
	@echo "If not running, start with: docker-compose -f .devcontainer/docker-compose.yml up mlflow"

tensorboard:
	@echo "Starting TensorBoard on http://localhost:6006"
	$(MAMBA_RUN) tensorboard --logdir=logs --host=0.0.0.0 --port=6006

optuna-dashboard:
	@echo "Starting Optuna Dashboard on http://localhost:8080"
	@bash scripts/launch_optuna_dashboard.sh

# DVC Commands
dvc-init:
	$(MAMBA_RUN) dvc init --no-scm

dvc-status:
	$(MAMBA_RUN) dvc status

dvc-push:
	$(MAMBA_RUN) dvc push

dvc-pull:
	$(MAMBA_RUN) dvc pull

# Help
help:
	@echo "Available commands:"
	@echo ""
	@echo "Environment:"
	@echo "  env-create        - Create mamba environment"
	@echo "  env-update        - Update mamba environment"
	@echo "  pip-sync          - Install Python dependencies"
	@echo ""
	@echo "Data Augmentation:"
	@echo "  augment-nlpaug    - Generate NLPAug dataset"
	@echo "  augment-textattack - Generate TextAttack dataset"
	@echo "  augment-hybrid    - Generate hybrid dataset"
	@echo "  augment-all       - Run all augmentation pipelines"
	@echo ""
	@echo "Original Training:"
	@echo "  train             - Standard training"
	@echo "  train-optuna      - Hyperparameter optimization with Optuna"
	@echo "  evaluate          - Evaluate trained model"
	@echo ""
	@echo "Multi-Agent Training:"
	@echo "  train-criteria    - Train criteria matching agent (Mode 1)"
	@echo "  train-evidence    - Train evidence binding agent (Mode 2)"
	@echo "  train-joint       - Train both agents jointly (Mode 3)"
	@echo ""
	@echo "Multi-Agent HPO:"
	@echo "  train-criteria-optuna - HPO for criteria matching agent"
	@echo "  train-evidence-optuna - HPO for evidence binding agent"
	@echo "  train-joint-optuna    - HPO for joint training"
	@echo ""
	@echo "Multi-Agent Evaluation:"
	@echo "  evaluate-criteria - Evaluate criteria matching agent"
	@echo "  evaluate-evidence - Evaluate evidence binding agent"
	@echo "  evaluate-joint    - Evaluate joint model"
	@echo ""
	@echo "Monitoring:"
	@echo "  mlflow-ui         - Open MLflow UI (must be running)"
	@echo "  tensorboard       - Start TensorBoard"
	@echo "  optuna-dashboard  - Start Optuna Dashboard"
	@echo ""
	@echo "Data Version Control:"
	@echo "  dvc-init          - Initialize DVC"
	@echo "  dvc-status        - Check DVC status"
	@echo "  dvc-push          - Push data to remote"
	@echo "  dvc-pull          - Pull data from remote"
	@echo ""
	@echo "Development:"
	@echo "  lint              - Run ruff linter"
	@echo "  format            - Format code with black"
	@echo "  test              - Run pytest tests"
	@echo "  clean             - Remove cache files"
