.PHONY: help train train-resume optuna optuna-resume evaluate validate clean clean-checkpoints clean-optuna test

# Default target
help:
	@echo "SpanBERT Evidence Extraction - Available Commands:"
	@echo ""
	@echo "Training:"
	@echo "  make train              - Start fresh training run"
	@echo "  make train-resume       - Auto-resume from latest checkpoint (if exists)"
	@echo "  make train EPOCHS=50    - Override number of epochs"
	@echo "  make train BATCH=16     - Override batch size"
	@echo ""
	@echo "Hyperparameter Search:"
	@echo "  make optuna             - Start Optuna hyperparameter search (auto-resumes)"
	@echo "  make optuna-resume      - Explicitly resume Optuna study"
	@echo "  make optuna TRIALS=50   - Run with specific number of trials"
	@echo "  make optuna TIMEOUT=3600 - Run with timeout in seconds"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate CHECKPOINT=artifacts/20231201_120000/best_model.pt"
	@echo "                          - Evaluate a specific checkpoint"
	@echo ""
	@echo "Validation:"
	@echo "  make validate           - Check Python syntax"
	@echo "  make test              - Run tests (if available)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean             - Remove all artifacts and checkpoints"
	@echo "  make clean-checkpoints - Remove only checkpoint files"
	@echo "  make clean-optuna      - Remove Optuna study database"

# Training commands
train:
	@echo "Starting fresh training..."
ifdef EPOCHS
	python -m src.train training.num_train_epochs=$(EPOCHS) $(ARGS)
else ifdef BATCH
	python -m src.train training.train_batch_size=$(BATCH) $(ARGS)
else
	python -m src.train $(ARGS)
endif

train-resume:
	@echo "Attempting to resume training from latest checkpoint..."
	python -m src.train $(ARGS)

# Optuna hyperparameter search (auto-resumes by default with persistent storage)
optuna:
	@echo "Starting Optuna hyperparameter search (auto-resumes if study exists)..."
ifdef TRIALS
	python -m src.optuna_search optuna.n_trials=$(TRIALS) $(ARGS)
else ifdef TIMEOUT
	python -m src.optuna_search optuna.timeout=$(TIMEOUT) $(ARGS)
else
	python -m src.optuna_search $(ARGS)
endif

optuna-resume:
	@echo "Resuming Optuna study..."
	python -m src.optuna_search $(ARGS)

# Evaluation
evaluate:
ifndef CHECKPOINT
	@echo "Error: CHECKPOINT parameter required"
	@echo "Usage: make evaluate CHECKPOINT=artifacts/20231201_120000/best_model.pt"
	@exit 1
endif
ifndef CONFIG
	$(eval CONFIG := $(shell dirname $(CHECKPOINT))/config.yaml)
endif
	@echo "Evaluating checkpoint: $(CHECKPOINT)"
	@if [ ! -f "$(CHECKPOINT)" ]; then \
		echo "Error: Checkpoint file not found: $(CHECKPOINT)"; \
		exit 1; \
	fi
	@if [ ! -f "$(CONFIG)" ]; then \
		echo "Error: Config file not found: $(CONFIG)"; \
		exit 1; \
	fi
	python scripts/evaluate.py \
		--config $(CONFIG) \
		--checkpoint $(CHECKPOINT) \
		--split test \
		--metrics_output reports/test_metrics.json \
		--predictions_output reports/test_predictions.json

# Validation
validate:
	@echo "Validating Python syntax..."
	python -m compileall src scripts

test:
	@echo "Running tests..."
	@if [ -d "tests" ]; then \
		python -m pytest tests/ -v; \
	else \
		echo "No tests directory found"; \
	fi

# Cleanup commands
clean:
	@echo "Cleaning all artifacts, checkpoints, and cache..."
	rm -rf artifacts/
	rm -rf optuna_studies/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf src/psya_agent/__pycache__/
	rm -rf .pytest_cache/
	rm -rf reports/
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete!"

clean-checkpoints:
	@echo "Removing checkpoint files..."
	find artifacts/ -type f -name "checkpoint.pt" -delete 2>/dev/null || true
	@echo "Checkpoint cleanup complete!"

clean-optuna:
	@echo "Removing Optuna study database..."
	rm -rf optuna_studies/
	@echo "Optuna database removed!"

# Development helpers
.PHONY: install dev-install format lint

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

dev-install:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install pytest black flake8 mypy

format:
	@echo "Formatting code with black..."
	black src/ scripts/

lint:
	@echo "Linting code..."
	flake8 src/ scripts/ --max-line-length=120 --ignore=E203,W503
	mypy src/ --ignore-missing-imports
