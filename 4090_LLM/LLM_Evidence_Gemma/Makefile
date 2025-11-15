.PHONY: help install install-dev train train-resume train-5fold train-5fold-resume train-quick train-quick-resume train-5fold-9b train-5fold-9b-resume evaluate clean lint format test check-gpu check-data data-stats train-classifier train-classifier-resume smoke-classifier

# Optional overrides for Hydra-based commands (set via `make TARGET OUTPUT_DIR=... RESUME_ENABLED=true RESUME_START_FOLD=3`)
HYDRA_RESUME_OVERRIDES := $(strip \
	$(if $(OUTPUT_DIR),output_dir=$(OUTPUT_DIR),) \
	$(if $(RESUME_ENABLED),resume.enabled=$(RESUME_ENABLED),) \
	$(if $(RESUME_START_FOLD),resume.start_fold=$(RESUME_START_FOLD),) \
)

# Default target
help:
	@echo "LLM_Evidence_Gemma Makefile Commands:"
	@echo ""
	@echo "Installation:"
	@echo "  make install          - Install package and dependencies"
	@echo "  make install-dev      - Install with development dependencies"
	@echo ""
	@echo "Training:"
	@echo "  make train            - 5-fold cross-validation (script runner)"
	@echo "  make train-resume     - (info) resume unsupported for script runner"
	@echo "  make train-5fold      - Hydra 5-fold CV (Gemma-2-2b)"
	@echo "  make train-5fold-resume OUTPUT_DIR=... RESUME_START_FOLD=N - Resume Hydra folds"
	@echo "  make train-5fold-9b   - Hydra 5-fold CV with Gemma-2-9b"
	@echo "  make train-5fold-9b-resume OUTPUT_DIR=... RESUME_START_FOLD=N - Resume 9B run"
	@echo "  make train-quick      - Hydra quick preset (5 folds, encoder frozen)"
	@echo "  make train-quick-resume OUTPUT_DIR=... RESUME_START_FOLD=N - Resume quick preset"
	@echo "  make train-classifier - LLM classification training (QLoRA)"
	@echo "  make train-classifier-resume - (info) resume unsupported for classifier"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate CHECKPOINT=path/to/model.pt  - Evaluate model"
	@echo "  make evaluate-best    - Evaluate best model from latest run"
	@echo ""
	@echo "Data:"
	@echo "  make check-data       - Verify data files exist"
	@echo "  make data-stats       - Show dataset statistics"
	@echo "  make create-cv-splits - Create cross-validation splits"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             - Run flake8 linting"
	@echo "  make format           - Format code with black"
	@echo "  make test             - Run tests"
	@echo ""
	@echo "Utilities:"
	@echo "  make check-gpu        - Check GPU availability"
	@echo "  make clean            - Remove generated files"
	@echo "  make smoke-classifier - Run tiny overfit smoke test"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,tracking]"

# Training
train:
	python src/training/train_gemma_qa.py \
		--data_dir data/redsm5 \
		--output_dir $${OUTPUT_DIR:-outputs/gemma_qa} \
		--model_name google/gemma-2-2b \
		--batch_size 4 \
		--num_epochs 100 \
		--early_stopping_patience 20

train-resume:
	$(error Resume is not supported for the standalone script runner. Use the Hydra targets (train-5fold*) instead.)

train-5fold:
	python src/training/train_gemma_qa_hydra.py \
		experiment=full_5fold \
		model.name=google/gemma-2-2b $(HYDRA_RESUME_OVERRIDES)

train-5fold-resume:
ifndef OUTPUT_DIR
	$(error OUTPUT_DIR must be set (e.g., OUTPUT_DIR=outputs/2025-11-08/14-55-23))
endif
	$(MAKE) train-5fold OUTPUT_DIR=$(OUTPUT_DIR) RESUME_ENABLED=true RESUME_START_FOLD=$(RESUME_START_FOLD)

train-5fold-9b:
	python src/training/train_gemma_qa_hydra.py \
		experiment=full_5fold \
		training.batch_size=1 \
		model.name=google/gemma-2-9b \
		model.use_qlora=true \
		model.lora_r=64 \
		model.lora_alpha=16 \
		model.lora_dropout=0.05 $(HYDRA_RESUME_OVERRIDES)

train-5fold-9b-resume:
ifndef OUTPUT_DIR
	$(error OUTPUT_DIR must be set (e.g., OUTPUT_DIR=outputs/2025-11-08/14-55-23))
endif
	$(MAKE) train-5fold-9b OUTPUT_DIR=$(OUTPUT_DIR) RESUME_ENABLED=true RESUME_START_FOLD=$(RESUME_START_FOLD)

train-quick:
	python src/training/train_gemma_qa_hydra.py \
		experiment=quick_test \
		model.name=google/gemma-2-2b $(HYDRA_RESUME_OVERRIDES)

train-quick-resume:
ifndef OUTPUT_DIR
	$(error OUTPUT_DIR must be set (e.g., OUTPUT_DIR=outputs/2025-11-08/14-55-23))
endif
	$(MAKE) train-quick OUTPUT_DIR=$(OUTPUT_DIR) RESUME_ENABLED=true RESUME_START_FOLD=$(RESUME_START_FOLD)

train-classifier:
ifndef TRAIN_FILE
	$(error TRAIN_FILE is not set. Usage: make train-classifier TRAIN_FILE=path/to/train.csv VAL_FILE=path/to/val.csv MODEL=google/gemma-2-2b)
endif
ifndef VAL_FILE
	$(error VAL_FILE is not set. Usage: make train-classifier TRAIN_FILE=... VAL_FILE=...)
endif
	python src/training/train_llm_classifier.py \
		--model_name $${MODEL:-sshleifer/tiny-gpt2} \
		--train_file $(TRAIN_FILE) \
		--validation_file $(VAL_FILE) \
		--text_column $${TEXT_COLUMN:-text} \
		--label_column $${LABEL_COLUMN:-label} \
		--mode $${MODE:-causal}

train-classifier-resume:
	$(error Resume is not supported for the classifier target yet. Capture checkpoints manually if needed.)

# Evaluation
evaluate:
ifndef CHECKPOINT
	$(error CHECKPOINT is not set. Usage: make evaluate CHECKPOINT=path/to/model.pt)
endif
	python src/training/evaluate.py \
		--checkpoint $(CHECKPOINT) \
		--data_dir data/redsm5 \
		--output_dir evaluation_results

evaluate-best:
	@LATEST_RUN=$$(ls -td outputs/*/* | head -1); \
	CHECKPOINT=$$LATEST_RUN/fold_0/best_model.pt; \
	if [ -f "$$CHECKPOINT" ]; then \
		echo "Evaluating $$CHECKPOINT"; \
		python src/training/evaluate.py \
			--checkpoint $$CHECKPOINT \
			--data_dir data/redsm5 \
			--output_dir evaluation_results; \
	else \
		echo "No checkpoint found at $$CHECKPOINT"; \
		exit 1; \
	fi

# Data utilities
check-data:
	@echo "Checking for required data files..."
	@test -f data/redsm5/redsm5_posts.csv && echo "✓ redsm5_posts.csv found" || echo "✗ redsm5_posts.csv missing"
	@test -f data/redsm5/redsm5_annotations.csv && echo "✓ redsm5_annotations.csv found" || echo "✗ redsm5_annotations.csv missing"

data-stats:
	python -c "import pandas as pd; \
		posts = pd.read_csv('data/redsm5/redsm5_posts.csv'); \
		annotations = pd.read_csv('data/redsm5/redsm5_annotations.csv'); \
		print(f'Posts: {len(posts)}'); \
		print(f'Annotations: {len(annotations)}'); \
		print(f'Positive examples: {annotations[annotations.status==1].shape[0]}'); \
		print(f'\\nSymptom distribution:'); \
		print(annotations[annotations.status==1].DSM5_symptom.value_counts())"

create-cv-splits:
	python -c "from src.data.cv_splits import create_cv_splits; \
		create_cv_splits('data/redsm5/redsm5_posts.csv', \
			'data/redsm5/redsm5_annotations.csv', \
			num_folds=5, \
			output_dir='data/cv_splits')"

# Code quality
lint:
	flake8 src/ --max-line-length=100 --ignore=E203,W503

format:
	black src/ --line-length=100

test:
	pytest tests/ -v

# Utilities
check-gpu:
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); \
		print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ 2>/dev/null || true
	@echo "Cleaned up generated files"

smoke-classifier:
	python scripts/smoke_overfit_classifier.py
