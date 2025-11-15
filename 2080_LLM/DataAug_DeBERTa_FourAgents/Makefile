PYTHON := python3
export PYTHONPATH := $(shell pwd)

.PHONY: setup lint test run splits hpo hpo-evidence hpo-best retrain-best check-gates clean

setup:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m compileall src

test:
	$(PYTHON) -m pytest tests -q

run:
	$(PYTHON) -m src.pipeline.run_pipeline --seed 42

splits:
	scripts/generate_splits.py --data data/redsm5_sample.jsonl --out-dir configs/data/splits --seed 42

hpo:
	bash scripts/run_hpo_evidence.sh

hpo-evidence:
	bash scripts/run_hpo_evidence.sh

# Two-stage Optuna HPO (robust driver)
TRIALS_A ?= 400
EPOCHS_A ?= 5
TRIALS_B ?= 120
EPOCHS_B ?= 12
K_TOP ?= 5
MODEL ?= microsoft/deberta-v3-base
SEED ?= 42

hpo-best:
	$(PYTHON) -m src.hpo.hpo_driver \
	  --model $(MODEL) --trials-a $(TRIALS_A) --epochs-a $(EPOCHS_A) \
	  --trials-b $(TRIALS_B) --epochs-b $(EPOCHS_B) --k-top $(K_TOP) --seed $(SEED)

# HF-backed two-stage HPO (GPU if available)
hpo-best-hf:
	USE_HF_TRAIN=1 USE_TORCH_COMPILE=1 DL_PREFETCH=4 $(PYTHON) -m src.hpo.hpo_driver \
	  --model $(MODEL) --trials-a $(TRIALS_A) --epochs-a $(EPOCHS_A) \
	  --trials-b $(TRIALS_B) --epochs-b $(EPOCHS_B) --k-top $(K_TOP) --seed $(SEED)

RETRAIN_SEEDS ?= 3
retrain-best:
	$(PYTHON) -m src.hpo.hpo_driver --retrain-best --retrain-seeds $(RETRAIN_SEEDS)

check-gates:
	@last=$$(ls -d outputs/evaluation/run_* 2>/dev/null | tail -n 1); \
	if [ -z "$$last" ]; then \
		echo "No evaluation runs found under outputs/evaluation"; \
		exit 1; \
	fi; \
	scripts/check_gates.sh --metrics $$last/test_metrics.json

clean:
	rm -rf outputs artifacts mlruns __pycache__ */__pycache__ .pytest_cache
