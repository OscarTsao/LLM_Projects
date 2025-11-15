PYTHON := python3
export PYTHONPATH := $(shell pwd)

.PHONY: setup lint test run splits check-gates clean

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

check-gates:
	@last=$$(ls -d outputs/evaluation/run_* 2>/dev/null | tail -n 1); \
	if [ -z "$$last" ]; then \
		echo "No evaluation runs found under outputs/evaluation"; \
		exit 1; \
	fi; \
	scripts/check_gates.sh --metrics $$last/test_metrics.json

clean:
	rm -rf outputs artifacts mlruns __pycache__ */__pycache__ .pytest_cache
