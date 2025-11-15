.PHONY: aug-install aug-singletons aug-k3 aug-all env-create env-update pip-sync pip-install lint format test clean help

PYTHON ?= python
ENV_NAME ?= redsm5

MAMBA_EXISTS := $(shell command -v mamba 2> /dev/null)
ifdef MAMBA_EXISTS
    RUN_CMD = mamba run -n $(ENV_NAME)
else
    RUN_CMD =
endif

aug-install:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements-augment.txt

aug-singletons:
	$(PYTHON) tools/generate_augsets.py \
		--input data/raw/annotations.csv \
		--output-root data/processed/augsets \
		--combo-mode singletons \
		--variants-per-sample 2 \
		--num-proc $$(( $$(nproc) - 1 ))

aug-k3:
	$(PYTHON) tools/generate_augsets.py \
		--input data/raw/annotations.csv \
		--output-root data/processed/augsets \
		--combo-mode bounded_k \
		--max-combo-size 3 \
		--variants-per-sample 2 \
		--num-proc $$(( $$(nproc) - 1 ))

aug-all:
	$(PYTHON) tools/generate_augsets.py \
		--input data/raw/annotations.csv \
		--output-root data/processed/augsets \
		--combo-mode all \
		--confirm-powerset \
		--variants-per-sample 1

env-create:
	mamba env create -f environment.yml

env-update:
	mamba env update -f environment.yml --prune

pip-sync:
	$(RUN_CMD) $(PYTHON) -m pip install -r requirements-augment.txt

pip-install:
	$(PYTHON) -m pip install -r requirements-augment.txt

lint:
	$(RUN_CMD) ruff check src tools tests

format:
	$(RUN_CMD) black src tools tests

test:
	$(RUN_CMD) pytest --maxfail=1 --disable-warnings -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

help:
	@echo "Available targets:"
	@echo "  aug-install      - Install augmentation dependencies"
	@echo "  aug-singletons   - Generate singleton augmentation datasets"
	@echo "  aug-k3           - Generate combos up to size 3"
	@echo "  aug-all          - Generate all combos (use with sharding)"
	@echo "  env-create       - Create mamba environment"
	@echo "  env-update       - Update mamba environment"
	@echo "  pip-sync         - Sync augmentation requirements"
	@echo "  pip-install      - Install augmentation requirements"
	@echo "  lint             - Run Ruff lint checks"
	@echo "  format           - Run Black formatter"
	@echo "  test             - Run unit tests"
	@echo "  clean            - Remove build artifacts"
