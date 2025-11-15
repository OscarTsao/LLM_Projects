PYTHON ?= python
OVERRIDES ?=

.PHONY: install train hpo clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) -m src.cli train $(OVERRIDES)

hpo:
	$(PYTHON) -m src.cli hpo $(OVERRIDES)

clean:
	rm -rf outputs mlruns optuna.db mlflow.db

