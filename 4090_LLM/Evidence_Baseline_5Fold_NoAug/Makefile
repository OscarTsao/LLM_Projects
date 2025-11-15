PYTHON ?= python
HYDRA_ARGS ?=
EVAL_CHECKPOINT ?= outputs/trial_0043_rebuild/fold_1/best_model.pt
MLFLOW_UI_PORT ?= 5000
RESUME_CHECKPOINT ?=

.PHONY: train train-new train-resume evaluate mlflow-ui

train: train-new

train-new:
	$(PYTHON) train.py training.auto_resume=false $(HYDRA_ARGS)

train-resume:
	@if [ -n "$(RESUME_CHECKPOINT)" ]; then \
		$(PYTHON) train.py training.auto_resume=true training.resume_checkpoint=$(RESUME_CHECKPOINT) $(HYDRA_ARGS); \
	else \
		$(PYTHON) train.py training.auto_resume=true $(HYDRA_ARGS); \
	fi

evaluate:
	$(PYTHON) evaluate.py evaluation.checkpoint=$(EVAL_CHECKPOINT) $(HYDRA_ARGS)

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///$$(pwd)/mlflow.db --default-artifact-root $$(pwd)/mlruns --port $(MLFLOW_UI_PORT)
