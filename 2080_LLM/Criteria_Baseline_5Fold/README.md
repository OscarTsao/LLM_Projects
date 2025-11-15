# Criteria Baseline Rebuild

Minimal training and evaluation scripts that reproduce the 0.8535 macro-F1 criteria classifier taken from `DataAugmentation_Evaluation` Optuna trial `0043`. The project points at the assets copied into `gemini_reranker/baselines/dataaug_trial_0043/` so you can re-run training, benchmark Gemini-tuned variants, or refresh checkpoints without rehydrating the full original repository.

## Layout

- `config/` – Hydra configuration groups covering dataset paths, model hyperparameters, training schedule, MLflow, and evaluation defaults.
- `src/data.py` – lightweight dataset assembly (ground truth + hybrid/nlpaug/textattack augmentation) and deterministic group splits.
- `src/model.py` – DeBERTa encoder with two-layer MLP head plus adaptive focal loss.
- `src/training.py` – Hydra-aware training loop with MLflow logging, mixed precision, and evaluation helpers.
- `train.py` – thin Hydra CLI that delegates to the training pipeline.
- `evaluate.py` – Hydra CLI to score saved checkpoints and log results.
- `Makefile` – convenience targets for training, evaluation, and launching the MLflow UI.

## Quickstart

```bash
cd Criteria_Baseline_Rebuild
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make train-new        # fresh run (alias: make train) – runs 5-fold CV by default
make train-resume     # resume from outputs/trial_0043_rebuild/fold_1/state_latest.pt
# override the resume target:
# make train-resume RESUME_CHECKPOINT=/path/to/state_latest.pt
# or: python train.py training.num_epochs=3
# disable AMP if you encounter overflow issues on older GPUs
# python train.py training.use_amp=false
```

Artifacts land under `outputs/trial_0043_rebuild/` by default, split into `fold_{k}/` subdirectories for the 5 folds:

- `best_model.pt` – best validation checkpoint for that fold (macro-F1 monitored)
- `optimizer.pt` / `scheduler.pt` – optional resume state
- `state_latest.pt` – full training state (model/optimizer/scheduler/AMP) for auto-resume
- `train_history.json` – per-epoch metrics for the fold
- `test_metrics.json` – held-out evaluation for the fold
- `resolved_config.yaml` – resolved Hydra configuration for the run

Disable cross-validation (falling back to the original single split) by running `python train.py dataset.cross_validation.enabled=false`.

Validation and test splits always contain **only** original, non-augmented examples; augmentation rows are added to the training split exclusively.

To re-score a saved checkpoint without training:

```bash
make evaluate                             # uses outputs/trial_0043_rebuild/fold_1/best_model.pt
# or specify a different checkpoint
python evaluate.py evaluation.checkpoint=/path/to/checkpoint.pt
```

Both scripts rely on the data already synced inside `../gemini_reranker/baselines/dataaug_trial_0043/`.

## Tracking Runs

Training and evaluation automatically log parameters, per-epoch metrics, and artifacts to a local MLflow backend (`mlflow.db` + `mlruns/`). Launch the UI with:

```bash
make mlflow-ui
```
