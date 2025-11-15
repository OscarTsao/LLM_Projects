# Evidence Binding Baseline

Lightweight training and evaluation scripts that reproduce the best `noaug-criteria-supermax` Optuna trial (`trial 1951`) from the **NoAug_Criteria_Evidence** search. The pipeline keeps the original Hydra/MLflow workflow while swapping in the optimized RoBERTa architecture, sliding-window tokenisation, and balanced focal loss tuned for evidence binding on the gated **ReDSM5** corpus (see `data/redsm5/README.md` for licensing details).

## Layout

- `config/` – Hydra configuration groups covering dataset paths, model hyperparameters, training schedule, MLflow, and evaluation defaults.
- `src/data.py` – dataset assembly (ground truth + optional augmentation) and deterministic 5-fold group splits with windowed tokenisation support.
- `src/model.py` – RoBERTa encoder with a max-pooled four-layer ReLU head plus balanced focal loss utilities for evidence binding.
- `src/training.py` – Hydra-aware training loop with MLflow logging, mixed precision, cosine-restart scheduling, and evaluation helpers.
- `train.py` – thin Hydra CLI that delegates to the training pipeline.
- `evaluate.py` – Hydra CLI to score saved checkpoints and log results.
- `Makefile` – convenience targets for training, evaluation, and launching the MLflow UI.

## Quickstart

```bash
cd Evidence_Baseline_5Fold_NoAug
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make train-new        # fresh run (alias: make train) – runs 5-fold CV by default
make train-resume     # resume from outputs/noaug_evidence_roberta/fold_1/state_latest.pt
# override the resume target:
# make train-resume RESUME_CHECKPOINT=/path/to/state_latest.pt
# or: python train.py training.num_epochs=3
# disable AMP if you encounter overflow issues on older GPUs
# python train.py training.use_amp=false
```

### Default Evidence Model Settings

- Backbone `roberta-base` with sequence length 288, sliding-window stride 48, gradient checkpointing enabled, and the first six encoder layers frozen.
- Max-pooled 4-layer ReLU classifier head (all hidden layers width 768) with 0.46 dropout applied between layers.
- Balanced focal loss (γ = 4.0) with effective-number class weighting (`β = 0.9999`) and automatically resolved α (≈0.54 per fold).
- AdamW optimizer (`β₁ = 0.846`, `β₂ = 0.974`, `ε = 7.8e-9`, weight decay `3.9e-4`), learning rate `8.6e-6`, gradient clipping at `0.51`, batch size 24 with gradient accumulation ×3 (effective 72).
- Torch compile (`reduce-overhead` mode) is enabled by default on PyTorch ≥ 2.2 and automatically falls back to eager execution if unsupported.
- Sliding-window dataloaders cache tokenisation results, auto-tune worker counts, and use pinned memory + non-blocking GPU transfers to keep the RTX 4070 Ti saturated.
- Cosine-with-hard-restarts scheduler (2 cycles) with warmup ratio `0.168`, 20 training epochs, and AMP enabled by default.
- Training uses 5-fold stratified group CV; every fold logs to MLflow and persists `best_model.pt`, `state_latest.pt`, and metrics under `outputs/<run>/fold_k/`.

Artifacts land under `outputs/noaug_evidence_roberta/` by default, split into `fold_{k}/` subdirectories for the 5 folds:

- `best_model.pt` – best validation checkpoint for that fold (macro-F1 monitored)
- `optimizer.pt` / `scheduler.pt` – optional resume state
- `state_latest.pt` – full training state (model/optimizer/scheduler/AMP) for auto-resume
- `train_history.json` – per-epoch metrics for the fold
- `test_metrics.json` – held-out evaluation for the fold
- `resolved_config.yaml` – resolved Hydra configuration for the run

Validation and test splits for every fold always contain **only** original, non-augmented examples; augmentation rows are not consumed in the default configuration.

To re-score a saved checkpoint without training:

```bash
make evaluate                             # uses outputs/noaug_evidence_roberta/fold_1/best_model.pt
# or specify a different checkpoint
python evaluate.py evaluation.checkpoint=/path/to/checkpoint.pt
```

Both scripts load the bundled assets under `data/redsm5/` (original posts + annotations) and optional `data/augmentation/` CSVs for synthetic positives.

## Data Prep

- The project expects `data/redsm5/redsm5_posts.csv` and `data/redsm5/redsm5_annotations.csv`. These ship with the repo for convenience; if you need to refresh them, copy the originals into that directory.
- ReDSM5 is a gated dataset. Review `data/redsm5/README.md` for licensing, access instructions, and usage constraints before distributing or modifying the files.
- Augmentation CSVs live in `data/augmentation/` and remain optional. Disable them by setting `dataset.use_augmented=[]` via Hydra overrides if you want to train only on the original corpus.

## Tracking Runs

Training and evaluation automatically log parameters, per-epoch metrics, and artifacts to a local MLflow backend (`mlflow.db` + `mlruns/`). Launch the UI with:

```bash
make mlflow-ui
```
