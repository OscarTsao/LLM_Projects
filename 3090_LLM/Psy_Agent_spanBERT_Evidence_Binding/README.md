# SpanBERT Evidence Binding

Fine-tuning SpanBERT (`SpanBERT/spanbert-base-cased`) to identify evidential spans inside Reddit mental-health posts. The pipeline uses Hugging Face Transformers, PyTorch, Hydra for configuration, and Optuna for hyper-parameter exploration.

## Project Layout

```
.
├── configs/                # Hydra configuration entry points
├── src/                    # Python package with data, model, training, optuna entry points
│   ├── psya_agent/         # Core library (data prep, features, metrics, trainer)
│   ├── optuna_search.py    # Optuna optimisation entry point
│   └── train.py            # Standard training entry point
├── scripts/                # Utility CLI scripts
│   └── evaluate.py         # Re-run evaluation for a saved checkpoint
├── artifacts/              # Saved model checkpoints + metrics (created at runtime)
├── Data/                   # Provided dataset (ground truth + annotations)
└── README.md
```

## Data

- Contexts: `Data/groundtruth/redsm5_ground_truth.json` (JSONL with `post_id`, `text`).
- Annotations: `Data/redsm5/redsm5_annotations.csv` with sentence-level evidence (`sentence_text`). Only rows flagged with `status == 1` are used.
- Splits: deterministic per `post_id` using ratios in the config (default 80/10/10) so the same post never leaks across splits.

## Training Workflow

1. Load posts + annotations, align evidence spans inside contexts (case-insensitive fallback).
2. Tokenise with SpanBERT tokenizer (fast) including stride support for long documents.
3. Fine-tune SpanBERT with a single linear QA head (start/end logits) using FP16 when CUDA is available.
4. Evaluate after every epoch, track best validation F1, and early stop with configurable patience.
5. Save `best_model.pt`, the resolved `config.yaml`, and `test_metrics.json` into `artifacts/<timestamp>/`.

### Hardware & Performance Optimizations

- **Local-only Hugging Face loading**: No network calls via `model.local_files_only=true`
- **Automatic mixed precision (AMP)**: FP16 training on CUDA for 2-3x speedup
- **Optimized DataLoader**: Persistent workers, prefetch factor, and pinned memory for faster data loading
- **PyTorch 2.0 compilation**: Optional `compile_model=true` for 10-20% additional speedup
- **Gradient checkpointing**: Optional memory optimization for training larger batches
- **Progress tracking**: Real-time tqdm progress bars with loss monitoring
- **Efficient gradient accumulation**: Proper handling with mixed precision
- **Minimal artifact footprint**: Only best checkpoint + metrics + config saved
- **Hydra run directory**: Pinned to project root to avoid runaway experiment folders

## Configuration (Hydra)

Main config: `configs/config.yaml`

Key sections:
- `data`: dataset paths, split ratios, seed.
- `model`: SpanBERT model name, dropout, `local_files_only` toggle.
- `features`: tokenizer settings (`max_length`, `doc_stride`, `n_best_size`, `max_answer_length`).
- `training`: hyper-parameters, hardware toggles, artifact directory.
- `optimization`: early stopping metric + patience.
- `optuna`: trial count for search runs.

Override any parameter from the CLI, e.g.:

```bash
python -m src.train training.num_train_epochs=4 training.train_batch_size=12
```

## Commands

### Standard Training

```bash
python -m src.train
```

Artifacts appear under `artifacts/<timestamp>/`.

### Hyper-parameter Search (Optuna)

```bash
python -m src.optuna_search optuna.n_trials=20
```

Optuna explores batch size, learning rate, epochs, and tokenisation window sizes; the best trial is retrained and saved under `artifacts/optuna_best/`.

### Evaluation / Prediction

```bash
python scripts/evaluate.py \
  --config artifacts/<timestamp>/config.yaml \
  --checkpoint artifacts/<timestamp>/best_model.pt \
  --split test \
  --metrics_output reports/test_metrics.json \
  --predictions_output reports/test_predictions.json
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+
- Hydra-core 1.3+
- Optuna 3.2+
- Pandas 2.0+
- tqdm 4.65+

Optional: CUDA GPU for fastest training and AMP acceleration.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Performance Tips

### For Maximum Speed
```bash
# Enable PyTorch 2.0 compilation (requires PyTorch 2.0+)
python -m src.train training.compile_model=true

# Use larger batch size with gradient accumulation
python -m src.train training.train_batch_size=16 training.gradient_accumulation_steps=2

# Increase workers for faster data loading
python -m src.train training.num_workers=4
```

### For Memory-Constrained GPUs
```bash
# Enable gradient checkpointing
python -m src.train model.gradient_checkpointing=true

# Use smaller batch size with gradient accumulation
python -m src.train training.train_batch_size=4 training.gradient_accumulation_steps=4

# Reduce sequence length
python -m src.train features.max_length=256
```

## Testing & Validation

- `python -m compileall src scripts` verifies Python syntax.
- Smoke training run (tiny split) executed to confirm end-to-end flow (`artifacts/20251001_220731`).
- `scripts/evaluate.py` validated against the saved checkpoint to reproduce reported metrics.

## Next Steps

1. Run full training with desired hyper-parameters once satisfied with exploratory runs.
2. Launch Optuna search for better F1/EM performance if more compute is available.
3. Inspect `test_metrics.json` and optional saved predictions for downstream analysis.
