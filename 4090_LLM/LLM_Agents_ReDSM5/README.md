# ReDSM5 LLM Classifier

Fine-tune decoder-only LLMs (Llama/Qwen) with a Hugging Face classification head for multi-label DSM-5 symptom classification on the ReDSM5 dataset. The project supports full fine-tuning, LoRA, and QLoRA, performs automated hyperparameter optimization, tunes per-label thresholds, and exports reproducible artifacts.

## Features
- ✅ Decoder-only LLM fine-tuning with multi-label classification head (`full_ft`, `lora`, `qlora`).
- ✅ Sliding-window tokenization for long Reddit posts with document-level pooling (`max`, `mean`, `logit_sum`).
- ✅ Custom BCE/Focal losses with class imbalance weighting and label smoothing.
- ✅ Per-label threshold grid search and optional temperature calibration.
- ✅ Early stopping on dev macro-F1, class-balanced metrics (macro/micro/weighted F1, per-label, PR-AUC), confusion counts.
- ✅ Smoke-test friendly scripts (`run_train.sh`, `run_eval.sh`, `run_hpo.sh`).
- ✅ Optuna or Ray Tune search over architecture + hyperparameters with automatic artifact capture.
- ✅ Reproducibility: config snapshot, thresholds, metrics JSON/CSV, prediction exports, optional W&B logging.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
> **Note:** CUDA-enabled PyTorch, bitsandbytes, and transformers require a compatible GPU driver. The scripts automatically fall back to CPU if CUDA is unavailable (training will be slow).

## Dataset Preparation
Two loading options are available:
1. **Hugging Face Hub** – provide `--hf_id` and optionally `--hf_config`.
2. **Local files** – place `train.jsonl`, `dev.jsonl`, `test.jsonl` (or CSV/TSV) under a directory and pass `--data_dir`. Required fields:
   - `text`: raw Reddit post.
   - One binary column per label listed in `configs/labels.yaml`.

The default labels are the nine DSM-5 depression symptoms. Update `configs/labels.yaml` to modify or drop labels.

## Configuration
- `configs/base.yaml`: Training defaults (model, optimization, sliding windows, loss options, precision, etc.).
- `configs/search_space.yaml`: HPO search space with conditional branches for each method.
- `configs/labels.yaml`: Label list and optional drops.

Override key values through the YAML files or CLI flags (e.g., `--max_train_samples 256` for smoke tests).

## Training
```bash
./scripts/run_train.sh
# or manually
python -m src.train \
  --config configs/base.yaml \
  --labels configs/labels.yaml \
  --out_dir outputs/run1 \
  --data_dir /path/to/redsm5_local \
  --use_wandb false
```
Outputs under `outputs/run1/` include:
- `best/`: best checkpoint (HF `save_pretrained` layout) + thresholds/config copies.
- `metrics_dev.json`, `metrics_test.json`: macro/micro/weighted F1.
- `label_report_{dev,test}.csv`: per-label precision/recall/F1/PR-AUC.
- `predictions_test.csv`: per-document probabilities and predictions.
- `thresholds.json`: tuned thresholds + optional temperatures.
- `config_used.yaml`: frozen config for reproducibility.

### Key Training Options
- `method`: `full_ft`, `lora`, `qlora`.
- `loss_type`: `bce`, `focal`; `class_weighting`: `none`, `inv`, `sqrt_inv`.
- `truncation_strategy`: `single` (no pooling) or `window_pool` (long-document handling).
- `pooler`: `max`, `mean`, `logit_sum` for window aggregation.
- `calibration`: `none` or `temperature`.
- `max_length`, `doc_stride`: sliding window hyperparameters.
- `gradient_checkpointing`, `bf16`/`fp16`, `tf32`: memory/perf controls.

### Mixed Precision & OOM Handling
The trainer requests bf16 when supported, otherwise fp16. Reduce `per_device_train_batch_size` or increase `grad_accum` if you encounter CUDA OOM. For rapid debugging, use `--max_train_samples`/`--max_eval_samples` to limit data.

## Evaluation
```bash
./scripts/run_eval.sh
# or
python -m src.eval \
  --ckpt outputs/run1/best \
  --labels configs/labels.yaml \
  --data_dir /path/to/redsm5_local \
  --split test
```
Writes evaluation artifacts (metrics, per-label report, predictions) to `outputs/run1/best/eval_{split}/` by default.

## Hyperparameter Optimization
```bash
./scripts/run_hpo.sh
# Optuna example
python -m src.hpo \
  --backend optuna \
  --config configs/base.yaml \
  --search_space configs/search_space.yaml \
  --labels configs/labels.yaml \
  --data_dir /path/to/redsm5_local \
  --out_dir outputs/hpo_optuna \
  --n_trials 40 --timeout 14400

# Ray Tune example
python -m src.hpo \
  --backend ray \
  --config configs/base.yaml \
  --search_space configs/search_space.yaml \
  --labels configs/labels.yaml \
  --data_dir /path/to/redsm5_local \
  --out_dir outputs/hpo_ray \
  --num_samples 64 --ray_cpus 4 --ray_gpus 1
```
Each trial trains a full model variant, saves its artifacts under `out_dir/trial_<id>/`, and reports dev macro-F1. The scheduler space includes method switches, optimizers, schedulers, LoRA/QLoRA knobs, max length, pooling mode, etc. The best configuration and metrics are saved as `best_config.yaml` and `best_metrics.json`.

## Repository Structure
```
├── configs/            # YAML configs (base defaults, search space, labels)
├── scripts/            # CLI wrappers for training/eval/HPO
├── src/
│   ├── data.py         # Dataset loading, sliding windows, collators
│   ├── eval.py         # Standalone checkpoint evaluation
│   ├── evidence.py     # Optional sentence-level evidence stub
│   ├── hpo.py          # Optuna/Ray HPO orchestrator
│   ├── losses.py       # BCE/Focal losses with class weighting/smoothing
│   ├── metrics.py      # Macro/micro/weighted F1 + per-label metrics
│   ├── models.py       # Model factory (full FT / LoRA / QLoRA)
│   ├── thresholds.py   # Threshold grid search & temperature scaling
│   ├── train.py        # Training loop, evaluation, artifact export
│   └── utils.py        # Logging, seeding, YAML/JSON helpers
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Reproducibility Checklist
- Deterministic seeding (`seed` in configs) applied to Python, NumPy, and PyTorch.
- `config_used.yaml` + `thresholds.json` + `metrics_*.json` saved for each run.
- Git commit hash captured via `src/utils.get_git_hash()` (exposed for extension).
- Optional W&B logging (`--use_wandb true`) for experiment tracking.

## Extending the Project
- Update `configs/labels.yaml` for new label sets.
- Implement richer sentence-level evidence in `src/evidence.py`.
- Integrate accelerate/DeepSpeed by adjusting `TrainingArguments` or leveraging `accelerate launch`.
- Add cross-validation or ensemble logic by expanding `src/hpo.py` or creating new scripts.

## Troubleshooting
- **HF auth**: export `HF_HOME` or `HF_TOKEN` as needed when pulling gated models.
- **bitsandbytes import errors**: ensure compatible CUDA version or install CPU-only variant.
- **OOM**: lower `per_device_train_batch_size`, reduce `max_length`, or switch to `qlora`.
- **Ray Tune cluster**: install `ray[default]` and ensure the Ray runtime is started before launching HPO.

Happy fine-tuning! 🚀
