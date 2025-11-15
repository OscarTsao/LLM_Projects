# DataAug DeBERTa Criteria

Single-agent criteria detection pipeline for mental-health posts. The system fine-tunes a fixed `microsoft/deberta-v3-base` encoder with configurable classification heads, loss functions, and prediction calibration. Optuna handles algorithmic hyper-parameter optimisation, and every run is tracked locally with MLflow (SQLite backend).

---

## Overview

- ☑️ **Task**: multi-label criteria classification only. Evidence binding, multi-task heads, and PEFT hooks have been removed.
- 🧠 **Backbone**: always `microsoft/deberta-v3-base`; Optuna sweeps head architecture, loss, optimiser, scheduler, thresholding, and (optionally) augmentation knobs.
- 🧪 **Experiment tracking**: MLflow writes to `experiments/mlflow.db` and stores artifacts under `experiments/artifacts/`.
- 🔁 **Determinism**: CLI exposes `--seed` and `--deterministic` to seed Python/NumPy/Torch and toggle deterministic algorithms.
- 🧩 **Reproducible tooling**: VS Code Dev Container with explicit bind mount (`/workspaces/DataAug_DeBERTa_Criteria`) so host files appear in-container.
- 📥 **Dataset fallback**: set `DATAAUG_USE_LOCAL_REDSM5=1` to force use of `Data/redsm5/*.csv` when working fully offline.

---

## Quick Start

```bash
# 1. (Optional) run hyperparameter search
make hpo              # 20 Optuna trials, local SQLite storage

# 2. Train with the best saved configuration
make hpo-best         # uses experiments/best_params.json

# 3. Inspect results
make mlflow-ui        # launches http://localhost:5000 using experiments/mlflow.db

# 4. Manual run (no HPO)
python -m src.dataaug_multi_both.cli.train \
  --epochs 3 --per-device-batch-size 8 --seed 1337
```

---

## Project Layout

```text
src/dataaug_multi_both/
├── cli/train.py                  # Training & Optuna driver (criteria-only)
├── data/dataset.py               # HF dataset loading + PyTorch dataset
├── hpo/search_space.py           # Algorithmic search space (head/loss/optim/etc.)
├── models/
│   ├── criteria_model.py         # DeBERTa encoder + head wrapper
│   └── heads/criteria_matching.py# Classification head factory
├── training/
│   ├── losses.py                 # BCE, weighted BCE, focal, asymmetric
│   └── metrics.py                # Threshold optimisation helpers
└── utils/mlflow_setup.py         # Local SQLite + artifact directory wiring

tests/                             # Lightweight smoke tests (HPO guardrails, heads, losses, MLflow)
.devcontainer/                     # VS Code dev-container (bind mount fixed)
Makefile                           # make hpo / hpo-best / mlflow-ui / smoke
```

---

## Installation

### Dev Container (recommended)

1. Open the repository in VS Code.
2. Install the “Dev Containers” extension.
3. `Ctrl/Cmd+Shift+P → Dev Containers: Reopen in Container`.
4. Verify the workspace is mounted: `ls /workspaces/DataAug_DeBERTa_Criteria` (should show host files).  
   If not, ensure Docker Desktop file sharing includes this path or reopen the folder.

### Local environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install optuna mlflow datasets transformers torch torchvision --upgrade

python -m pytest -q        # smoke tests
```

---

## CLI Usage

The unified CLI (`python -m src.dataaug_multi_both.cli.train`) supports both scripted HPO and single-run training.

Common flags:
- `--dataset-id` (`irlab-udc/redsm5` default) and optional `--dataset-revision`.
- Hardware knobs (fixed, non-HPO): `--per-device-batch-size`, `--max-seq-length`, `--grad-accumulation-steps`, `--amp-dtype {none,fp16,bf16}`, `--gradient-checkpointing`.
- Algorithmic sweeps via Optuna: executed when `--hpo` is passed; parameters come from `hpo/search_space.py`.
- Determinism: `--seed`, `--deterministic`.

Example manual run with deterministic mode:

```bash
python -m src.dataaug_multi_both.cli.train \
  --epochs 4 \
  --per-device-batch-size 8 \
  --max-seq-length 384 \
  --seed 2025 \
  --deterministic
```

---

## Optuna Search Space

Only algorithmic knobs are optimised—batch size, sequence length, AMP level, num_workers, etc., are *not* part of the search.

| Category | Parameters |
|----------|------------|
| **Pooling** | `pooling ∈ {cls, mean}` |
| **Head** | `head.type`, `head.dropout`, `head.bias`, `head.mlp.layers`, `head.mlp.hidden_dim`, `head.mlp.activation`, `head.mlp.layernorm`, `head.glu.hidden_dim`, `head.glu.gate_bias`, `head.msd.n_samples`, `head.msd.alpha` |
| **Loss** | `loss.name`, `loss.focal.gamma`, `loss.focal.alpha_pos`, `loss.asym.gamma_pos`, `loss.asym.gamma_neg` |
| **Prediction** | `pred.threshold.policy ∈ {fixed,opt_global}`, `pred.threshold.global` (only when optimising) |
| **Optimiser** | `optim.name` (adamw), `optim.lr_encoder`, `optim.lr_head`, `optim.weight_decay` |
| **Scheduler** | `sched.name ∈ {linear, cosine}`, `sched.warmup_ratio` |
| **Training** | `train.grad_clip_norm` |

Guardrails prevent forbidden prefixes such as `batch_size`, `max_seq_length`, `num_workers`, `amp_dtype`, `torch_compile`, etc. (see `tests/test_hpo_params.py`).

---

## MLflow Tracking

- Tracking URI: `sqlite:///experiments/mlflow.db`
- Artifacts: `experiments/artifacts/`
- Each run logs: Optuna parameters, hardware config, epoch metrics (`train.loss`, `val.loss`, `val.macro_f1`, chosen threshold), and the best-model state dict.

Launch the UI locally:

```bash
make mlflow-ui       # opens http://localhost:5000
```

Runs started inside the dev container are visible on the host because the workspace is bind-mounted.

---

## Determinism & Reproducibility

- `--seed` seeds Python, NumPy and Torch (CPU & CUDA).
- `--deterministic` enables `torch.use_deterministic_algorithms(True)` and keeps CuDNN in deterministic mode.  
  The CLI sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing Torch to avoid runtime warnings.
- Validation thresholding optionally performs a global sweep (`pred.threshold.policy=opt_global`) through `training/metrics.optimize_global_threshold`.

---

## Dev Container Troubleshooting

Host files not visible inside container?

1. Check `.devcontainer/devcontainer.json` uses the bind mount:  
   `"source=${localWorkspaceFolder},target=/workspaces/DataAug_DeBERTa_Criteria,type=bind"`.
2. After rebuild, run `ls /workspaces/DataAug_DeBERTa_Criteria`. If empty:
   - Ensure the host path is shared with Docker Desktop (macOS/Windows).
   - Close any dangling containers and reopen through VS Code.
3. File permissions issues: the post-create step runs  
   `sudo chown -R vscode:vscode /workspaces/DataAug_DeBERTa_Criteria`. Re-run `Dev Containers: Rebuild Container` if needed.

---

## Tests

```bash
make smoke           # python -m pytest -q
make test            # full pytest run
```

The suite verifies:
- HPO param prefixes exclude hardware knobs (`tests/test_hpo_params.py`)
- Head factory output shapes (`tests/test_heads.py`)
- Loss functions backpropagate (`tests/test_losses.py`)
- Threshold search returns sane values (`tests/test_thresholding.py`)
- MLflow SQLite initialisation (`tests/test_mlflow_sqlite.py`)

---

Happy experimenting! 🧪
