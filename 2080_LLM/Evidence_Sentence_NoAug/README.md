# AI/ML Experiment Template

Minimal template for ML experiments using PyTorch, Transformers, Hydra, MLflow, and
Optuna. This repo is configured for a BERT‑based binary classification task with
NSP‑style criterion–sentence inputs and local experiment tracking.

## Quickstart

- Python 3.10+ recommended.
- Create and activate a virtual environment, then install:

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

## Layout

- `src/Project/SubProject/models/model.py` – example model wrapper for Transformers.
- `src/Project/SubProject/utils/` – utility helpers (`get_logger`, `set_seed`, MLflow helpers).
- `mlruns/` – local MLflow runs (if using file-based tracking).
- `outputs/` – suggested place for artifacts.

## Data & Input Format

- Task: Binary classification with Hugging Face BERT variants.
- Input format (NSP‑style pairing):
  - `[CLS] <criterion> [SEP] <sentence> [SEP]`
  - Criterion text from `data/DSM5/`
  - Sentence text from posts (e.g., `data/redsm5/posts.csv`)
- All preprocessing/tokenization choices should be logged to MLflow.

## Configuration (Hydra)

- Keep modular configs in `configs/` and override via CLI:
  - Example: `+trainer.max_epochs=3 model.name=bert-base-uncased`
- Typical parameters: data paths, model/tokenizer name, training hparams, seeds,
  MLflow/Optuna settings, and output dirs.

## MLflow (Local DB + Artifacts)

Use a local MLflow server with SQLite DB and file artifacts per the project
constitution.

1) Start MLflow UI/server in this repo root:

```
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

2) Configure the client in code to point at the server (default http://127.0.0.1:5000):

```python
from Project.SubProject.utils import configure_mlflow, enable_autologging, mlflow_run

configure_mlflow(tracking_uri="http://127.0.0.1:5000", experiment="demo")
enable_autologging()

with mlflow_run("hello", tags={"stage": "dev"}, params={"lr": 1e-4}):
    # your training loop here
    pass
```

This setup uses `mlflow.db` for the tracking database and `mlruns/` as the local
artifact store.

## Optional HPO (Optuna)

- If you run HPO, prefer Optuna with local storage:
  - Storage: `sqlite:///optuna.db`
  - Integrate trials with Hydra for search spaces and with MLflow for logging.

## Reproducibility

- Use `Project.SubProject.utils.set_seed` to set global seeds.
  - Example: `from Project.SubProject.utils import set_seed; set_seed(42)`

## Development

- Run linters/formatters:
```
ruff check src tests
black src tests
```
- Run tests (add your own under `tests/`):
```
pytest
```

Style and formatting gates:
- Google‑style docstrings + type hints
- Black (line length 100), Ruff, MyPy

