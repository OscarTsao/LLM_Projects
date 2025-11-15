# Gemini Reranker

Hydra-driven pipelines for preference-based reranking and span extraction using Gemini-provided labels and MLflow tracking. The repository ships with an offline baseline checkpoint, deterministic demo data, and tooling to run the full candidate generation ➜ judging ➜ dataset ➜ training ➜ inference workflow on CPU in minutes.

## Environment Setup

Activate the environment with either Conda or a plain virtualenv before running the pipeline.

**Conda**

```bash
conda env create -f environment.yml
conda activate gemini-reranker
python -m nltk.downloader punkt
```

**Pip / virtualenv**

```bash
python -m venv .venv
source .venv/bin/activate      # On Windows use: .venv\Scripts\activate
make setup                     # installs -e .[dev] (same as pip install -r requirements.txt) and downloads NLTK punkt
```

## Quickstart

With the environment active you can build the offline pipeline end-to-end in minutes:

```bash
make demo-data

# Candidate generation (top-8 sentences per criterion)
python -m criteriabind.cli.candidate_gen split=train candidate_gen.k=8

# Deterministic mock judge (no networking required)
python -m criteriabind.cli.judge split=train judge.provider=mock

# Convert judgments into pairwise/listwise datasets
python -m criteriabind.cli.pair_builder split=train

# Train the cross-encoder ranker offline
python -m criteriabind.train.train_criteria_ranker \
  train.max_steps=300 hydra.job.name=ranker_offline_demo

# Run offline inference on the held-out split
python -m criteriabind.cli.infer split=test
```

The commands above log into `mlflow.db` (SQLite) and emit artefacts under `.pytest_cache`/`outputs/…` when `output_dir` is overridden, or under `outputs/…` by default. To bring up a local MLflow UI:

```bash
make mlflow-up    # serves http://127.0.0.1:5000 with ./mlruns/ artefacts
```

## Real Data + Gemini Workflow

Switch `data.source` in the Hydra config (or via overrides) to point at your production dataset. The loader supports JSONL, CSV, Parquet, and Hugging Face datasets; map column names through `data.mapping.*`. A minimal CSV example:

```bash
python -m criteriabind.cli.candidate_gen \
  data.source=csv \
  data.path=/data/notes \
  '+data.path_or_name=train.csv' \
  '+data.mapping.sample_id=note_id' \
  '+data.mapping.note=note_text' \
  '+data.mapping.criterion_name=criterion_name'
```

Once candidate jobs exist you can label with Gemini (requires `GEMINI_API_KEY`, or set `judge.api_key`):

```bash
export GEMINI_API_KEY=...
python -m criteriabind.cli.judge split=train judge.provider=gemini \
  judge.model=gemini-1.5-pro
python -m criteriabind.cli.pair_builder split=train
python -m criteriabind.train.train_criteria_ranker train=ranker_real hydra.job.name=ranker_real_run
```

Gemini-specific safeguards live in `conf/judge/gemini.yaml`: `sample_rate`, `max_requests_per_minute`, `max_total_cost_usd`, and `cache_uri` (SQLite cache). Set `judge.dry_run=true` to log prompts without sending requests, or `judge.fallback_to_mock=true` to drop back to the deterministic provider when credentials are missing.

## Why K? Choosing K

`candidate_gen.k` controls how many snippets per (note, criterion) move forward to judging and training. Larger values improve oracle recall (more chances to include the ground-truth evidence) but require more judging budget and increase training time. The default `k=8` balances recall with speed on the demo set. Keep an eye on the `recall@K` number printed by the generator—when it plateaus near 1.0, increasing `k` mostly adds noise. Override the value inline when you need deeper pools:

```bash
python -m criteriabind.cli.candidate_gen split=train candidate_gen.k=12
```

## Hydra CLI Examples

All entrypoints accept Hydra overrides. Common patterns:

```bash
python -m criteriabind.train.train_criteria_ranker \
  train.max_steps=300 \
  hardware.device=cpu \
  train.save_top_k=0 \
  mlflow.tracking_uri=sqlite:///tmp/mlflow.db

python -m criteriabind.train.train_evidence_span \
  train=span_fast hydra.job.name=span_demo \
  hardware.device=auto \
  data.max_samples=200

python -m criteriabind.cli.judge split=train judge.provider=mock
python -m criteriabind.cli.pair_builder split=train
python -m criteriabind.cli.infer split=test
python -m criteriabind.cli.evaluate
```

Hydra outputs live in `outputs/%Y-%m-%d/%H-%M-%S-job`. Override `output_dir` if you need deterministic paths.

## Model Initialisation

The default configuration fine-tunes `microsoft/deberta-v3-small` from Hugging Face for both ranking and span extraction. If you have a previous checkpoint you want to warm-start from, set `model.from_pretrained_path=/path/to/checkpoint` (the directory should contain the usual `config.json`/`pytorch_model.bin` layout or a saved `training_state.pt`). Tokenisation is cached under `~/.cache/huggingface`.

## Tooling & Layout

- `conf/` – Hydra configuration tree (defaults, hardware, MLflow, data, models, judges).
- `src/criteriabind/` – package code (data loaders, models, training, CLI, utilities).
- `demo_data/` – synthetic ReDSM5-style JSONL splits created by `make demo-data`.
- `baselines/` – frozen baseline checkpoint and augmentation artefacts.
- `tests/` – pytest suite covering config composition, dataset caching, judge fallbacks, and offline end-to-end smoke runs.
- `requirements.txt` / `environment.yml` / `Dockerfile` – reproducible environment manifests.

### Make Targets

| Target            | Description |
|-------------------|-------------|
| `make setup`      | Install editable package with dev extras and download NLTK punkt |
| `make demo-data`  | Generate the tiny ReDSM5 demo splits |
| `make judge`      | Run the mock Gemini judge (defaults to offline provider) |
| `make judge-gemini-train` | Run the Gemini judge on the train split (requires API key) |
| `make judge-gemini-test`  | Run the Gemini judge on the test split (requires API key) |
| `make pairs`      | Build pairwise datasets from judged outputs |
| `make train-criteria` | Launch the ranker training loop via Hydra |
| `make train-evidence` | Launch the span extractor loop |
| `make infer`      | Score candidates and emit ranked predictions |
| `make eval`       | Evaluate the ranker on the validation split |
| `make mlflow-up`  | Serve the local MLflow UI against `sqlite:///mlflow.db` |
| `make quickstart` | Alias for the offline demo pipeline (candidate → judge → pairs → train) |
| `make dpo-train`  | Launch the LLM DPO finetuning entrypoint (`conf/train/llm_dpo.yaml`) |

## MLflow Tracking

Every entrypoint:

1. Calls `mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)` (SQLite by default).
2. Sets the experiment `gemini_reranker` and run name `hydra.job.name` unless overridden.
3. Logs the resolved Hydra config, Git metadata, hardware snapshot, metrics, and (optionally) checkpoints.

Set `MLFLOW_TRACKING_URI` externally or override `mlflow.tracking_uri` to direct runs elsewhere. To skip checkpoint uploads in constrained environments, run with `train.save_top_k=0 train.save_every_steps=0`.

## Privacy & Networking

The default configuration is entirely offline: candidate generation, judging, pair building, training, and inference run locally without network calls. Judging uses a deterministic mock provider whose decisions are reproducible under a fixed seed. When you are ready to use Gemini for labeling, set `judge.provider=gemini` and provide an API key via `GEMINI_API_KEY` or `judge.api_key`; the CLI will fail fast with a clear error if the key is missing. All artifacts stay on disk (`mlflow.db`, `./mlruns/`, `outputs/…`) so you can reason about privacy boundaries explicitly. Enable the optional Vertex path via `judge.vertex_enabled=true` with `judge.vertex_project`/`judge.vertex_location` when running inside GCP.

## Post-SFT Finetuning with DPO

1. Convert judgments into DPO preference pairs:
   ```bash
   python -m criteriabind.cli.prefs_to_dpo \
     input_path=outputs/2025-01-01/12-00-00-judge/judgments_gemini.jsonl \
     output_path=/data/dpo/train.jsonl
   ```
2. Launch DPO training (defaults in `conf/train/llm_dpo.yaml`):
   ```bash
   python -m criteriabind.train.train_dpo \
     model.base_model_path=/models/my_sft_checkpoint \
     data.path_or_name=/data/dpo/train.jsonl \
     train.max_steps=500 \
     hydra.job.name=dpo_my_model
   ```
   LoRA is enabled by default; disable via `hardware.lora.use_lora=false`. Metrics, adapters, and merged weights stream to MLflow under the `gemini_reranker_dpo` experiment.

## Testing & CI

```bash
make test          # pytest -q
make lint          # ruff check .
make type          # mypy src
```

`.github/workflows/ci.yml` executes formatting, linting, typing, demo data generation, and the test suite on each push/PR. A `.pre-commit-config.yaml` is provided for local hooks (ruff, mypy, EOF checks).

## Judge Providers

The default judge config (`conf/judge/offline_mock.yaml`) uses a deterministic mock so the demo pipeline stays offline. `conf/judge/gemini.yaml` enables Gemini via `judge.provider=gemini`, JSON-only prompting, request caching (`cache_uri`), retry/backoff, and budget controls (`sample_rate`, `max_requests_per_minute`, `max_total_cost_usd`). Set `GEMINI_API_KEY` or `judge.api_key` before calling the real API.

## License

MIT. See `baselines/` README for provenance of the mirrored Optuna checkpoint and augmentation files.
