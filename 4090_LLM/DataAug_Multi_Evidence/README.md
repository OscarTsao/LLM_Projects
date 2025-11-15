DataAug Multi Both

Storage-optimized multi-task NLP training & HPO project.

Environment and Dependencies

- Use Poetry for dependency management:
  - Install: `poetry install`
  - Update lock: `poetry lock`
- Augmentation backends require extras beyond PyTorch. `poetry install` now brings in `nlpaug`, `textattack`, `tensorflow`, and `tensorflow-hub`, and the runtime swaps DataLoader workers to the `spawn` start method so GPU-aware augmenters (e.g., CLARE) run without manual intervention.
- Poetry lock is the source of truth. For Docker builds, export requirements from the lock:
  - `poetry export -f requirements.txt --output docker/requirements.txt --without-hashes`
  - Keep the exported file in sync with `poetry.lock` (CI recommended).

Data Configuration

- Data is loaded strictly via Hugging Face Datasets with explicit splits.
- Configure dataset in `configs/data/dataset.yaml`:
  ```yaml
  dataset:
    id: irlab-udc/redsm5
    revision: null
    splits:
      train: train
      validation: validation
      test: test
    streaming: false
    cache_dir: ${oc.env:HF_HOME,~/.cache/huggingface}
  ```

Evaluation Artifacts

- Per-trial evaluation reports are the authoritative source for test metrics:
  - Path: `experiments/trial_<uuid>/evaluation_report.json`
- Optional study summary (best trial reference):
  - Schema: `specs/002-storage-optimized-training/contracts/study_output_schema.json`
  - CLI stub: `python -m dataaug_multi_both.cli.evaluate_study --study-id <uuid> --best-trial-id <uuid>`

Common Commands

- Lint: `poetry run ruff check .`
- Format: `poetry run black .`
- Types: `poetry run mypy src`
- Tests: `poetry run pytest -v`

Performance tip: the HPO objective now (a) uses `spawn`-mode DataLoader workers to keep CUDA-capable augmenters alive, (b) auto-selects train/eval batch-size search spaces based on detected GPU memory (with separate knobs for validation throughput), and (c) spins up several workers by default so GPU time is not lost waiting on CPU-bound augmentation. Override with `resources_num_workers` or pin batch sizes in Optuna parameters if you need different values.

Notes

- Keep large data/artifacts out of Git. See `.gitignore`.
- Prefer VS Code Dev Containers for consistent Poetry setup.

> Migration note: v2.0 removes the criteria classification component; the repository is now single-agent (evidence binding).
