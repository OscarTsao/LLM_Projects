# Quick Start Guide: NO-AUG Repository

This guide will get you up and running with the NO-AUG baseline repository in minutes.

## Prerequisites

- Python 3.10+
- Poetry (will be installed by setup script if missing)

## Step 1: Initial Setup (5 minutes)

```bash
cd /experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence
./setup.sh
```

This will:
- Install Poetry (if needed)
- Install all dependencies
- Install pre-commit hooks
- Run tests to verify setup

## Step 2: Prepare Your Data (10 minutes)

### Option A: Use Existing CSV Files

If you have CSV files with `text`, `status`, and `cases` columns:

```bash
# Copy your data files
cp /path/to/your/train.csv data/processed/train.csv
cp /path/to/your/val.csv data/processed/val.csv
cp /path/to/your/test.csv data/processed/test.csv
```

### Option B: Generate Ground Truth from Raw Data

If you have raw data that needs validation:

```bash
# For criteria task (uses status field)
make groundtruth-criteria INPUT=/path/to/raw/data.csv

# For evidence task (uses cases field)
make groundtruth-evidence INPUT=/path/to/raw/data.csv
```

## Step 3: Run Quick Sanity Check (5 minutes)

Test that everything works with a minimal HPO run:

```bash
# For criteria task
make hpo-sanity TASK=criteria MODEL=bert_base

# For evidence task
make hpo-sanity TASK=evidence MODEL=bert_base
```

This runs 3 quick trials to verify the pipeline works.

## Step 4: Run Multi-stage or Maximal HPO (Hours)

Two turnkey targets orchestrate the new Optuna/MLflow pipelines:

```bash
# Multi-stage pipeline (S0→S1→S2→Refit) for all agents
make full-hpo-all

# Maximal single-stage search for all agents
make maximal-hpo-all
```

Both targets honour the environment knobs documented in the Makefile. The most
useful overrides are:

| Variable            | Purpose                              | Default |
| ------------------- | ------------------------------------ | ------- |
| `HPO_TRIALS_S0`     | Stage‑0 trial budget                 | `64`    |
| `HPO_TRIALS_S1`     | Stage‑1 trial budget                 | `32`    |
| `HPO_TRIALS_S2`     | Stage‑2 trial budget                 | `16`    |
| `HPO_TRIALS`        | Maximal mode trial budget            | `100`   |
| `HPO_EPOCHS`        | Training epochs per trial            | `6`     |
| `HPO_SEEDS`         | Comma separated list of seeds        | `1`     |
| `AGENTS`            | Override agents list (space separated)| all four |

Example – run a quick smoke sweep on criteria only:

```bash
HPO_TRIALS_S0=8 HPO_TRIALS_S1=6 HPO_TRIALS_S2=4 AGENTS="criteria" make full-hpo-all
```

## Step 5: Train Best Model (1-2 hours)

Train the final model with best hyperparameters:

```bash
make train-best STUDY=./outputs/hpo/best_study.pkl TASK=criteria
```

## Step 6: View Results

```bash
# Export metrics to CSV
make export-metrics EXPERIMENT=noaug_baseline OUTPUT=./results.csv

# Or view in MLflow UI (metadata in mlflow.db, artifacts in ./mlruns)
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
# Then open http://localhost:5000 in your browser

# Inspect Top-K trials directly from Optuna
poetry run psy-agents show-best --agent criteria --study noaug-criteria-max --topk 5
```

### On-the-fly Augmentation (Evidence)

Enable CPU-light augmentation for the evidence task directly from the CLI:

```bash
psy-agents train --agent evidence --model-name bert-base-uncased \
  --aug-lib both --aug-methods all --aug-p-apply 0.2 --aug-ops-per-sample 1 \
  --aug-max-replace 0.3
```

During HPO you can sweep augmentation-only spaces after your baseline study:

```bash
make tune-evidence-aug
psy-agents show-best --agent evidence --study aug-evidence-ext --topk 5
```

Augmentation decisions (methods, counts, timings) are logged to MLflow and
an example set is saved as `augmentation/aug_examples.jsonl` in each run.

## Common Commands

### Testing

```bash
# Run all tests
make test

# Run specific test file
poetry run pytest tests/test_groundtruth.py -v

# Run with coverage
poetry run pytest --cov=src/psy_agents_noaug tests/
```

### Code Quality

```bash
# Format code
make format

# Check formatting without changing
make format-check

# Run linters
make lint

# Run all pre-commit hooks
make pre-commit-run
```

### Data Validation

```bash
# Validate ground truth for criteria task
poetry run python scripts/make_groundtruth.py \
  --input data/processed/train.csv \
  --output-dir data/processed \
  --task criteria \
  --split train

# Validate ground truth for evidence task
poetry run python scripts/make_groundtruth.py \
  --input data/processed/train.csv \
  --output-dir data/processed \
  --task evidence \
  --split train
```

### Using Different Models

```bash
# Use RoBERTa instead of BERT
make hpo-sanity TASK=criteria MODEL=roberta_base

# Use DeBERTa
make hpo-sanity TASK=criteria MODEL=deberta_v3_base
```

## Directory Structure

```
NoAug_Criteria_Evidence/
├── src/psy_agents_noaug/  # Source code
├── configs/                # Hydra configurations
├── scripts/                # Executable scripts
├── tests/                  # Test suite
├── data/
│   ├── raw/               # Raw data (DSM criteria)
│   └── processed/         # Processed data (splits, ground truth)
├── mlruns/                # MLflow tracking data
└── outputs/               # Training outputs
```

## STRICT Validation Rules

Remember the STRICT field mapping:

- **Criteria Task**: Uses ONLY `status` field
- **Evidence Task**: Uses ONLY `cases` field
- **NO mixing**: Ground truth files are validated to prevent contamination

## Configuration Files

All settings are in `configs/`:

- `configs/config.yaml` - Main configuration
- `configs/model/` - Model architectures (BERT, RoBERTa, DeBERTa)
- `configs/task/` - Task definitions (criteria, evidence)
- `configs/hpo/` - HPO stage configurations
- `configs/training/` - Training hyperparameters

## Troubleshooting

### "Module not found" errors

```bash
poetry shell  # Activate Poetry environment
```

### "Permission denied" on scripts

```bash
chmod +x scripts/*.py
```

### Test failures

```bash
# Check that DSM criteria file exists
ls -la data/raw/redsm5/dsm_criteria.json

# Re-run setup
./setup.sh
```

### MLflow connection errors

```bash
# Remove old tracking data
rm -rf mlruns/
mkdir mlruns/
```

## Advanced Usage

### Custom Configuration

Create custom config file:

```yaml
# configs/model/custom.yaml
_target_: psy_agents_noaug.models.encoders.BERTEncoder

variant: "large"
freeze_encoder: true
pooling_strategy: "mean"
max_length: 512
```

### Python API

```python
from psy_agents_noaug.data.loaders import ReDSM5Loader, DSMCriteriaLoader
from psy_agents_noaug.models.encoders import BERTEncoder
from psy_agents_noaug.models.criteria_head import CriteriaModel

# Load data
loader = ReDSM5Loader(data_path="./data/processed")
df = loader.load_csv("train")

# Create model
encoder = BERTEncoder(variant="base")
model = CriteriaModel(encoder, num_classes=7)
```

### Custom HPO Search Space

Edit `configs/hpo/stage1_coarse.yaml`:

```yaml
search_space:
  lr:
    type: "loguniform"
    low: 1.0e-5
    high: 5.0e-5

  batch_size:
    type: "categorical"
    choices: [8, 16, 32, 64]
```

## Next Steps

1. Read `README.md` for detailed documentation
2. Check `SETUP_SUMMARY.md` for complete file listing
3. Explore test files in `tests/` for usage examples
4. Review model implementations in `src/psy_agents_noaug/models/`

## Getting Help

- Check docstrings in source code
- Run tests to see examples: `make test`
- View configuration examples in `configs/`
- Read inline comments in scripts

## Important Reminders

1. This is the **NO-AUG baseline** - no augmentation code
2. Always validate ground truth with STRICT rules
3. Use fixed seeds for reproducibility (default: 42)
4. Track all experiments with MLflow
5. Run tests before committing changes

Good luck! 🚀
