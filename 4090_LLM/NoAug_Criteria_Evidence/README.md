# PSY Agents NO-AUG

## Quick Start

# setup
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"

# sanity
pytest -q

# train (stub; wire to your trainer as needed)
psy-agents train --agent criteria --model-name bert-base-uncased --epochs 1 --outdir ./_runs

# HPO (Stage A global sweep)
HPO_EPOCHS=6 make tune-criteria-max
HPO_EPOCHS=6 make tune-evidence-max

# Show winners
psy-agents show-best --agent criteria --study noaug-criteria-max --topk 5
psy-agents show-best --agent evidence --study noaug-evidence-max --topk 5

> Notes:
> - MLflow logs to ./_runs/mlruns by default (file URI).
> - Optuna storage defaults to ./_optuna/noaug.db (sqlite).
> - No augmentation: ensure criteria uses "status", evidence uses "cases" only.

## Recent Updates (October 2025)

**Production-Ready HPO System:**
- ✅ Fixed Optuna 4.5.0 compatibility (MOTPESampler → NSGAIISampler)
- ✅ Implemented functional training bridge with synthetic data
- ✅ Comprehensive HEAD search (pooling/layers/hidden/activation/dropout)
- ✅ QA null policy search (threshold/ratio/calibrated)
- ✅ 9 model backbones, 5 schedulers, 4 optimizers, regularization knobs

**Interface Parity Achieved:**
- ✅ All 4 models in `src/Project/` now accept `head_cfg` and `task_cfg`
- ✅ Fixed output keys: Share/Joint return `"logits"` (was `"criteria_logits"`)
- ✅ Backward compatible with direct parameter passing

**Code Quality:**
- ✅ Updated to PyTorch 2.x AMP API (torch.amp instead of torch.cuda.amp)
- ✅ Registered pytest markers (eliminates warnings)
- ✅ 67/69 tests passing (97.1%), 31% code coverage

**Verified Smoke Tests:**
- 3-trial HPO run completes successfully
- HEAD parameters correctly logged to MLflow
- Top-K JSON export functional

All project documentation now resides in the [`docs/`](docs/) directory.

Primary entry points:
- `docs/README.md` – project overview and repository structure
- `docs/QUICK_START.md` – quick setup and usage guide
- `docs/CI_CD_SETUP.md` – CI/CD pipeline reference
- `docs/TESTING.md` – testing strategy and commands

Additional guides are available alongside these files for setup, training
infrastructure, data pipeline details, and CLI/Makefile usage.

Model implementations for the four supported architectures live in
`src/psy_agents_noaug/architectures/` (`criteria`, `evidence`, `share`, `joint`).
