# HPO Guide - Production-Ready with Real Data

This guide covers the two hyperparameter optimization (HPO) systems available in this project, **both now connected to real redsm5 data**.

## 📊 Data Status

✅ **ALL HPO systems now use REAL redsm5 data:**
- **Criteria task**: `data/redsm5/redsm5_annotations.csv` (2,081 samples)
- **Evidence task**: `data/processed/redsm5_matched_evidence.csv` (2,935 samples)
- **NO synthetic data** - production ready!

## 🎯 Two HPO Approaches

### 1. Multi-Stage HPO (Progressive Refinement)

**Concept**: Progressive refinement across 4 stages with increasing trials.

**Stages:**
- **Stage 0 (Sanity)**: 8 trials, 3 epochs - Quick validation
- **Stage 1 (Coarse)**: 20 trials - Broad search
- **Stage 2 (Fine)**: 50 trials - Narrow search
- **Stage 3 (Refit)**: Train best config on train+val combined

**Use when:**
- You want systematic, interpretable optimization
- You need intermediate results at each stage
- You prefer smaller, manageable trial counts
- You want to validate before committing to large runs

**Commands:**

```bash
# Single architecture
make hpo-s0 HPO_TASK=criteria HPO_MODEL=roberta_base
make hpo-s1 HPO_TASK=criteria HPO_MODEL=roberta_base
make hpo-s2 HPO_TASK=criteria HPO_MODEL=roberta_base
make refit HPO_TASK=criteria

# All stages in sequence
make full-hpo HPO_TASK=criteria HPO_MODEL=roberta_base

# ALL architectures (criteria, evidence, share, joint)
make full-hpo-all
```

**Search Space:**
- Learning rate: 1e-6 to 5e-5 (log uniform)
- Weight decay: 1e-5 to 1e-1 (log uniform)
- Batch size: 8, 16, 32
- Warmup ratio: 0.0 to 0.2
- Dropout: 0.0 to 0.3

**Output:**
- `outputs/hpo_stage0/` - Sanity check results
- `outputs/hpo_stage1/` - Coarse search results
- `outputs/hpo_stage2/best_config.yaml` - Best hyperparameters
- `outputs/hpo_stage3/` - Refit results

---

### 2. Maximal HPO (Single Large Run)

**Concept**: Single massive optimization with 600-1200 trials exploring huge search space.

**Trial Counts (default):**
- **Criteria**: 800 trials
- **Evidence**: 1200 trials
- **Share**: 600 trials
- **Joint**: 600 trials

**Use when:**
- You want maximum performance
- You have compute budget for large searches
- You want to explore model architectures too
- You need the best possible hyperparameters

**Commands:**

```bash
# Single architecture
make tune-criteria-max    # 800 trials
make tune-evidence-max    # 1200 trials
make tune-share-max       # 600 trials
make tune-joint-max       # 600 trials

# ALL architectures sequentially
make maximal-hpo-all

# Custom settings with CLI
poetry run python scripts/run_all_hpo.py \
    --mode maximal \
    --architectures criteria evidence \
    --n-trials 500 \
    --parallel 4
```

**Search Space (much larger):**

**Models:**
- BERT: base, large
- RoBERTa: base, large
- DeBERTa-v3: base, large
- XLM-RoBERTa: base
- ~~ELECTRA: base, large~~ (excluded - incompatible with CriteriaModel)

**Optimizers:**
- AdamW, AdamW 8-bit, Adafactor, Lion

**Schedulers:**
- Linear, Cosine, Cosine with restarts, Polynomial, OneCycle

**Head Architecture:**
- Pooling: CLS, mean, max, attention
- Layers: 1-4
- Hidden size: 256, 384, 512, 768, 1024, 1536, 2048
- Activation: GELU, ReLU, SiLU

**Loss Functions:**
- Classification: CE, CE with label smoothing, Focal
- QA: QA-CE, QA-CE-LS, QA-Focal

**And much more** (tokenization, regularization, etc.)

**Output:**
- `./_runs/` - Training runs and MLflow logs
- `./_optuna/noaug.db` - Optuna study database
- `./_runs/{agent}_{study}_topk.json` - Top-K configurations

---

## 🚀 Quick Start

### Option A: Multi-Stage (Recommended for first run)

```bash
# 1. Ensure data exists
ls data/redsm5/redsm5_annotations.csv
ls data/processed/redsm5_matched_evidence.csv

# 2. Run sanity check first
make hpo-s0 HPO_TASK=criteria

# 3. If successful, run full pipeline
make full-hpo HPO_TASK=criteria

# 4. Or run all architectures
make full-hpo-all
```

### Option B: Maximal (For maximum performance)

```bash
# 1. Run single architecture
make tune-criteria-max

# 2. Or run all architectures (long running!)
make maximal-hpo-all

# 3. View results
poetry run python -m psy_agents_noaug.cli show-best \
    --agent criteria \
    --study noaug-criteria-max
```

---

## 📁 Wrapper Script: run_all_hpo.py

**Purpose**: Run HPO sequentially for all architectures (criteria, evidence, share, joint).

### Usage Examples

```bash
# Multi-stage HPO for all architectures
python scripts/run_all_hpo.py --mode multistage

# Maximal HPO for all architectures
python scripts/run_all_hpo.py --mode maximal

# Specific architectures only
python scripts/run_all_hpo.py --mode multistage --architectures criteria evidence

# Custom maximal HPO settings
python scripts/run_all_hpo.py \
    --mode maximal \
    --n-trials 500 \
    --parallel 4 \
    --outdir ./_runs/my_experiment

# Skip sanity check in multi-stage
python scripts/run_all_hpo.py --mode multistage --skip-sanity

# Stop on first error
python scripts/run_all_hpo.py --mode maximal --stop-on-error
```

### Options

```
--mode {multistage,maximal}    HPO mode (required)
--architectures ARCH [ARCH ...]  Architectures to run (default: all 4)
--model MODEL                  Model for multi-stage (default: roberta_base)
--skip-sanity                  Skip stage 0 in multi-stage
--n-trials N                   Number of trials for maximal
--parallel N                   Parallel jobs for maximal (default: 1)
--timeout SECONDS              Timeout for maximal mode
--outdir DIR                   Output directory for maximal
--stop-on-error                Stop if any architecture fails
```

---

## 🔬 Comparison: Multi-Stage vs Maximal

| Feature | Multi-Stage | Maximal |
|---------|------------|---------|
| **Trials** | 78 total (8+20+50) | 600-1200 |
| **Runtime** | Hours | Days |
| **Search space** | Medium (5 params) | Huge (20+ params) |
| **Model search** | ❌ Fixed model | ✅ 9 models |
| **Interpretability** | ✅✅✅ High | ⚠️ Lower |
| **Resource cost** | Low | High |
| **Use case** | Quick iteration | Final tuning |

---

## 💡 Best Practices

### 1. Start Small
```bash
# Always start with sanity check
make hpo-s0 HPO_TASK=criteria
```

### 2. Use Multi-Stage for Development
```bash
# Faster feedback during development
make full-hpo HPO_TASK=criteria
```

### 3. Use Maximal for Production
```bash
# Final tuning before publication
make tune-criteria-max
```

### 4. Monitor with MLflow
```bash
# Terminal 1: Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Terminal 2: Run HPO
make hpo-s1 HPO_TASK=criteria

# Open browser: http://localhost:5000
```

### 5. Parallel Execution
```bash
# Maximal HPO supports parallelism
poetry run python scripts/run_all_hpo.py \
    --mode maximal \
    --parallel 4  # Use 4 GPUs/CPUs
```

### 6. Resume Failed Runs
```bash
# Optuna stores state in database
# Rerun same command - it will resume
make tune-criteria-max
```

---

## 📊 Real Data Details

### Criteria Task
- **File**: `data/redsm5/redsm5_annotations.csv`
- **Samples**: 2,081 annotations
- **Split**: 80% train (1,665), 10% val (208), 10% test (208)
- **Task**: Binary classification (status: present/absent)
- **Columns**: sentence_text, DSM5_symptom, status

### Evidence Task
- **File**: `data/processed/redsm5_matched_evidence.csv`
- **Samples**: 2,935 matched evidence spans
- **Split**: 80% train (2,348), 10% val (294), 10% test (293)
- **Task**: Span extraction (start/end positions)
- **Columns**: post_text, sentence_text, start_char, end_char

### Data Loading
Both HPO systems use:
- ✅ Real dataset loaders from `src/Project/{Criteria,Evidence}/data/dataset.py`
- ✅ Proper train/val/test splitting with seed control
- ✅ Transformers tokenizers
- ✅ Hardware-optimized DataLoaders

---

## 🐛 Troubleshooting

### Issue: "Dataset file not found"
```bash
# Check files exist
ls data/redsm5/redsm5_annotations.csv
ls data/processed/redsm5_matched_evidence.csv

# If missing, generate groundtruth
make groundtruth
```

### Issue: "Out of memory"
```bash
# Reduce batch size (multi-stage)
# Edit configs/hpo/stage0_sanity.yaml
# Change batch_size choices to [4, 8, 16]

# Or use gradient accumulation
# Maximal HPO will auto-adjust
```

### Issue: "ImportError: No module named 'Project'"
```bash
# Ensure you're in project root
cd /path/to/NoAug_Criteria_Evidence

# Check sys.path includes src/
python -c "import sys; print(sys.path)"
```

### Issue: "Trial pruned immediately"
```bash
# This is normal - Optuna prunes bad configs early
# Check MLflow for pruned trial details
mlflow ui
```

### Issue: "Protobuf/Sentencepiece errors with DeBERTa in parallel mode"

**Symptoms:**
```
TypeError: Descriptors cannot be created directly
TypeError: duplicate file name sentencepiece_model.proto
```

**Cause**: DeBERTa tokenizers use sentencepiece with protobuf, which has multiprocessing issues when `--parallel > 1`.

**Fix**: Already implemented! The environment variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is set automatically in:
- `scripts/tune_max.py` (line 28)
- All Makefile parallel HPO targets

**Manual Override** (if needed):
```bash
# Force environment variable
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python scripts/tune_max.py --parallel 4 ...

# Or fallback to single worker
python scripts/tune_max.py --parallel 1 ...
```

**See**: `docs/PROTOBUF_FIX.md` for detailed analysis and troubleshooting.

---

## 📈 Expected Results

### Multi-Stage HPO
- **Stage 0**: Val F1 ~0.60-0.75 (sanity check)
- **Stage 1**: Val F1 ~0.70-0.80 (coarse)
- **Stage 2**: Val F1 ~0.75-0.85 (fine)
- **Stage 3**: Test F1 typically +2-5% over stage 2

### Maximal HPO
- **Best trial**: Val F1 ~0.80-0.90+ (task dependent)
- **Model matters**: DeBERTa-v3 typically best
- **Convergence**: ~100-200 trials to near-optimal

---

## 🔗 Related Documentation

- **Training Guide**: `docs/TRAINING_GUIDE.md`
- **CLI Guide**: `docs/CLI_AND_MAKEFILE_GUIDE.md`
- **Project Overview**: `CLAUDE.md`
- **Data Pipeline**: `docs/DATA_PIPELINE_IMPLEMENTATION.md`

---

## ✅ Production Checklist

- [x] Real redsm5 data connected
- [x] Multi-stage HPO implemented
- [x] Maximal HPO implemented
- [x] Wrapper script for all architectures
- [x] MLflow logging
- [x] Optuna pruning
- [x] Reproducibility (seeds)
- [x] Hardware optimization
- [x] Documentation complete

**Status**: 🚀 PRODUCTION READY!
