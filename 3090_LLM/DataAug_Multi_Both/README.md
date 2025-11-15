# DataAug Multi Both - Mental Health NLP

Deep learning framework for multi-task mental health criteria detection with hyperparameter optimization.

## 🎯 Overview

This project implements a dual-agent NLP system for mental health text analysis:
- **Criteria Matching**: Multi-label classification of mental health criteria
- **Evidence Binding**: Span extraction for evidence identification

**Key Features:**
- 15 pre-trained transformer models (BERT, DeBERTa, RoBERTa, etc.)
- Comprehensive HPO with Optuna (97-100 hyperparameters)
- PEFT methods (LoRA, Adapters, IA3)
- Advanced optimization (AdamW, Lion, Adafactor)
- MLflow experiment tracking
- Automated checkpoint management

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
conda activate llmhe  # or your environment
pip install tiktoken --upgrade
pip install --upgrade optuna "sqlalchemy>=2.0.0"
pip install -e .

# 2. Test HPO (3 trials, ~15 minutes)
make hpo-test

# 3. View results
make mlflow-ui
# Open http://localhost:5000 in browser

# 4. Run full HPO (optional, 50 trials, 4-8 hours)
make hpo
```

---

## 📁 Project Structure

```
DataAug_Multi_Both/
├── src/dataaug_multi_both/
│   ├── cli/                    # Command-line interfaces
│   │   ├── train.py           # Training & HPO entry point
│   │   └── evaluate_study.py  # Evaluation tools
│   ├── data/                   # Data loading & augmentation
│   │   ├── dataset.py         # Dataset classes
│   │   ├── dataset_loader.py  # HuggingFace dataset loading
│   │   └── augmentation.py    # Data augmentation
│   ├── models/                 # Model architectures
│   │   ├── multi_task_model.py      # Main multi-task model
│   │   ├── encoders/hf_encoder.py   # HuggingFace encoder wrapper
│   │   └── heads/                    # Task-specific heads
│   │       ├── criteria_matching.py # Classification head
│   │       └── evidence_binding.py  # Span extraction head
│   ├── training/               # Training infrastructure
│   │   ├── trainer.py         # Training loop
│   │   ├── losses.py          # Loss functions
│   │   └── checkpoint_manager.py  # Checkpoint management
│   ├── hpo/                    # Hyperparameter optimization
│   │   ├── search_space.py    # Optuna search space (97-100 params)
│   │   ├── trial_executor.py  # Trial execution
│   │   └── metrics_buffer.py  # Metrics tracking
│   └── utils/                  # Utilities
│       ├── mlflow_setup.py    # MLflow configuration
│       ├── logging.py         # Logging utilities
│       └── storage_monitor.py # Storage management
├── configs/                    # Hydra configurations
├── experiments/                # Experiment outputs (auto-generated)
│   ├── hpo_production.db      # Optuna study database
│   ├── mlflow_db/             # MLflow tracking database
│   └── trial_*/               # Trial checkpoints & logs
├── tests/                      # Unit & integration tests
├── Makefile                    # Build automation
└── README.md                   # This file
```

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM (32GB+ recommended for production HPO)
- 50GB+ disk space for experiments

### Setup

```bash
# Using conda (recommended)
conda create -n mental_health_nlp python=3.10
conda activate mental_health_nlp

# Install dependencies
pip install tiktoken --upgrade
pip install --upgrade optuna "sqlalchemy>=2.0.0"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .

# Verify installation
python -c "import tiktoken, optuna, torch; print('✅ All dependencies installed')"
```

---

## 🎯 Usage

### Hyperparameter Optimization

**Quick Test (3 trials, ~15 min):**
```bash
make hpo-test
```

**Default Run (50 trials, 4-8 hours):**
```bash
make hpo
```

**Production Run (500 trials, 40-80 hours):**
```bash
make hpo-production
```

### View Results

```bash
# MLflow UI
make mlflow-ui  # http://localhost:5000

# CLI results
make hpo-results

# Search space info
make hpo-info
```

### Evaluation

```bash
make evaluate ARGS='--study-db experiments/hpo_production.db --study-name mental_health_hpo_production'
```

---

## 🏗️ Model Architecture

### Multi-Task Model

```
Input Text → [Transformer Encoder] → [Pooling]
                                          ├→ [Criteria Head] → Multi-Label Classification
                                          └→ [Evidence Head] → Span Extraction
```

**Backbone Models (15):**
- BERT (3 variants), DeBERTa (2), SpanBERT (2)
- XLM-RoBERTa (3), ELECTRA (1), Longformer (2)
- BioBERT (1), ClinicalBERT (1)

**Pooling:** CLS, Mean, Attention, Scalar Mix

**Criteria Head:** Linear, MLP, GLU, Multi-Sample Dropout

**Evidence Head:** Linear, MLP, Biaffine, BIO-CRF, Sentence Reranker

---

## 🔬 HPO Search Space

**Total Parameters:** ~97-100

| Category | Options |
|----------|---------|
| **PEFT** | LoRA, LoRA+, AdaLoRA, Pfeiffer, Houlsby, Compacter, IA3 |
| **Optimizers** | AdamW, Adafactor, Lion, Adam |
| **Schedulers** | Linear, Cosine, Cosine Restart, One Cycle |
| **Losses** | CE, Focal, BCE, Weighted BCE, Adaptive Focal, Hybrid |
| **Adversarial** | FGM, PGD |
| **Adaptation** | DAPT, TAPT |
| **Augmentation** | NLPAug, TextAttack |

---

## 🔄 Reproducibility

### Deterministic Training
- Fixed random seeds: 42, 1337, 2025
- CUDA deterministic algorithms enabled
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` (auto-set)

### Checkpoints

**Auto-saved:**
```
experiments/trial_<uuid>/
├── checkpoints/
│   ├── checkpoint_epoch0001.pt           # Model state
│   └── checkpoint_epoch0001.pt.meta.json # Metadata
├── logs/train.log
└── config.json
```

**Contents:** Model, optimizer, scheduler states, metrics, random states

---

## 📊 Results

### Metrics
- **Primary:** F1 Score (macro-averaged)
- **Criteria:** Precision, Recall, F1 (per-class & macro)
- **Evidence:** Exact Match, Token F1, Character F1
- **Training:** Loss, gradient norm, LR

### Export

```bash
# JSON summary
python -m dataaug_multi_both.cli.evaluate_study \
    --study-db experiments/hpo_production.db \
    --study-name mental_health_hpo_production \
    --output results.json

# Best config
python -c "
import optuna, json
study = optuna.load_study(study_name='mental_health_hpo_production', storage='sqlite:///experiments/hpo_production.db')
with open('best_config.json', 'w') as f:
    json.dump(study.best_params, f, indent=2)
"
```

---

## 🛠️ Development

```bash
# Code quality
make format  # Format code
make lint    # Run linters
make check   # All checks

# Testing
make test           # All tests
make test-unit      # Unit only
make test-coverage  # With coverage

# Maintenance
make clean      # Remove caches
make clean-all  # Remove all (incl. experiments)

# Help
make help  # Show all commands
```

---

## 🐛 Troubleshooting

**tiktoken error:**
```bash
pip install tiktoken --upgrade
```

**CUDA OOM:**
- Reduce batch_size in search space
- Enable gradient checkpointing
- Use fp16/bf16

**Study not found:**
```bash
ls experiments/*.db  # Check DB exists
make hpo-results     # Verify study name
```

---

## 📈 Monitoring

### Real-time

```bash
# Trial logs
tail -f experiments/trial_*/logs/train.log

# GPU usage
watch -n 1 nvidia-smi

# Study progress
make hpo-results
```

### Analysis

```python
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

study = optuna.load_study(
    study_name='mental_health_hpo_production',
    storage='sqlite:///experiments/hpo_production.db'
)

# Plot history
fig = plot_optimization_history(study)
fig.write_html('history.html')

# Plot importances
fig = plot_param_importances(study)
fig.write_html('importances.html')
```

---

## 📚 Dataset

**REDSM5:** Mental Health Spanish Dataset
- Source: `irlab-udc/redsm5` (HuggingFace)
- Splits: train, validation, test
- Tasks: Criteria classification + evidence extraction

---

## ✅ Status

**Version:** 0.1.0
**Updated:** 2025-10-11
**Status:** ✅ Production Ready

**Verified:**
- ✅ Dependencies installed
- ✅ HPO running successfully
- ✅ Progress tracking working
- ✅ Checkpointing functional
- ✅ MLflow enabled
- ✅ Reproducible

---

## 📄 License

Academic research project. Cite appropriately if used.

---

**Quick Commands:**
```bash
make help           # Show all commands
make hpo-test       # Test HPO (3 trials)
make hpo            # Run HPO (50 trials)
make hpo-production # Production HPO (500 trials)
make mlflow-ui      # View results
make hpo-results    # CLI results
```
