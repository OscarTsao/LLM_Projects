# Refactored_Psy Directory Exploration Report

**Generated:** 2025-10-23
**Location:** /experiment/YuNing/Refactored_Psy

---

## Executive Summary

The `/experiment/YuNing/Refactored_Psy` directory contains a well-organized collection of psychiatric symptom classification projects with two primary target directories (`NoAug_Criteria_Evidence` and `DataAug_Criteria_Evidence`) and fourteen source repositories in the `psy-ref-repos` subdirectory. Total size: **198.6 MB**

---

## Directory Structure Overview

```
/experiment/YuNing/Refactored_Psy/
├── NoAug_Criteria_Evidence/          (8.3 MB) [TARGET - No Data Augmentation]
├── DataAug_Criteria_Evidence/        (8.3 MB) [TARGET - With Data Augmentation]
└── psy-ref-repos/                    (182 MB) [SOURCE REPOSITORIES]
    ├── DataAugmentation_ReDSM5/      (9.0 MB)
    ├── DataAug_Multi_Both/           (32 MB)
    ├── Criteria_Evidence_Agent/      (11 MB)
    ├── Psy_RAG/                      (22 MB)
    ├── NoAug_Criteria_Evidence/      (332 KB)
    └── [9 other repositories]
```

---

## TARGET DIRECTORIES ANALYSIS

### 1. NoAug_Criteria_Evidence (8.3 MB)

**Status:** Ready with Git history  
**Last Commit:** 7562c21 (Initial commit)

#### Structure:
```
NoAug_Criteria_Evidence/
├── pyproject.toml              [Template-based, needs customization]
├── README.md                   [Generic ML template]
├── src/
│   └── Project/
│       ├── Criteria/           [Criteria matching task]
│       ├── Evidence/           [Evidence binding task]
│       ├── Joint/              [Combined training]
│       └── Share/              [Shared utilities]
│           ├── models/model.py
│           ├── data/dataset.py
│           ├── utils/
│           │   ├── checkpoint.py
│           │   ├── mlflow_utils.py
│           │   ├── optuna_utils.py
│           │   ├── log.py
│           │   └── seed.py
│           └── engine/
│               ├── train_engine.py
│               └── eval_engine.py
├── configs/                    [Hydra configuration]
│   ├── criteria/
│   ├── evidence/
│   ├── joint/
│   ├── share/
│   └── data/
├── data/
│   ├── redsm5/
│   │   ├── redsm5_posts.csv    (2,125 lines)
│   │   ├── redsm5_annotations.csv (2,082 lines)
│   │   └── README.md
│   └── processed/
│       └── redsm5_matched_evidence.csv
├── artifacts/                  [Empty - for models/weights]
├── outputs/                    [Empty - for training artifacts]
├── mlruns/                     [MLflow tracking]
├── tests/                      [Test directory]
└── .devcontainer/              [VS Code Dev Container]
```

#### Data Files:
- ReDSM-5 dataset (posts and annotations)
- Processed evidence data
- No DSM-5 criteria JSON files present yet

#### Configuration:
- Uses Hydra for configuration management
- MLflow/Optuna ready
- Uses setuptools for packaging
- Python 3.10+ required

---

### 2. DataAug_Criteria_Evidence (8.3 MB)

**Status:** Ready with Git history  
**Last Commit:** 7562c21 (Initial commit)

#### Structure:
Identical to NoAug_Criteria_Evidence, including:
- Same Project structure (Criteria, Evidence, Joint, Share)
- Same data files (ReDSM-5 dataset + processed evidence)
- Same configuration structure
- Separated for augmented vs non-augmented experiments

#### Differences from NoAug:
- Intended for augmented data training pipeline
- Otherwise structurally identical (ready for modifications)

---

## SOURCE REPOSITORIES ANALYSIS

### Requested Components Present

#### 1. NoAug_Criteria_Evidence (Source)
**Location:** `/experiment/YuNing/Refactored_Psy/psy-ref-repos/NoAug_Criteria_Evidence/`
**Size:** 332 KB

- Basic criteria evidence agent structure
- Data directory (empty - data in target)
- Configs, artifacts, outputs directories

---

#### 2. DataAugmentation_ReDSM5
**Location:** `/experiment/YuNing/Refactored_Psy/psy-ref-repos/DataAugmentation_ReDSM5/`
**Size:** 9.0 MB

**Key Components:**

**Data Files:**
- `/Data/ReDSM5/redsm5_posts.csv` (2,125 lines)
- `/Data/ReDSM5/redsm5_annotations.csv` (2,082 lines)
- `/Data/GroundTruth/Final_Ground_Truth.json` (48,973 lines, 3.2 MB)
- `/Data/Augmentation/` (empty - for augmented data)

**Source Code Structure:**
```
src/
├── agents/              [Multi-agent architecture]
│   ├── base.py
│   ├── criteria_matching.py
│   ├── evidence_binding.py
│   └── multi_agent_pipeline.py
├── augmentation/        [Data augmentation pipelines]
│   ├── base.py
│   ├── nlpaug_pipeline.py
│   ├── textattack_pipeline.py
│   └── hybrid_pipeline.py
├── data/               [Data loaders and processing]
│   ├── redsm5_loader.py
│   ├── criteria_descriptions.py
│   ├── evidence_loader.py
│   └── joint_dataset.py
├── training/          [Training and evaluation]
│   ├── train.py
│   ├── train_criteria.py
│   ├── train_evidence.py
│   ├── train_joint.py
│   ├── train_optuna.py
│   ├── evaluate.py
│   ├── evaluate_criteria.py
│   ├── evaluate_evidence.py
│   ├── evaluate_joint.py
│   ├── data_module.py
│   ├── dataset_builder.py
│   ├── modeling.py
│   └── engine.py
└── utils/             [Utilities]
```

**Configuration (Hydra):**
```
conf/
├── config.yaml         [Main configuration]
├── model/              [Model configs: BERT, RoBERTa, DeBERTa]
│   ├── bert_base.yaml
│   ├── bert_base_rtx5090.yaml
│   ├── roberta_base.yaml
│   ├── deberta_base.yaml
│   └── [GPU-specific variants]
├── dataset/            [Data augmentation modes]
│   ├── original.yaml
│   ├── original_nlpaug.yaml
│   ├── original_textattack.yaml
│   ├── original_hybrid.yaml
│   └── original_nlpaug_textattack.yaml
├── agent/              [Agent configs]
│   ├── criteria.yaml
│   ├── evidence.yaml
│   └── joint.yaml
└── training_mode/      [Training modes]
    ├── criteria.yaml
    ├── evidence.yaml
    └── joint.yaml
```

**Key Features:**
- Multiple augmentation strategies (NLPAug, TextAttack, Hybrid)
- Multi-agent pipeline for criteria matching and evidence binding
- Hyperparameter optimization with Optuna
- MLflow experiment tracking
- Support for BERT, RoBERTa, DeBERTa models
- Mixed precision training (AMP)
- Group-based train/val/test splitting

**Tests:**
- `/tests/agents/test_criteria_matching.py`
- `/tests/agents/test_multi_agent_pipeline.py`
- `/tests/training/test_dataset_builder.py`
- `/tests/utils/test_timestamp.py`

---

#### 3. DataAug_Multi_Both
**Location:** `/experiment/YuNing/Refactored_Psy/psy-ref-repos/DataAug_Multi_Both/`
**Size:** 32 MB

**Key Components:**

**Source Code Structure:**
```
src/dataaug_multi_both/
├── cli/                [Command-line interface]
├── data/               [Data augmentation module]
│   └── augmentation.py
├── hpo/                [Hyperparameter optimization]
├── training/           [Training module]
├── utils/              [Utilities]
└── __init__.py
```

**Configuration:**
- Uses Poetry for dependency management
- Has `poetry.lock` for reproducibility
- Hydra configuration support
- Optuna HPO integration
- MLflow experiment tracking

**Key Features:**
- Data augmentation pipelines
- Hyperparameter optimization with Optuna
- Deep learning training with PyTorch
- Configuration management with Hydra

---

#### 4. Criteria_Evidence_Agent
**Location:** `/experiment/YuNing/Refactored_Psy/psy-ref-repos/Criteria_Evidence_Agent/`
**Size:** 11 MB

**Key Components:**

**Data Files:**
- `/Data/DSM-5/DSM_Criteria_Array_Fixed.json` (42 KB)
- `/Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json` (27 KB)
- `/Data/DSM-5/single_disorder_dsm.json` (1.7 KB)
- `/Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json` (1.9 KB)
- `/Data/groundtruth/redsm5_ground_truth.json`
- `/Data/redsm5/` (ReDSM-5 dataset)

**Source Code Structure:**
```
src/
├── data/           [Data handling]
├── models/         [Model implementations]
├── utils/          [Utilities]
└── [other modules]
```

**Configuration:**
- Uses pyproject.toml (setuptools)
- Makefile-based operations
- Comprehensive documentation (AGENTS.md, CLAUDE.md, README.md)
- Metrics tracking (metrics.json)

**Key Files:**
- `AGENTS.md` - Multi-agent system description
- `OPTIMIZATION_REPORT.md` - Optimization results
- `CLAUDE.md` - Development guidelines
- `metrics.json` - Performance metrics

---

#### 5. Psy_RAG
**Location:** `/experiment/YuNing/Refactored_Psy/psy-ref-repos/Psy_RAG/`
**Size:** 22 MB

**Key Components:**

**Configuration:**
- Poetry-based project
- `poetry.lock` for dependency management
- Complex specs structure (`.specify/`, `.claude/`, `.codex/`, `.gemini/`)
- GitHub workflows support (`.github/`)

**Source Code Structure:**
```
src/
├── config/         [Configuration modules]
├── models/         [Model implementations]
└── utils/          [Utilities]
```

**Data Files:**
- `/Data/DSM-5/` - DSM-5 criteria JSON files

**Key Features:**
- Retrieval-Augmented Generation (RAG) implementation
- Hypothesis testing support
- Make-based operations
- Extensive documentation

---

### Other Source Repositories

**Additional repositories in psy-ref-repos:**

1. **DataAug_DeBERTa_Criteria** (38 MB)
   - DeBERTa model variant for criteria matching
   - Specs and extensive configuration
   - Complete DSM-5 criteria definitions

2. **DataAug_DeBERTa_Evidence** (32 MB)
   - DeBERTa model variant for evidence binding
   - Comprehensive test suite
   - Multiple specification documents

3. **DataAug_DeBERTa_FourAgents** (928 KB)
   - Four-agent system architecture
   - Schema definitions (criteria.schema.json, predictions.schema.json)
   - Specification tooling

4. **Criteria_Agent_Training** (34 MB)
   - Training pipeline for criteria matching
   - Complete DSM-5 dataset
   - Makefile-based workflow

5. **DataAug_Multi_Evidence** (416 KB)
   - Multi-agent evidence binding
   - Lightweight configuration

6. **Psy_redsm5_Criteria_Evidence_Agent** (4.6 MB)
   - Combined criteria and evidence agent
   - ReDSM-5 integration

7. **Psy_RAG_Agent** (120 KB)
   - RAG agent wrapper
   - Lightweight implementation

8. **DataAugmentation_Evaluation** (120 KB)
   - Evaluation utilities
   - Metrics calculation

---

## KEY COMPONENTS INVENTORY

### Data Assets

**ReDSM-5 Dataset:**
- Posts: `redsm5_posts.csv` (2,125 lines, ~2.3 MB)
- Annotations: `redsm5_annotations.csv` (2,082 lines, ~951 KB)
- Ground Truth: `Final_Ground_Truth.json` (48,973 lines, 3.2 MB)
- Location: Multiple repos have copies for redundancy

**DSM-5 Criteria Files:**
```
DSM_Criteria_Array_Fixed.json                 (Full criteria with all disorders)
DSM_Criteria_Array_Fixed_Simplify.json        (Simplified criteria)
DSM_Criteria_Array_Fixed_Major_Depressive.json (Major Depressive Disorder only)
single_disorder_dsm.json                      (Single disorder template)
```
- Available in: 7 different source repositories
- Total: ~80 KB of DSM-5 definitions

**Processed Data:**
- `redsm5_matched_evidence.csv` (in target directories)

---

### Model Implementations

**Base Classes:**
- `BaseAgent` (abstract)
- Agent output and configuration dataclasses

**Agent Types:**
1. **CriteriaMatchingAgent** - Maps evidence to DSM-5 criteria
2. **EvidenceBindingAgent** - Binds relevant evidence snippets
3. **JointTrainingModel** - Combined training for both tasks
4. **MultiAgentPipeline** - Orchestrates multiple agents

**Model Architectures Supported:**
- BERT (base, from HuggingFace)
- RoBERTa (base)
- DeBERTa (base, with GPU-specific configs)
- Each with standard and RTX5090-specific variants

---

### Augmentation Pipelines

**Pipeline Types:**
1. **NLPAug Pipeline** (`nlpaug_pipeline.py`)
   - Synonym replacement
   - Contextual word embeddings

2. **TextAttack Pipeline** (`textattack_pipeline.py`)
   - Back-translation
   - Paraphrase generation

3. **Hybrid Pipeline** (`hybrid_pipeline.py`)
   - Combines NLPAug + TextAttack
   - Configurable strategy selection

**Base Infrastructure:**
- `AugmentationConfig` dataclass
- Timestamp utilities
- Comparison metrics (SequenceMatcher)

---

### Configuration System

**Framework:** Hydra
- Hierarchical YAML configurations
- Command-line overrides
- Multi-run support
- Experiment tracking integration

**Configuration Categories:**
- **Dataset configs** - Data augmentation modes
- **Model configs** - Architecture and hyperparameter choices
- **Agent configs** - Agent-specific settings
- **Training mode configs** - Task-specific configurations

---

### Utilities and Infrastructure

**Checkpoint Management:**
- Saving/loading model states
- Resume capability

**MLflow Integration:**
- Automatic experiment tracking
- Local and remote backend support
- Autologging for PyTorch models
- Tag and parameter management

**Optuna Integration:**
- Hyperparameter optimization
- Study persistence
- Trial management utilities

**Logging and Monitoring:**
- Structured logging
- Seed management for reproducibility
- MLflow metric tracking

---

### Testing Infrastructure

**Test Directories:**
- Unit tests in `/tests/` subdirectories
- Test coverage for:
  - Multi-agent pipeline
  - Criteria matching agent
  - Dataset building
  - Utility functions (timestamp, etc.)

---

## DEPENDENCY ECOSYSTEM

### Python Version
- Minimum: 3.10
- Recommended: 3.10+

### Core Dependencies

**Deep Learning:**
- `torch >= 2.2.0`
- `transformers >= 4.40`

**Experiment Management:**
- `mlflow >= 2.8`
- `optuna >= 3.4` (HPO)
- `hydra-core >= 1.3`

**Data Processing:**
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`

**ML Utilities:**
- `scikit-learn >= 1.3.0`
- `pyyaml >= 6.0.1`

**Text Augmentation:**
- `nlpaug` (for NLPAug pipeline)
- `textattack` (for TextAttack pipeline)

### Development Dependencies
- `pytest >= 7.4`
- `ruff >= 0.6`
- `black >= 24.3`
- `mypy >= 1.8`

### Package Management
- **Setuptools** - Used in target directories
- **Poetry** - Used in some source repos (DataAug_Multi_Both, DataAug_DeBERTa_*, Psy_RAG)

---

## WHAT EXISTS vs WHAT NEEDS CREATION

### EXISTS:
1. ✅ **Target Directories** - Both NoAug and DataAug targets exist with basic structure
2. ✅ **Source Repositories** - 14 reference repos with implementations
3. ✅ **ReDSM-5 Dataset** - Raw posts and annotations present in multiple locations
4. ✅ **Ground Truth Data** - Final_Ground_Truth.json available
5. ✅ **DSM-5 Criteria JSON** - Complete criteria definitions available
6. ✅ **Augmentation Pipelines** - 3 strategies implemented (NLPAug, TextAttack, Hybrid)
7. ✅ **Agent Implementations** - Multi-agent system with criteria matching and evidence binding
8. ✅ **Training Infrastructure** - Hydra configs, MLflow, Optuna integration
9. ✅ **Data Loading** - ReDSM-5 loader utilities present
10. ✅ **Configuration Management** - Comprehensive Hydra setup

### NEEDS MIGRATION/CUSTOMIZATION:
1. ⚠️ **Target pyproject.toml** - Template needs customization with actual dependencies
2. ⚠️ **DSM-5 Criteria JSON** - Need to copy to target directories if not already present
3. ⚠️ **Augmentation Code** - ReDSM5 augmentation modules need integration into targets
4. ⚠️ **Agent Code** - Agent implementations need integration/refactoring into targets
5. ⚠️ **Training Scripts** - Need to create/port training entry points
6. ⚠️ **Evaluation Scripts** - Need to create/port evaluation scripts
7. ⚠️ **Target Configs** - Hydra configs may need adaptation
8. ⚠️ **README Files** - Target READMEs are template-based, need customization

### ARCHITECTURE DECISIONS NEEDED:
1. Which augmentation pipeline(s) to include in DataAug target?
2. How to unify multi-agent implementations across target and source?
3. Whether to consolidate duplicate DSM-5 criteria files
4. How to structure the Share/utils modules
5. Training entry point design (CLI vs scripts vs notebooks)
6. Whether to keep both targets or merge with configuration

---

## REPOSITORY METADATA

### Git Status:
- **NoAug_Criteria_Evidence:** Git initialized, has .git directory, 1 initial commit
- **DataAug_Criteria_Evidence:** Git initialized, has .git directory, 1 initial commit
- **Source repositories:** All have .git directories with full histories

### MLflow Database:
- Located in both targets: `mlflow.db` (SQLite)
- `mlruns/` directory for local tracking

### Optuna Database:
- Located in both targets: `optuna.db` (SQLite)

### Development Containers:
- `.devcontainer/` in all targets and several source repos
- `devcontainer.json` configuration present

---

## FILE SIZE BREAKDOWN

```
Total: 198.6 MB

By Directory:
- psy-ref-repos/                182 MB (91.8%)
  - DataAug_DeBERTa_Criteria       38 MB
  - DataAug_Multi_Both            32 MB
  - DataAug_DeBERTa_Evidence      32 MB
  - Criteria_Agent_Training       34 MB
  - DataAugmentation_ReDSM5        9.0 MB
  - Psy_RAG                        22 MB
  - Psy_redsm5_Criteria_Evidence   4.6 MB
  - [Others]                       ~10.4 MB

- DataAug_Criteria_Evidence      8.3 MB (4.2%)
- NoAug_Criteria_Evidence        8.3 MB (4.2%)
```

---

## CONFIGURATION HIGHLIGHTS

### Main Hydra Config (config.yaml):
```yaml
defaults:
  - dataset: original
  - model: bert_base
  - optional training_mode: null

seed: 1337
output_dir: outputs/train
log_dir: logs
resume: true
save_total_limit: 1
metric_for_best_model: roc_auc
regression_threshold: 0.5

dataloader:
  num_workers: 0
  pin_memory: false
  persistent_workers: false

metrics: [accuracy, precision, recall, f1, roc_auc]

mlflow:
  enabled: true
  tracking_uri: sqlite:///mlflow.db
  experiments:
    training: redsm5-classification
    optuna: redsm5-optuna

n_trials: 500
```

---

## RECOMMENDATIONS

### For Full Integration:
1. **Copy DSM-5 Criteria** to target Data directories
2. **Port Augmentation Pipelines** to target src/Project/Share/augmentation/
3. **Port Agent Implementations** to target src/Project/
4. **Consolidate Configurations** - Merge best practices from all source repos
5. **Create Unified Training Scripts** - Entry points for all modes (criteria, evidence, joint)
6. **Unify Dependencies** - Consolidate pyproject.toml/poetry.lock files
7. **Document Migration** - Create migration guide for teams using old repos
8. **Establish CI/CD** - Use GitHub Actions if available
9. **Standardize Testing** - Ensure pytest configuration is identical across targets

### For Operational Clarity:
1. **Remove Duplicate Data** - Consolidate ReDSM-5 dataset copies
2. **Archive Old Repos** - Ensure psy-ref-repos is for reference only
3. **Create ARCHITECTURE.md** - Document component interactions
4. **Establish Code Ownership** - Clear responsibility for each module
5. **Setup MLflow Server** - Configure proper experiment tracking

