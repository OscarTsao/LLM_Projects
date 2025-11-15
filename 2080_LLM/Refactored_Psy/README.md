# Refactored_Psy - Psychiatric Symptom Classification Project

## Overview

This directory contains a comprehensive psychiatric symptom classification system with two production-ready target directories and 14 reference source repositories. The project focuses on using transformer-based models (BERT, RoBERTa, DeBERTa) with multi-agent architectures for:

1. **Criteria Matching** - Mapping psychiatric evidence to DSM-5 diagnostic criteria
2. **Evidence Binding** - Identifying relevant evidence snippets from text
3. **Joint Training** - Combined multi-task learning approaches

**Total Project Size:** 198.6 MB | **Python Version:** 3.10+ | **Last Updated:** 2025-10-23

---

## Quick Navigation

### Production Ready Targets
- **NoAug_Criteria_Evidence/** - Criteria + Evidence classification without data augmentation
- **DataAug_Criteria_Evidence/** - Criteria + Evidence classification with data augmentation

### Documentation (START HERE)
1. **EXPLORATION_REPORT.md** - Comprehensive 400+ line analysis of entire directory structure
2. **QUICK_REFERENCE.md** - Quick lookup guide for file paths and components
3. **FILE_INVENTORY.txt** - Detailed inventory of all files and dependencies
4. **README.md** - This file

### Source Reference Repositories
Located in `psy-ref-repos/` - 14 repositories containing:
- Complete augmentation implementations
- Multi-agent architectures
- DSM-5 criteria definitions
- Training pipelines
- Configuration examples

---

## Key Components

### What Exists

✅ **Data Assets**
- ReDSM-5 dataset (2,125 posts, 2,082 annotations)
- Ground truth labels (Final_Ground_Truth.json, 3.2 MB)
- DSM-5 criteria definitions (4 variants, 80 KB total)
- Processed evidence data

✅ **Code Components**
- Multi-agent system (CriteriaMatchingAgent, EvidenceBindingAgent, JointTrainingModel)
- Augmentation pipelines (NLPAug, TextAttack, Hybrid)
- Data loaders and utilities
- Training and evaluation engines
- MLflow and Optuna integration

✅ **Infrastructure**
- Hydra configuration framework (40+ YAML configs)
- MLflow experiment tracking
- Optuna hyperparameter optimization
- PyTorch training infrastructure
- Dev containers for VS Code

### What Needs Integration

⚠️ **Target Directories** need:
1. Copy DSM-5 criteria JSON files to `data/dsm5/`
2. Port augmentation code to `src/Project/Share/augmentation/`
3. Port agent implementations to `src/Project/agents/`
4. Port training scripts to `src/`
5. Copy Hydra configurations to `configs/`
6. Update `pyproject.toml` with actual dependencies
7. Port test suite from source repos
8. Update README with project-specific documentation

---

## File Structure

```
Refactored_Psy/
├── NoAug_Criteria_Evidence/          [8.3 MB] Target - No augmentation
│   ├── pyproject.toml
│   ├── src/Project/
│   │   ├── Criteria/                 Criteria matching models
│   │   ├── Evidence/                 Evidence binding models
│   │   ├── Joint/                    Joint training models
│   │   └── Share/                    Shared utilities
│   ├── data/redsm5/                  ReDSM-5 dataset
│   ├── configs/                      Hydra configurations
│   └── ...
│
├── DataAug_Criteria_Evidence/        [8.3 MB] Target - With augmentation
│   └── [Same structure as NoAug]
│
├── psy-ref-repos/                    [182 MB] Source Repositories
│   ├── DataAugmentation_ReDSM5/      [9.0 MB] PRIMARY - Augmentation & agents
│   ├── Criteria_Evidence_Agent/      [11 MB]  PRIMARY - DSM-5 criteria
│   ├── DataAug_Multi_Both/           [32 MB]  Implementation reference
│   ├── Psy_RAG/                      [22 MB]  RAG approach
│   └── [10 other supporting repos]
│
└── Documentation
    ├── EXPLORATION_REPORT.md         Detailed analysis (START HERE)
    ├── QUICK_REFERENCE.md            Quick lookup guide
    ├── FILE_INVENTORY.txt            Complete file listing
    └── README.md                      This file
```

---

## Critical Data Files

| File | Location | Size | Purpose |
|------|----------|------|---------|
| redsm5_posts.csv | Multiple locations | 2.3 MB | Reddit posts (2,125 samples) |
| redsm5_annotations.csv | Multiple locations | 951 KB | DSM-5 annotations (2,082 samples) |
| Final_Ground_Truth.json | DataAugmentation_ReDSM5/Data/GroundTruth/ | 3.2 MB | 48,973 comprehensive labels |
| DSM_Criteria_Array_Fixed.json | 7 source repos | 42 KB | Complete disorder criteria |
| DSM_Criteria_Array_Fixed_Simplify.json | 7 source repos | 27 KB | Simplified criteria |

---

## Critical Source Files

### For Augmentation Integration
```
Source: /psy-ref-repos/DataAugmentation_ReDSM5/src/augmentation/
Files:
- base.py                     Base augmentation class
- nlpaug_pipeline.py          Synonym + contextual replacement
- textattack_pipeline.py      Back-translation + paraphrase
- hybrid_pipeline.py          Combined approach
```

### For Agent Integration
```
Source: /psy-ref-repos/DataAugmentation_ReDSM5/src/agents/
Files:
- base.py                     BaseAgent class
- criteria_matching.py        Criteria matching agent
- evidence_binding.py         Evidence binding agent
- multi_agent_pipeline.py     Multi-agent orchestration
```

### For Configuration
```
Source: /psy-ref-repos/DataAugmentation_ReDSM5/conf/
Files: 24 YAML files across model/, dataset/, agent/, training_mode/ directories
```

### For DSM-5 Criteria
```
Source: /psy-ref-repos/Criteria_Evidence_Agent/Data/DSM-5/
Files: 4 JSON files (fixed, simplified, specific, template)
```

---

## Dependencies

**Core:**
- torch >= 2.2.0
- transformers >= 4.40
- mlflow >= 2.8
- optuna >= 3.4
- hydra-core >= 1.3
- pandas >= 2.0.0
- numpy >= 1.24.0

**Augmentation:**
- nlpaug (NLPAug pipeline)
- textattack (TextAttack pipeline)

**Development:**
- pytest >= 7.4
- ruff >= 0.6
- black >= 24.3
- mypy >= 1.8

See `EXPLORATION_REPORT.md` for complete dependency list.

---

## Getting Started

### For Exploration
1. Read `EXPLORATION_REPORT.md` for comprehensive overview
2. Use `QUICK_REFERENCE.md` for quick lookups
3. Check `FILE_INVENTORY.txt` for detailed file counts

### For Integration (To make targets production-ready)
1. Copy DSM-5 criteria from `psy-ref-repos/Criteria_Evidence_Agent/Data/DSM-5/` to target `data/dsm5/`
2. Port augmentation code from `psy-ref-repos/DataAugmentation_ReDSM5/src/augmentation/` to target `src/Project/Share/augmentation/`
3. Port agents from `psy-ref-repos/DataAugmentation_ReDSM5/src/agents/` to target `src/Project/agents/`
4. Port training scripts from `psy-ref-repos/DataAugmentation_ReDSM5/src/training/` to target `src/`
5. Copy Hydra configs from `psy-ref-repos/DataAugmentation_ReDSM5/conf/` to target `configs/`
6. Update `pyproject.toml` with actual dependencies

### For Development
```bash
cd NoAug_Criteria_Evidence
python -m pip install -e '.[dev]'
# Then port code from source repos and update imports
```

---

## Multi-Agent Architecture

The system uses a multi-agent approach:

```
Input Text (Reddit Post)
    ↓
[Criteria Matching Agent]  → DSM-5 Criteria (Y/N for each)
    ↓
[Evidence Binding Agent]   → Evidence snippets + confidence
    ↓
[Joint Training Model]     → Combined predictions
```

- **CriteriaMatchingAgent**: Maps text to DSM-5 diagnostic criteria
- **EvidenceBindingAgent**: Identifies supporting evidence for criteria
- **MultiAgentPipeline**: Orchestrates both agents
- **JointTrainingModel**: Combined end-to-end training

---

## Augmentation Strategies

Three data augmentation pipelines available:

1. **NLPAug Pipeline**
   - Synonym replacement (WordNet)
   - Contextual word embeddings
   
2. **TextAttack Pipeline**
   - Back-translation
   - Paraphrase generation
   
3. **Hybrid Pipeline**
   - Combines NLPAug + TextAttack
   - Configurable strategy selection

---

## Configuration System

Uses Hydra with hierarchical YAML configs:

```yaml
defaults:
  - dataset: original|original_nlpaug|original_textattack|original_hybrid
  - model: bert_base|roberta_base|deberta_base
  - optional training_mode: criteria|evidence|joint
```

All configs modifiable via command line:
```bash
python train.py model=bert_base training.batch_size=32
```

---

## Key Source Repositories

| Repo | Size | Purpose | Key Files |
|------|------|---------|-----------|
| DataAugmentation_ReDSM5 | 9.0 MB | CRITICAL - Augmentation + agents | src/augmentation/, src/agents/, conf/ |
| Criteria_Evidence_Agent | 11 MB | CRITICAL - DSM-5 criteria | Data/DSM-5/*.json |
| DataAug_Multi_Both | 32 MB | Implementation reference | src/dataaug_multi_both/ |
| Psy_RAG | 22 MB | RAG approach | src/config/, src/models/ |
| DataAug_DeBERTa_Criteria | 38 MB | DeBERTa variant | configs/, Data/DSM-5/ |
| DataAug_DeBERTa_Evidence | 32 MB | DeBERTa variant | configs/, tests/ |
| Criteria_Agent_Training | 34 MB | Training pipeline | Makefile, Data/DSM-5/ |

See `QUICK_REFERENCE.md` for details on other 7 repositories.

---

## Documentation Files

- **EXPLORATION_REPORT.md** (400+ lines)
  - Complete directory analysis
  - Component inventory
  - Dependency ecosystem
  - Integration recommendations

- **QUICK_REFERENCE.md** (200+ lines)
  - Critical file paths
  - Component dependencies
  - Migration checklist
  - Quick commands

- **FILE_INVENTORY.txt** (400+ lines)
  - Complete file listing
  - Size breakdown
  - What exists vs. needs integration
  - Critical integration paths

- **README.md** (This file)
  - Project overview
  - Quick navigation
  - Getting started

---

## Infrastructure Details

### Databases
- **MLflow**: `*/mlflow.db` (SQLite) - Experiment tracking
- **Optuna**: `*/optuna.db` (SQLite) - Hyperparameter optimization

### Development Containers
- VS Code dev container configs present
- `.devcontainer/devcontainer.json` in targets and source repos

### Git Repositories
- Both targets: Git initialized, 1 initial commit
- All source repos: Full commit histories present

---

## Troubleshooting & FAQ

**Q: Where should I start?**
A: Read `EXPLORATION_REPORT.md` first, then use `QUICK_REFERENCE.md` for lookups.

**Q: How do I make targets production-ready?**
A: Follow the migration checklist in `QUICK_REFERENCE.md` and `FILE_INVENTORY.txt`.

**Q: Where are the augmentation pipelines?**
A: In `psy-ref-repos/DataAugmentation_ReDSM5/src/augmentation/` (NLPAug, TextAttack, Hybrid).

**Q: Where are the DSM-5 criteria?**
A: In `psy-ref-repos/Criteria_Evidence_Agent/Data/DSM-5/` (4 JSON files).

**Q: How many models are supported?**
A: BERT, RoBERTa, DeBERTa (each with CPU and RTX5090 variants).

**Q: What's the difference between NoAug and DataAug targets?**
A: NoAug uses original data only, DataAug includes augmented training data.

---

## Key Statistics

| Metric | Value |
|--------|-------|
| Total Size | 198.6 MB |
| Target Directories | 2 |
| Source Repositories | 14 |
| Python Files | 150+ |
| YAML Config Files | 40+ |
| JSON Data Files | 10+ |
| ReDSM-5 Posts | 2,125 |
| ReDSM-5 Annotations | 2,082 |
| DSM-5 Disorders | 100+ |
| Agents | 5 types |
| Augmentation Pipelines | 3 strategies |
| Model Architectures | 3 (BERT, RoBERTa, DeBERTa) |

---

## Next Steps

1. **Review Documentation**
   - Start with `EXPLORATION_REPORT.md`
   - Use `QUICK_REFERENCE.md` for lookups
   - Check `FILE_INVENTORY.txt` for details

2. **Understand Architecture**
   - Study `psy-ref-repos/Criteria_Evidence_Agent/CLAUDE.md` for development patterns
   - Review `psy-ref-repos/Criteria_Evidence_Agent/AGENTS.md` for agent architecture
   - Check `psy-ref-repos/DataAugmentation_ReDSM5/README.md` for augmentation details

3. **Begin Integration**
   - Copy DSM-5 criteria to targets
   - Port augmentation code
   - Port agent implementations
   - Port training scripts

4. **Test & Validate**
   - Run pytest suite
   - Validate configurations
   - Test augmentation pipelines
   - Verify multi-agent system

---

## Contact & Support

For specific questions about components:
- Augmentation: See `psy-ref-repos/DataAugmentation_ReDSM5/README.md`
- Agents: See `psy-ref-repos/Criteria_Evidence_Agent/CLAUDE.md`
- Training: See `psy-ref-repos/DataAug_Multi_Both/`
- Configuration: See all `*/conf/` directories

---

**Last Updated:** 2025-10-23  
**Directory:** /experiment/YuNing/Refactored_Psy  
**Documentation Status:** Complete

