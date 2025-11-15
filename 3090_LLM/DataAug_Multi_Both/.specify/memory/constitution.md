<!--
Sync Impact Report:
- Version change: Initial → 1.0.0
- Added sections: All core principles and governance
- Modified principles: N/A (initial creation)
- Templates requiring updates: ✅ All templates reviewed and aligned
- Follow-up TODOs: None
-->

# DataAug Multi Both HPO Constitution

## Core Principles

### I. Reproducibility-First (NON-NEGOTIABLE)
All experiments MUST be fully reproducible through deterministic seeding, configuration versioning, and environment isolation. Every training run MUST record exact dependency versions, random seeds, data splits, and hyperparameters. Container environments MUST pin all dependencies to exact versions via Poetry lock files. Configuration files MUST be version-controlled and immutable during execution.

**Rationale**: HPO experiments are only valuable if results can be reproduced and compared reliably across different machines and time periods.

### II. Storage-Optimized Artifact Management
Training pipelines MUST implement intelligent checkpoint retention policies. The default policy is to retain only the best model checkpoint for each trial. The system MUST proactively prune artifacts when the disk space on the project's SSD storage drops below 10% capacity. In such a low-storage scenario, the policy becomes more aggressive: only the single best model checkpoint for the entire HPO study is guaranteed to be retained. All metrics MUST remain accessible regardless of artifact pruning.

**Rationale**: Long-running HPO with 1000+ trials and multi-GB models requires aggressive storage management to prevent disk exhaustion while maintaining experiment continuity. Storage availability is critical for the continuation of the HPO process.

### III. Dual-Agent Architecture
The system MUST support two specialized agents sharing a common encoder: a criteria matching agent and an evidence binding agent. Both agents MUST be trainable simultaneously with shared representations. Evaluation MUST include agent-specific metrics: standard ML metrics (accuracy, AUC, F1, precision, recall) for criteria matching, plus exact match, has-answer score, and character-level F1/precision/recall for evidence binding.

**Evaluation Protocol**: For HPO studies, each trial MUST evaluate its best model on the validation set. For large-scale HPO studies (1000+ trials), test set evaluation SHOULD occur once per study (after all trials complete) rather than per trial, to prevent test set overfitting. The best model from the entire study MUST be evaluated on the held-out test set with results recorded in a study-level evaluation report.

**Rationale**: The dual-agent design with shared encoder enables efficient learning of complementary tasks while maintaining specialized evaluation criteria for each agent. Per-study test evaluation for large-scale HPO prevents overfitting to the test set across hundreds of trials.

### IV. MLflow-Centric Experiment Tracking
All training runs, hyperparameter trials, and evaluations MUST log to a local MLflow database within the project directory. Metrics, parameters, artifacts, and model references MUST be tracked continuously. The system MUST buffer metrics to disk during tracking outages and retry automatically with exponential backoff. Experiment metadata MUST survive artifact pruning operations.

**Rationale**: Centralized experiment tracking enables systematic comparison of HPO trials and provides audit trails for model development decisions.

### V. Auto-Resume Capability
Training pipelines MUST support automatic resumption from the latest valid checkpoint after interruption. Resume operations MUST not duplicate logged metrics or corrupt experiment state. The system MUST validate checkpoint integrity via checksum/hash before resuming and fall back gracefully to earlier checkpoints if corruption is detected. Checkpoints MUST be written atomically (temp file then rename) to prevent partial writes.

**Rationale**: Long-running HPO experiments are frequently interrupted by system maintenance, resource constraints, or failures, requiring robust resume capabilities.

### VI. Portable Development Environment
All training and evaluation MUST execute within containerized environments that work consistently across different machines. Container specifications MUST include all dependencies (managed via Poetry), accelerator access, and data mount requirements. Development containers MUST support interactive debugging and Jupyter notebooks for analysis.

**Rationale**: Consistent environments across workstations, servers, and cloud instances eliminate environment drift and enable seamless collaboration.

### VII. Makefile-Driven Operations
All common operations (start training, resume HPO, evaluate models, cleanup artifacts) MUST be accessible through simple Makefile commands. Commands MUST be self-documenting and handle environment setup automatically. Complex multi-step operations MUST be abstracted into single make targets.

**Rationale**: Simplified command interfaces reduce cognitive load and enable reliable automation of complex ML workflows.

## Technical Standards

### Dependency Management
All Python dependencies MUST be managed through Poetry with exact version pinning in poetry.lock. Container images MUST use multi-stage builds for optimization. Hugging Face model and dataset dependencies MUST specify exact revisions or commit hashes where possible.

### Code Quality
All code MUST pass ruff linting, black formatting, and mypy type checking. Test coverage MUST exceed 80% for core training and evaluation logic. Integration tests MUST validate end-to-end HPO workflows including resume scenarios.

### Data Handling
Training, validation, and test data MUST be loaded from Hugging Face datasets with explicit split specifications. Test set evaluation MUST occur only after training completion to prevent data leakage. Data loading MUST be deterministic and reproducible across runs.

## Development Workflow

### Configuration Management
All hyperparameters and training configurations MUST be managed through Hydra configuration files. Configuration schemas MUST be validated at runtime. Each HPO trial MUST save its complete resolved configuration alongside results.

### Evaluation Protocol
Each HPO trial MUST evaluate its best model on the validation set during training. For large-scale HPO studies (1000+ trials), test set evaluation MUST occur once per study after all trials complete, evaluating only the best model from the entire study. The study-level evaluation report MUST contain test metrics, configuration, and model checkpoint references. For smaller HPO studies (<100 trials), per-trial test evaluation MAY be used if test set overfitting risk is acceptable. When multiple checkpoints tie for best validation performance, all tied checkpoints should be evaluated. However, their preservation is subject to the storage optimization policy (Principle II), which may prune them to ensure training continuity.

### Cleanup and Maintenance
Automated cleanup procedures MUST remove obsolete checkpoints while preserving experiment metadata. Storage monitoring MUST warn when approaching capacity limits. Log rotation MUST prevent unbounded log file growth.

## Governance

This constitution supersedes all other development practices and guidelines. All code changes MUST comply with these principles. Violations require explicit justification and approval through the complexity tracking process defined in plan templates.

Amendments to this constitution require documentation of rationale, impact analysis, and migration plan for existing code. Version increments follow semantic versioning: MAJOR for backward-incompatible principle changes, MINOR for new principles or expanded guidance, PATCH for clarifications and refinements.

All development decisions MUST be traceable to constitutional principles. When principles conflict, Storage Optimization (Principle II) takes precedence over all others to ensure experiments can continue. In a conflict between storage and reproducibility, storage optimization is prioritized. Reproducibility guarantees are then focused on keeping the single best model of the HPO study, including its checkpoint, configuration, and test metrics.

**Version**: 1.1.0 | **Ratified**: 2025-10-10 | **Last Amended**: 2025-10-10

**Changelog**:
- v1.1.0 (2025-10-10): Amended Principle III and Evaluation Protocol to allow per-study test evaluation for large-scale HPO (1000+ trials) to prevent test set overfitting. Hybrid approach: per-trial validation evaluation, per-study test evaluation.