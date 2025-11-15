# Feature Specification: BERT Training Pipeline for Evidence Sentence Classification

**Feature Branch**: `002-bert-training-pipeline`
**Created**: 2025-11-12
**Status**: Draft
**Input**: Implement complete BERT-based binary classification training pipeline for DSM-5 evidence sentence detection with Hydra configuration, MLflow tracking, and reproducible experiments

## Clarifications

### Session 2025-11-12

- Q: Classification task architecture - train 9 separate models, one unified model with criterion input, multi-label classifier, or grouped models? → A: One unified binary classifier with criterion text as input (NSP format: `[CLS] <criterion> [SEP] <sentence> [SEP]`). The model learns to classify any criterion-sentence pair.
- Q: Dataset pairing strategy - pair every sentence with all 9 criteria, only annotated combinations, only positives, or stratified sampling? → A: Only pair annotated sentence-symptom combinations. Use the existing annotations file to determine which criterion to pair with which sentence (includes both status=1 and status=0 examples).
- Q: Data splitting strategy - split by posts, by sentences, by symptom type, or k-fold CV? → A: Split by posts (70/15/15 train/val/test). All sentences from the same post stay together in the same split to prevent data leakage. Use stratified splitting to maintain label balance across splits.
- Q: Class imbalance handling - no special handling, weighted loss, oversampling, or focal loss? → A: Hybrid approach using weighted focal loss. Combines focal loss (down-weights easy examples, focuses on hard ones) with class weights (inversely proportional to class frequencies) to handle both imbalance and hard example learning.
- Q: Training performance optimizations - none, core optimizations, advanced optimizations, or full suite? → A: Full optimization suite from Optimization_Examples. Implement: mixed precision (bf16/fp16 with autocast), TF32, SDPA/FlashAttention, gradient checkpointing, fused AdamW, torch.compile, CUDA graphs (if shapes static), pinned memory DataLoader, gradient accumulation if needed.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Train Binary Classifier for Evidence Detection (Priority: P1)

As a researcher, I want to train a BERT-based binary classifier on DSM-5 criterion-sentence pairs so that I can identify which Reddit post sentences contain evidence for specific depression symptoms.

**Why this priority**: Core functionality required for the entire project. Without this, no model can be trained or evaluated.

**Independent Test**: Can be fully tested by running a training command with a small subset of data and verifying a model is saved to MLflow with logged metrics (loss, accuracy).

**Acceptance Scenarios**:

1. **Given** ReDSM5 dataset and DSM-5 criteria files exist in `data/`, **When** I run the training script with Hydra config, **Then** a BERT model is trained on criterion-sentence pairs in NSP format and logged to MLflow
2. **Given** training has completed successfully, **When** I check MLflow UI, **Then** I can see run parameters, metrics (train/val loss, accuracy), and the saved model artifact
3. **Given** a training run, **When** I specify a random seed via Hydra, **Then** the results are reproducible across runs with the same seed

---

### User Story 2 - Evaluate Model Performance (Priority: P1)

As a researcher, I want to evaluate trained models on test data so that I can measure classification performance using standard metrics (precision, recall, F1).

**Why this priority**: Evaluation is essential to understand if the model works. Cannot proceed without validation.

**Independent Test**: Can be tested by loading a trained model from MLflow and running evaluation on a held-out test set, verifying metrics are computed and logged.

**Acceptance Scenarios**:

1. **Given** a trained model in MLflow, **When** I run the evaluation script, **Then** test metrics (precision, recall, F1, accuracy) are computed and logged
2. **Given** multiple models in MLflow, **When** I compare their evaluation metrics, **Then** I can identify the best performing model
3. **Given** evaluation on test set, **When** predictions are made, **Then** per-class metrics are available for analysis

---

### User Story 3 - Configure Experiments via Hydra (Priority: P1)

As a researcher, I want to configure all training parameters (model name, learning rate, batch size, data paths) via YAML configs so that I can easily run experiments with different settings without modifying code.

**Why this priority**: Critical for reproducibility and experiment management. Required by constitution P3.

**Independent Test**: Can be tested by modifying a YAML config file or passing CLI overrides and verifying the training run uses those parameters (visible in MLflow logs).

**Acceptance Scenarios**:

1. **Given** default Hydra configs exist, **When** I run training without overrides, **Then** default parameters are used and logged
2. **Given** I want to test a different BERT model, **When** I override `model.name=bert-large-uncased`, **Then** training uses that model
3. **Given** multiple experiment configurations, **When** I create separate YAML files, **Then** I can easily switch between them using Hydra

---

### User Story 4 - Hyperparameter Optimization with Optuna (Priority: P2)

As a researcher, I want to automatically search for optimal hyperparameters (learning rate, batch size, dropout) using Optuna so that I can improve model performance without manual tuning.

**Why this priority**: Important for model optimization but not required for basic functionality. Can be added after baseline pipeline works.

**Independent Test**: Can be tested by running an Optuna study with a small search space and verifying multiple trials are logged to both Optuna DB and MLflow.

**Acceptance Scenarios**:

1. **Given** an Optuna search space defined in Hydra, **When** I run the HPO script, **Then** multiple trials are executed and logged
2. **Given** HPO has completed, **When** I check the Optuna study, **Then** I can see the best hyperparameters found
3. **Given** Optuna trial results, **When** I inspect MLflow, **Then** each trial is logged as a separate run with its parameters and metrics

---

### User Story 5 - Load and Preprocess ReDSM5 Dataset (Priority: P1)

As a developer, I want to load ReDSM5 posts and annotations, pair them with DSM-5 criteria, and format them as NSP-style inputs so that the BERT model receives properly formatted training data.

**Why this priority**: Data loading is a prerequisite for training. Cannot train without properly formatted data.

**Independent Test**: Can be tested by running the dataset loader and verifying outputs match expected NSP format: `[CLS] <criterion> [SEP] <sentence> [SEP]`.

**Acceptance Scenarios**:

1. **Given** ReDSM5 CSV files in `data/redsm5/`, **When** I load the dataset, **Then** posts and annotations are parsed correctly
2. **Given** DSM-5 criteria JSON in `data/data/DSM5/`, **When** I create criterion-sentence pairs, **Then** each pair matches a criterion to a sentence with its label
3. **Given** paired data, **When** tokenization is applied, **Then** inputs follow NSP format with proper [CLS], [SEP] tokens and attention masks

---

### Edge Cases

- What happens when a post has no annotated evidence sentences? (Should be included as negative examples)
- How does system handle very long sentences that exceed BERT's 512 token limit? (Truncation strategy needed)
- What if MLflow tracking URI is unavailable or database is locked? (Graceful failure with informative error)
- How to handle class imbalance in binary classification? (Using weighted focal loss with configurable parameters)
- What if GPU runs out of memory during training? (Enable gradient checkpointing, reduce batch size, or use gradient accumulation)
- How to handle missing DSM-5 criteria files? (Fail fast with clear error message)
- What if focal loss hyperparameters (alpha, gamma) need tuning? (Make configurable via Hydra, can be part of Optuna search space)
- What if bf16 is not supported on GPU? (Fall back to fp16 with GradScaler automatically)
- What if torch.compile fails or causes issues? (Make it optional via Hydra config flag, can disable for debugging)
- What if FlashAttention is not installed? (Fall back to SDPA which is built-in to PyTorch)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load ReDSM5 posts from `data/redsm5/redsm5_posts.csv` and annotations from `data/redsm5/redsm5_annotations.csv`
- **FR-002**: System MUST load DSM-5 criteria from `data/data/DSM5/MDD_Criteira.json`
- **FR-003**: System MUST create criterion-sentence pairs using only annotated combinations from `redsm5_annotations.csv` (pair each sentence with its corresponding DSM5_symptom criterion). Format: `[CLS] <criterion_text> [SEP] <sentence_text> [SEP]`. Label is determined by `status` field (1=evidence present, 0=evidence absent).
- **FR-004**: System MUST use BERT-family models from Hugging Face Transformers with a binary classification head (single unified model, not separate models per symptom)
- **FR-005**: System MUST support post-level train/validation/test splits (70/15/15) with stratification to maintain label balance. All sentences from the same post MUST stay in the same split to prevent data leakage.
- **FR-006**: System MUST implement training loop with configurable epochs, learning rate, batch size, and optimizer. Loss function MUST use weighted focal loss to handle class imbalance (class weights inversely proportional to frequencies, focal loss parameters configurable via Hydra).
- **FR-007**: System MUST compute and log training/validation metrics (loss, accuracy, precision, recall, F1) per epoch
- **FR-008**: System MUST save trained models as MLflow artifacts with model registry support
- **FR-009**: System MUST implement evaluation script that loads models from MLflow and computes test metrics
- **FR-010**: System MUST use Hydra for all configuration management (data paths, model config, training hyperparameters)
- **FR-011**: System MUST track all experiments to MLflow using `sqlite:///mlflow.db` and artifact store `./mlruns`
- **FR-012**: System MUST set and log random seeds for Python, NumPy, PyTorch, and Transformers for reproducibility
- **FR-013**: System MUST log Hydra config, Git commit SHA, and environment info (pip freeze) as MLflow artifacts
- **FR-014**: System MUST support optional Optuna HPO with SQLite storage `sqlite:///optuna.db`
- **FR-015**: System MUST implement PyTorch Dataset and DataLoader for efficient batch processing with proper collation
- **FR-016**: System MUST handle tokenization with proper padding, truncation, and attention masks
- **FR-017**: System MUST support both CPU and GPU training with automatic device detection
- **FR-018**: System MUST pass Black formatting (line length 100), Ruff linting, and MyPy type checking
- **FR-019**: System MUST provide utilities for setting deterministic backends (`cudnn.deterministic=True`)
- **FR-020**: System MUST implement proper logging using the existing `get_logger` utility
- **FR-021**: System MUST implement full PyTorch optimization suite: mixed precision (bf16 if supported, else fp16 with GradScaler), TF32 matmul acceleration, SDPA or FlashAttention for efficient attention, gradient checkpointing for memory savings, fused AdamW optimizer
- **FR-022**: System MUST support torch.compile for graph optimization and kernel fusion (configurable via Hydra)
- **FR-023**: System MUST implement efficient DataLoader with pinned memory, multiple workers, and persistent workers
- **FR-024**: System MUST support optional gradient accumulation for effective larger batch sizes when GPU memory is limited
- **FR-025**: System MUST configure BERT model to use efficient attention implementation (attn_implementation="sdpa" or "flash_attention_2") and disable key-value cache during training

### Key Entities

- **CriterionSentencePair**: Represents a training example with criterion text, sentence text, label (0/1), and metadata (post_id, sentence_id, symptom)
- **ReDSM5Dataset**: PyTorch Dataset that loads and tokenizes criterion-sentence pairs
- **BERTBinaryClassifier**: Model wrapper containing BERT base model and binary classification head
- **TrainingRun**: Represents an experiment tracked in MLflow with parameters, metrics, and artifacts
- **HydraConfig**: Configuration object containing all experiment settings (data, model, training, MLflow, Optuna)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Training pipeline can complete a full epoch on ReDSM5 dataset within reasonable time (e.g., <15 minutes on GPU with full optimizations for baseline BERT, <30 minutes without optimizations)
- **SC-002**: Trained model achieves reasonable binary classification performance (F1 > 0.6 on validation set as baseline, target F1 > 0.75 with optimized training)
- **SC-003**: All training runs are fully logged to MLflow with parameters, metrics, and artifacts
- **SC-004**: Training is reproducible: running with same seed and config produces identical metrics (within floating-point precision)
- **SC-005**: Configuration can be modified via Hydra CLI overrides without code changes
- **SC-006**: Code passes all quality gates: Black, Ruff, MyPy checks
- **SC-007**: Dataset loading handles all 1,484 posts and creates correct number of criterion-sentence pairs based on annotations
- **SC-008**: MLflow model registry contains at least one registered model that can be loaded for inference
- **SC-009**: Evaluation script produces per-symptom metrics for all 9 DSM-5 symptoms plus SPECIAL_CASE
- **SC-010**: System can train on CPU (for development) and GPU (for production) without code changes
- **SC-011**: PyTorch optimizations provide measurable speedup: training with full optimizations should be at least 2x faster than baseline (measured on same hardware)
- **SC-012**: Mixed precision training reduces GPU memory usage by at least 30% compared to FP32 training, enabling larger batch sizes

## Technical Context

### Existing Codebase State

Currently implemented:
- Basic model wrapper in `src/Project/SubProject/models/model.py` (needs refactoring)
- MLflow utilities in `src/Project/SubProject/utils/mlflow_utils.py`
- Logging utilities in `src/Project/SubProject/utils/log.py`
- Seed utilities in `src/Project/SubProject/utils/seed.py`

Not implemented (empty files):
- `src/Project/SubProject/data/dataset.py` - needs full implementation
- `src/Project/SubProject/engine/train_engine.py` - needs full implementation
- `src/Project/SubProject/engine/eval_engine.py` - needs full implementation
- `scripts/` - needs training and evaluation scripts
- `configs/` - needs Hydra YAML configurations

### Data Available

- ReDSM5 dataset: `data/redsm5/redsm5_posts.csv` (1,484 posts) and `data/redsm5/redsm5_annotations.csv` (1,547 annotations)
- DSM-5 criteria: `data/data/DSM5/MDD_Criteira.json` (9 depression criteria A.1-A.9)
- Symptom distribution varies: DEPRESSED_MOOD (328), WORTHLESSNESS (311), SUICIDAL_THOUGHTS (165), etc.
- Dataset includes hard negatives (392 posts with no symptoms)

### Constitution Compliance

This feature MUST comply with all principles in `.specify/memory/constitution.md` v1.1.0:
- P1: BERT-based binary classification with Hugging Face
- P2: NSP-style criterion-sentence input format
- P3: Hydra configuration management
- P4: MLflow experiment tracking (local SQLite)
- P5: Optional Optuna HPO
- P6: Full reproducibility (seeds, environment capture, config logging)
