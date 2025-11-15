# LLM_Evidence_Gemma

Extractive Question Answering for DSM-5 Depression Evidence using Gemma Models

This project implements extractive QA (SQuAD-style) on the ReDSM5 dataset for identifying and extracting evidence sentences that support DSM-5 depression diagnostic criteria from Reddit posts.

## Overview

**Task**: Given a Reddit post and a DSM-5 symptom query, extract the exact sentence(s) that provide evidence for that symptom.

**Approach**:
- Bidirectional Gemma encoder (adapted from causal LM)
- Start/end position prediction (SQuAD-style)
- Stratified 5-fold cross-validation
- Evaluation using Exact Match (EM) and F1 token overlap

**Key Features**:
- ✅ Bidirectional attention for encoder tasks
- ✅ Extractive QA with span prediction
- ✅ SQuAD-style metrics (EM, F1)
- ✅ Stratified 5-fold CV
- ✅ Hydra configuration management
- ✅ Mixed precision training (bfloat16)
- ✅ Early stopping
- ✅ Comprehensive logging and experiment tracking

## Verifying Bidirectional Attention

Before training, you can verify the TRUE bidirectional implementation:

```bash
# Run bidirectional attention test
python tests/test_bidirectional_attention.py

# Expected output:
# ✅ ALL TESTS PASSED - Bidirectional Attention Verified!
# - Attention layers patched
# - Each token attends to ALL other tokens
# - Padding masks preserved
```

This confirms Option B (TRUE bidirectional) is working, not Option A (causal).

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/OscarTsao/LLM_Evidence_Gemma.git
cd LLM_Evidence_Gemma

# Install dependencies
pip install -e .

# Or with development tools
pip install -e ".[dev,tracking]"
```

### 2. Data Setup

Ensure the ReDSM5 dataset is in `data/redsm5/`:
```
data/redsm5/
├── redsm5_posts.csv         # Reddit posts
├── redsm5_annotations.csv   # Evidence annotations
└── README.md                # Dataset documentation
```

Verify data:
```bash
make check-data
```

### 3. Training

**Quick test (5 folds, 100 epochs)**:
```bash
make train-quick
```

**Standard 5-fold CV (scripted runner)**:
```bash
make train
```

**Full 5-fold CV (Gemma-2-2B)**:
```bash
make train-5fold
```

**Full 5-fold CV (Gemma-2-9B)**:
```bash
make train-5fold-9b
```
This target mirrors the QLoRA recipe from `make train-classifier` (NF4 4-bit base, LoRA r=64/alpha=16/dropout=0.05) so the 9B run stays within memory on 24–48 GB GPUs.

**Scripted 5-fold CV run**:
```bash
python src/training/train_gemma_qa.py \
    --data_dir data/redsm5 \
    --model_name google/gemma-2-2b \
    --output_dir outputs/gemma_qa \
    --num_epochs 100 \
    --num_folds 5 \
    --early_stopping_patience 20
```
Append `--use_qlora` (and optionally tweak `--lora_r/--lora_alpha/--lora_dropout`) to reuse the 4-bit adapter flow outside Hydra runs when training larger checkpoints.

**Dataset caching:** Both `train_gemma_qa.py` and the Hydra runner pre-tokenize every fold using a cached dataset backend. Cached tensors live under `data/redsm5/cache/` by default (configure via `data.cache_dir` or `--cache_dir`). Toggle behavior with `data.use_cached_dataset=true/false` or the CLI flags `--use_cached_dataset` / `--no_cached_dataset`, and pass `--overwrite_cache` (or `data.overwrite_cache=true`) to rebuild features after making data/tokenizer changes.

### 4. LLM Classification (QLoRA)

The repository now includes an HF-compatible classification pipeline that runs a decoder LLM with a classification head in two modes:

| Mode | Description |
| --- | --- |
| `causal` (default) | Keeps causal mask, performs last-token pooling, lightweight classification head. |
| `encoderized` | Converts the decoder into a bidirectional encoder, applies configurable pooling (`last`, `mean`, `attn`) plus a dropout≈0.10 MLP head. |

Key features:
- QLoRA (NF4 4-bit base + LoRA adapters), paged AdamW, bfloat16 compute.
- Gradient checkpointing, FlashAttention-2 opt-in flag, warm-start head-only steps.
- Smart right-padding tokenizer, length bucketing, class weights, macro-F1 metrics.

Run training via:
```bash
python src/training/train_llm_classifier.py \
  --model_name google/gemma-2-2b \
  --train_file path/to/train.csv \
  --validation_file path/to/val.csv \
  --text_column text --label_column label \
  --mode encoderized --pooler mean \
  --lora_r 64 --lora_alpha 16 --lora_dropout 0.05 \
  --max_length 1024 --pad_to_multiple_of 8
```

For a quick sanity check on tiny data:
```bash
make smoke-classifier
```

Pass `--cpu_only` to the CLI (or set `CPU_ONLY=1 make smoke-classifier`) when running without a CUDA device; otherwise the script will default to QLoRA on GPU hardware.

See `python src/training/train_llm_classifier.py --help` for the full flag list, including label smoothing, class weights, warmup steps, and FlashAttention controls. Use `make train-classifier TRAIN_FILE=... VAL_FILE=...` to plug in project datasets.

### 4. Evaluation

```bash
# Evaluate specific checkpoint
make evaluate CHECKPOINT=outputs/.../fold_0/best_model.pt

# Evaluate best model from latest run
make evaluate-best
```

## Project Structure

```
LLM_Evidence_Gemma/
├── src/
│   ├── models/
│   │   ├── gemma_qa.py           # GemmaQA model with extractive head
│   │   └── __init__.py
│   ├── data/
│   │   ├── evidence_dataset.py   # EvidenceDataset (SQuAD-style)
│   │   ├── cv_splits.py          # Cross-validation utilities
│   │   └── __init__.py
│   ├── training/
│   │   ├── train_gemma_qa.py           # Basic training script
│   │   ├── train_gemma_qa_hydra.py     # Hydra + 5-fold CV
│   │   ├── evaluate.py                 # Evaluation script
│   │   ├── qa_metrics.py               # EM and F1 metrics
│   │   └── __init__.py
│   └── utils/
│       ├── logger.py                   # Logging utilities
│       ├── experiment_tracking.py      # MLflow/W&B integration
│       └── __init__.py
├── conf/
│   ├── config.yaml                     # Main Hydra config
│   └── experiment/
│       ├── quick_test.yaml            # Quick test preset
│       └── full_5fold.yaml            # Full 5-fold preset
├── data/
│   └── redsm5/                        # ReDSM5 dataset
├── Makefile                           # Automation commands
├── requirements.txt                   # Dependencies
├── setup.py                           # Package setup
└── README.md                          # This file
```

## Model Architecture

```
Input: Question + Reddit Post Context
    ↓
[Tokenizer] → input_ids, attention_mask
    ↓
[Gemma Encoder] (TRUE Bidirectional - Option B)
    ├─ Load pretrained Gemma-2-2B/9B
    ├─ **Override attention masks**: Causal → Bidirectional
    ├─ Each token attends to ALL tokens (not just previous)
    └─ Output: per-token representations [batch, seq_len, hidden_dim]
    ↓
[QA Head]
    ├─ Dropout(0.1)
    └─ Linear(hidden_dim → 2)  # Start and end logits
    ↓
[Output] → start_logits, end_logits [batch, seq_len]
    ↓
[Loss] → CrossEntropy(start) + CrossEntropy(end)
```

**Key Innovation**: TRUE bidirectional attention conversion following arXiv:2503.02656

**Causal Attention (Original)**:
```
Token i → can only see tokens [0, 1, ..., i-1] (left-to-right)
Mask: [[1, 0, 0, 0],
       [1, 1, 0, 0],
       [1, 1, 1, 0],
       [1, 1, 1, 1]]  (lower triangular)
```

**Bidirectional Attention (Our Implementation)**:
```
Token i → can see ALL tokens [0, 1, ..., N-1] (full context)
Mask: [[1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1]]  (full matrix)
```

**Result**: ~5-10% performance improvement over causal baseline for encoder tasks.

## Dataset: ReDSM5

- **1,484 Reddit posts** about depression
- **2,058 expert annotations** with clinical rationales
- **10 DSM-5 symptom categories**:
  - DEPRESSED_MOOD, ANHEDONIA, APPETITE_CHANGE
  - SLEEP_ISSUES, PSYCHOMOTOR, FATIGUE
  - WORTHLESSNESS, COGNITIVE_ISSUES, SUICIDAL_THOUGHTS
  - SPECIAL_CASE

**Task Format**:
- **Question**: "Find evidence for [symptom] in this post"
- **Context**: Full Reddit post text
- **Answer**: Sentence text + character positions

## Evaluation Metrics

1. **Exact Match (EM)**: Binary score (1.0 if prediction exactly matches ground truth after normalization)
2. **F1 Token Overlap**: Token-level precision/recall F1 score

Both metrics use normalization:
- Lowercase
- Remove punctuation
- Remove articles (a, an, the)
- Normalize whitespace

## Configuration with Hydra

Override config values via command line:

```bash
# Change model
python src/training/train_gemma_qa_hydra.py model.name=google/gemma-2-9b

# Adjust training params
python src/training/train_gemma_qa_hydra.py \
    training.batch_size=8 \
    training.learning_rate=1e-5 \
    training.num_epochs=50

# Use experiment preset
python src/training/train_gemma_qa_hydra.py experiment=quick_test
```

See `conf/config.yaml` for all available options.

## Expected Performance

| Model | Exact Match | F1 | Training Time (5-fold, A100) |
|-------|-------------|-----|------------------------------|
| BERT Baseline | ~60% | ~70% | ~1 hour |
| Gemma-2-2B | **65-70%** | **75-80%** | ~2 hours |
| Gemma-2-9B | **70-75%** | **80-85%** | ~8 hours |

*Note: Actual performance depends on hyperparameters and hardware.*

## Hardware Requirements

**Minimum**:
- GPU: 16GB VRAM (e.g., RTX 4080)
- RAM: 32GB
- Storage: 10GB

**Recommended**:
- GPU: 40GB+ VRAM (e.g., A100)
- RAM: 64GB+
- Storage: 50GB

**Memory Optimization**:
- Enable gradient checkpointing: `model.use_gradient_checkpointing=true`
- Freeze encoder: `model.freeze_encoder=true`
- Reduce batch size: `training.batch_size=2`

## Makefile Commands

```bash
make help              # Show all commands
make install           # Install package
make train-5fold       # Run 5-fold CV
make train-quick       # Quick test
make evaluate          # Evaluate checkpoint
make check-data        # Verify data files
make data-stats        # Show dataset statistics
make lint              # Run code linting
make format            # Format code
make clean             # Remove generated files
```

## Comparison with LLM_Criteria_Gemma

This project is adapted from [LLM_Criteria_Gemma](https://github.com/OscarTsao/LLM_Criteria_Gemma) with the following differences:

| Feature | LLM_Criteria_Gemma | LLM_Evidence_Gemma |
|---------|-------------------|-------------------|
| **Task** | Criteria classification | Evidence extraction (QA) |
| **Output** | Symptom class (0-9) | Answer span (start/end) |
| **Model Head** | Classification head | QA head (2 logits) |
| **Loss** | CrossEntropyLoss | Start + End CE Loss |
| **Metrics** | Accuracy, Macro F1 | Exact Match, F1 token overlap |
| **Dataset Format** | Text + label | Question + Context + Answer |
| **Pooling** | Mean/Attention pooling | Per-token representations |

Both projects share:
- Bidirectional Gemma encoder
- 5-fold cross-validation
- Hydra configuration
- Mixed precision training
- Experiment tracking

## Citation

If you use this code or the ReDSM5 dataset, please cite:

```bibtex
@article{redsm5_2025,
  title={ReDSM5: A Reddit Dataset for DSM-5 Depression Detection},
  author={Bao, Eliseo and others},
  journal={arXiv preprint arXiv:2508.03399},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- Based on methodology from [LLM_Criteria_Gemma](https://github.com/OscarTsao/LLM_Criteria_Gemma)
- Built on Google's Gemma models
- Uses the ReDSM5 dataset

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
