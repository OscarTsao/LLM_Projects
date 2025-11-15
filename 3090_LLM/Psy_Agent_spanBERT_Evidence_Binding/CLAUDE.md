# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpanBERT fine-tuning pipeline for identifying evidential spans in Reddit mental health posts. Uses Hugging Face Transformers, PyTorch, Hydra configuration, and Optuna for hyperparameter search. The system is structured as a question-answering task where the model learns to extract evidence spans from social media text.

## Commands

### Training
```bash
# Standard training run
python -m src.train

# Override config parameters
python -m src.train training.num_train_epochs=4 training.train_batch_size=12

# Hyperparameter search with Optuna
python -m src.optuna_search optuna.n_trials=20
```

### Evaluation
```bash
# Evaluate a saved checkpoint
python scripts/evaluate.py \
  --config artifacts/<timestamp>/config.yaml \
  --checkpoint artifacts/<timestamp>/best_model.pt \
  --split test \
  --metrics_output reports/test_metrics.json \
  --predictions_output reports/test_predictions.json
```

### Validation
```bash
# Check Python syntax
python -m compileall src scripts
```

## Architecture

### Data Pipeline (`src/psya_agent/data.py`)
- **QAExample**: Core data structure pairing context text with annotated evidence spans (answer_text, answer_start)
- **Post-level splitting**: Ensures no data leakage by splitting at post_id level before creating train/val/test sets
- **Span alignment**: Case-insensitive fallback to locate sentence annotations within full post contexts

### Feature Engineering (`src/psya_agent/features.py`)
- **TokenizedFeature**: Wraps tokenized inputs with offset_mapping for character-to-token alignment
- **Striding**: Long documents handled via doc_stride with return_overflowing_tokens to generate multiple features per example
- **Training labels**: Start/end positions mapped to token indices; defaults to CLS token when gold span falls outside current stride window
- **Evaluation features**: Preserves offset_mapping and example_index for post-processing logits back to character spans

### Model (`src/psya_agent/modeling.py`)
- **SpanBertForQuestionAnswering**: SpanBERT encoder + dropout + single linear layer producing (start_logits, end_logits)
- **local_files_only**: Avoids HuggingFace Hub network calls when True (uses cached models only)

### Training Loop (`src/psya_agent/train_utils.py`)
- **run_training**: End-to-end orchestration: data prep → tokenization → training → evaluation → artifact saving
- **Gradient accumulation**: Effective batch size = train_batch_size × gradient_accumulation_steps
- **Mixed precision**: Automatic via torch.amp.GradScaler when CUDA available and mixed_precision != "off"
- **Early stopping**: Tracks best validation metric (default: f1) with configurable patience
- **Post-processing**: _postprocess_predictions merges logits from multiple stride windows per example, selects top n_best_size start/end pairs, filters by max_answer_length

### Metrics (`src/psya_agent/metrics.py`)
- **F1**: Token-level overlap computed after text normalization (lowercase, punctuation removal)
- **Exact Match**: Binary score after normalization

### Hyperparameter Search (`src/optuna_search.py`)
- Explores: learning_rate, train_batch_size, num_train_epochs, doc_stride, max_length, warmup_ratio
- Best trial retrained and saved to `artifacts/optuna_best/`

## Configuration (configs/config.yaml)

Key relationships:
- **data.seed**: Controls post-level split determinism
- **training.seed**: Controls model initialization and shuffling
- **model.local_files_only**: When True, sets HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1
- **model.gradient_checkpointing**: Enable to save memory at cost of ~20% speed
- **training.compile_model**: Enable for PyTorch 2.0+ for 10-20% speedup
- **training.mixed_precision**: Set to "auto" for FP16 on CUDA (2-3x speedup)
- **training.num_workers**: More workers = faster data loading (but more CPU/memory)
- **optimization.metric**: Must match a key returned by aggregate_metrics (f1 or exact_match)
- **features.doc_stride**: Smaller values = more overlapping windows = more features per long document
- **features.max_length**: Tokenizer truncation length (includes question + context)

## Performance Optimizations

### DataLoader Optimizations
- **persistent_workers=True**: Keeps worker processes alive between epochs
- **prefetch_factor=2**: Pre-loads 2 batches per worker
- **pin_memory=True**: Faster GPU transfer via page-locked memory
- **drop_last=True**: More stable batch normalization

### Training Optimizations
- **torch.compile()**: JIT compilation for faster forward/backward passes (PyTorch 2.0+)
- **Gradient checkpointing**: Trade compute for memory by recomputing activations
- **AMP (Automatic Mixed Precision)**: FP16 compute with FP32 master weights
- **Gradient accumulation**: Simulate larger batches without memory overhead
- **non_blocking=True**: Asynchronous GPU transfers

### Best Practices
1. Start with `num_workers=2`, increase if CPU is not bottleneck
2. Enable `compile_model` for production runs on PyTorch 2.0+
3. Use gradient checkpointing only if running out of memory
4. Monitor GPU utilization - if <80%, increase batch size or workers
5. For debugging, set `logging.use_tqdm=true` for progress bars

## Artifacts

After training, `artifacts/<timestamp>/` contains:
- `best_model.pt`: Full state_dict saved to CPU
- `config.yaml`: Resolved OmegaConf snapshot
- `test_metrics.json`: Final evaluation results

## Development Notes

- The model uses an empty question for all examples (QA framework requirement, but questions are unused)
- CLS token positions used as "no answer" signal when gold span not in current stride
- Data splits are deterministic per post_id hash to ensure reproducibility
- Tokenizer must be "fast" (Rust-backed) for offset_mapping support
