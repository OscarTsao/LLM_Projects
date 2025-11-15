# Criteria Binder: SpanBERT-based Evidence Binding for Criteria Matching

A fully working Python project that trains and serves a SpanBERT-based evidence binding model for "criteria matching" tasks. Given a criterion text and a document, the model predicts both a label (match/no-match) and supporting evidence spans in the document.

## 🚀 Quick Start

### Installation

#### Option 1: Using Conda (Recommended)

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate criteria_binder

# Install in development mode
pip install -e .
```

#### Option 2: Using pip

```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```bash
# Train model with Hydra config
python -m src.cli train --config-path src/config --config-name train

# Evaluate on test set using Hydra overrides
python -m src.cli eval --config-path src/config --config-name eval eval.checkpoint=outputs/run1/best eval.split=test

# Run predictions with Hydra config
python -m src.cli predict --config-path src/config --config-name predict predict.checkpoint=outputs/run1/best predict.input_path=data/examples/test.jsonl predict.output_path=outputs/predictions.jsonl
```

### Using Scripts

```bash
# Train with default settings
./scripts/train.sh

# Evaluate trained model
CHECKPOINT_PATH=outputs/run1/best ./scripts/eval.sh

# Run predictions
INPUT_FILE=data/examples/test.jsonl OUTPUT_FILE=outputs/predictions.jsonl ./scripts/predict.sh
```

## 📋 Architecture

### Model Overview

The model uses a **cross-encoder architecture** based on SpanBERT with:

- **Span Head**: Predicts start/end position scores for evidence spans in the document
- **Classification Head** (optional): Predicts match/no-match labels for multi-task learning
- **Input Format**: `[CLS] <criterion_text> [SEP] <document_text> [SEP]`
- **Sliding Windows**: Handles long documents by keeping the entire criterion in each window

### Key Features

- ✅ **Multi-span decoding** with top-K selection and NMS
- ✅ **Sliding window processing** for long documents
- ✅ **Character-level span alignment** with robust offset mapping
- ✅ **Multi-task learning** combining span extraction and classification
- ✅ **Comprehensive evaluation** with IoU-based span metrics
- ✅ **Production-ready** with Docker support and comprehensive testing

## 📊 Data Format

### Input Schema (JSONL)

```json
{
  "id": "case_0001_c1",
  "criterion_text": "Reports persistent low mood (≥ 2 weeks)",
  "document_text": "Patient has felt down almost every day for the last three weeks...",
  "label": 1,
  "evidence_char_spans": [[86, 130]]
}
```

**Fields:**

- `id`: Unique identifier
- `criterion_text`: Criterion to match against
- `document_text`: Document text to search in
- `label`: Optional classification label (0/1 or multi-class)
- `evidence_char_spans`: List of `[start_char, end_char]` evidence spans in document

### Output Schema

```json
{
  "id": "case_0001_c1",
  "pred_label": 1,
  "pred_label_probs": [0.1, 0.9],
  "pred_spans_char": [
    [86, 130],
    [200, 235]
  ],
  "pred_spans_scores": [12.5, 9.3]
}
```

## ⚙️ Configuration

Default configurations live under `src/config/` and are managed with [Hydra](https://github.com/facebookresearch/hydra).

```bash
# Inspect the composed training config
python -m src.cli train --config-path src/config --config-name train --cfg job

# Override parameters on the command line
python -m src.cli train --config-path src/config --config-name train train.batch_size=16 model.max_length=256 logging.output_dir=outputs/custom_run
```

## 🧠 Model Components

### SpanBertEvidenceBinder

The core model (`src/models/binder.py`) implements:

- **Backbone**: SpanBERT encoder from HuggingFace
- **Span Heads**: Linear layers for start/end position prediction
- **Classification Head**: Optional linear layer for label prediction
- **Loss Function**: Combined span loss (CrossEntropy) + classification loss
- **Token Masking**: Prevents prediction on criterion tokens

### Data Processing

#### Alignment (`src/data/alignment.py`)

- Character ↔ token position mapping
- Sliding window creation with stride
- Offset mapping for span conversion
- Unicode-safe text handling

#### Collator (`src/training/collator.py`)

- Dynamic padding and batching
- Window-based data loading
- Proper token type ID assignment
- Training vs inference mode support

#### Decoding (`src/utils/decode.py`)

- Top-K span extraction from position scores
- Non-Maximum Suppression (NMS) for overlapping spans
- Cross-window span aggregation
- Character-level output conversion

## 📈 Training & Evaluation

### Training Loop

The trainer (`src/training/train.py`) includes:

- **Optimization**: AdamW with linear warmup and cosine decay
- **Mixed Precision**: FP16 support via `torch.amp.autocast`
- **Gradient Accumulation**: For effective large batch training
- **Early Stopping**: Based on validation metrics
- **Checkpointing**: Save top-k models by combined score

### Metrics

Comprehensive evaluation (`src/utils/metrics.py`):

- **Span Metrics**: Character-level precision/recall/F1 with IoU matching
- **Classification Metrics**: Accuracy, macro/micro F1
- **Combined Score**: Weighted average of span F1 and label F1

### Example Training Run

```bash
# Start training
python -m src.cli train --config src/config.yaml \
    --overrides logging.output_dir=outputs/my_run

# Monitor progress
tail -f outputs/my_run/train.log

# Evaluate best checkpoint
python -m src.cli eval --checkpoint outputs/my_run/best --split test
```

## 🐳 Docker Usage

### Build and Run

```bash
# Build image
docker-compose build

# Run training
docker-compose run criteria-binder-train

# Development mode
docker-compose run criteria-binder-dev

# Interactive shell
docker-compose run criteria-binder bash
```

### GPU Support

The Docker setup includes NVIDIA GPU support. Ensure you have:

- NVIDIA drivers installed
- Docker with GPU support
- `nvidia-container-toolkit`

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_alignment.py -v
pytest tests/test_decode.py -v
pytest tests/test_collator.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

The test suite covers:

- ✅ Character-token alignment with Unicode support
- ✅ Span decoding algorithms and NMS
- ✅ Data collation and sliding windows
- ✅ Edge cases and error handling
- ✅ Round-trip span conversion

## 📚 Examples

### Example 1: Medical Criteria Matching

```json
{
  "criterion_text": "Reports persistent low mood (≥ 2 weeks)",
  "document_text": "Patient has felt down almost every day for the last three weeks, experiencing sadness and hopelessness.",
  "evidence_char_spans": [[26, 80]]
}
```

### Example 2: Multi-span Evidence

```json
{
  "criterion_text": "Reports fatigue or loss of energy",
  "document_text": "Patient feels exhausted all the time, even after sleeping 10 hours. Simple tasks feel overwhelming.",
  "evidence_char_spans": [
    [8, 35],
    [75, 97]
  ]
}
```

## 🔧 Advanced Usage

### Custom Backbones

Switch to different models:

```bash
# Use BERT instead of SpanBERT
python -m src.cli train --config src/config.yaml \
    --overrides model.name=bert-base-uncased

# Use domain-specific model
python -m src.cli train --config src/config.yaml \
    --overrides model.name=emilyalsentzer/Bio_ClinicalBERT
```

### Multi-class Classification

For more than 2 labels, update the config:

```yaml
model:
  num_labels: 5 # Adjust based on your classes
  use_label_head: true
```

### Hyperparameter Tuning

Key parameters to tune:

- `train.lr`: Learning rate (2e-5 to 5e-5)
- `model.lambda_span`: Balance between span and classification loss
- `model.max_length`: Sequence length vs memory trade-off
- `decode.nms_iou_thresh`: Strictness of span overlap filtering

## 🚨 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce batch size and increase gradient accumulation
--overrides train.batch_size=8 train.grad_accum=2
```

#### 2. Long Training Times

```bash
# Use mixed precision and larger batch size
--overrides train.fp16=true train.batch_size=32
```

#### 3. Poor Span Performance

```bash
# Increase span loss weight
--overrides model.lambda_span=0.8
```

#### 4. Model Not Converging

```bash
# Try different learning rate
--overrides train.lr=1e-5 train.warmup_ratio=0.1
```

### Debugging Tips

1. **Check data loading**: Verify offset mappings with small examples
2. **Monitor losses**: Use `--log-level DEBUG` for detailed logs
3. **Validate spans**: Test alignment functions independently
4. **Profile memory**: Use smaller batch sizes for debugging

## 📖 Project Structure

```
criteria_binder/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
├── pyproject.toml              # Build configuration
├── setup.cfg                   # Package metadata
├── docker/
│   ├── Dockerfile              # Container definition
│   └── docker-compose.yml      # Multi-service setup
├── data/examples/              # Sample datasets
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── src/
│   ├── __init__.py
│   ├── config.yaml             # Default configuration
│   ├── cli.py                  # Command-line interface
│   ├── models/
│   │   └── binder.py           # SpanBERT model implementation
│   ├── data/
│   │   ├── dataset.py          # Dataset loading
│   │   └── alignment.py        # Character-token alignment
│   ├── training/
│   │   ├── train.py            # Training loop
│   │   ├── eval.py             # Evaluation utilities
│   │   ├── collator.py         # Data collation
│   │   └── callbacks.py        # Training callbacks
│   └── utils/
│       ├── decode.py           # Span decoding with NMS
│       ├── metrics.py          # Evaluation metrics
│       ├── logging.py          # Logging utilities
│       ├── seed.py             # Reproducibility
│       └── io.py               # File I/O helpers
├── scripts/
│   ├── train.sh                # Training script
│   ├── eval.sh                 # Evaluation script
│   └── predict.sh              # Prediction script
└── tests/
    ├── test_alignment.py       # Alignment tests
    ├── test_decode.py          # Decoding tests
    └── test_collator.py        # Collator tests
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Submit a pull request

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Built on [SpanBERT](https://github.com/facebookresearch/SpanBERT) architecture
- Uses [HuggingFace Transformers](https://huggingface.co/transformers/) library
- Inspired by question-answering and information extraction research

---

For questions or issues, please open a GitHub issue or check the troubleshooting section above.
