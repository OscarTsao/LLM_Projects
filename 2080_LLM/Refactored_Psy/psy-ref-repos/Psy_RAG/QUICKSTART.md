# Quick Start Guide

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify GPU availability** (for RTX 3090):
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Quick Demo

Run the demonstration script to see the system in action:

```bash
python demo.py
```

This will:
- Build the FAISS index from DSM-5 criteria
- Analyze sample posts for mental health criteria matches
- Show performance metrics and results

## Basic Usage

### 1. Build Index (First Time)

```bash
python main.py --mode build_index --save_index
```

### 2. Analyze Single Post

```bash
python main.py --mode single_post --post_text "I feel very depressed and hopeless about my future"
```

### 3. Evaluate Multiple Posts

```bash
python main.py --mode evaluate --num_posts 100 --load_index data/indices/faiss_index
```

## Configuration

Edit `src/config/settings.py` to customize:
- Model parameters
- Similarity thresholds
- Batch sizes
- GPU optimizations

## Performance Tips

For RTX 3090:
- The system automatically uses FP16 precision
- Batch size is optimized for 16GB VRAM
- FAISS uses GPU acceleration when available

## Output

Results are saved as JSON files in the `results/` directory with:
- Matched criteria with confidence scores
- Supporting text spans
- Processing statistics
- Diagnosis counts

## Troubleshooting

- **CUDA OOM**: Reduce batch size in settings
- **Model loading errors**: Check internet connection
- **FAISS errors**: Rebuild index

## Testing

Run the test suite:

```bash
python run_tests.py
```

Or run specific tests:

```bash
pytest tests/test_rag_pipeline.py -v
```
