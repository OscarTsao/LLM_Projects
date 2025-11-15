# RAG System for DSM-5 Criteria Matching

A comprehensive Retrieval-Augmented Generation (RAG) system that matches social media posts against DSM-5 criteria descriptions using BGE-M3 embeddings and SpanBERT for filtering and token extraction.

## Features

- **BGE-M3 Embedding Model**: High-quality multilingual embeddings for semantic similarity
- **FAISS Vector Database**: Efficient similarity search with GPU acceleration
- **SpanBERT Integration**: Advanced filtering and token extraction for criteria matching
- **RTX 3090 Optimizations**: Performance optimizations for NVIDIA RTX 3090 GPU
- **Comprehensive Testing**: Full test suite with coverage reporting
- **Modular Architecture**: Clean, maintainable code structure

## Project Structure

```
Psy_RAG/
├── src/
│   ├── models/
│   │   ├── embedding_model.py      # BGE-M3 embedding implementation
│   │   ├── faiss_index.py          # FAISS vector database
│   │   ├── spanbert_model.py       # SpanBERT for filtering
│   │   └── rag_pipeline.py         # Main RAG pipeline
│   ├── utils/
│   │   ├── data_loader.py          # Data loading utilities
│   │   └── performance_optimizer.py # RTX 3090 optimizations
│   └── config/
│       └── settings.py             # Configuration settings
├── tests/
│   ├── test_data_loader.py         # Data loader tests
│   ├── test_embedding_model.py     # Embedding model tests
│   ├── test_faiss_index.py         # FAISS index tests
│   ├── test_rag_pipeline.py        # RAG pipeline tests
│   └── conftest.py                 # Test fixtures
├── data/
│   ├── embeddings/                 # Generated embeddings
│   └── indices/                    # FAISS indices
├── results/                        # Output results
├── logs/                          # Log files
├── main.py                        # Main execution script
├── run_tests.py                   # Test runner
└── requirements.txt               # Dependencies
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Psy_RAG
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify GPU availability** (for RTX 3090):
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Usage

### 1. Build FAISS Index

First, build the FAISS index from DSM-5 criteria:

```bash
python main.py --mode build_index --save_index
```

This will:
- Load DSM-5 criteria from `Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json`
- Generate BGE-M3 embeddings for all criteria
- Build and save FAISS index to `data/indices/faiss_index/`

### 2. Evaluate Posts

Evaluate social media posts against DSM-5 criteria:

```bash
python main.py --mode evaluate --num_posts 100 --load_index data/indices/faiss_index
```

Parameters:
- `--num_posts`: Number of posts to evaluate (default: 100)
- `--similarity_threshold`: FAISS similarity threshold (default: 0.7)
- `--spanbert_threshold`: SpanBERT confidence threshold (default: 0.5)
- `--top_k`: Number of top criteria to retrieve (default: 10)

### 3. Single Post Analysis

Analyze a single post:

```bash
python main.py --mode single_post --post_text "I feel very depressed and hopeless about my future" --load_index data/indices/faiss_index
```

### 4. Run Tests

Execute the test suite:

```bash
python run_tests.py
```

Or run specific tests:

```bash
pytest tests/test_rag_pipeline.py -v
```

## Configuration

Edit `src/config/settings.py` to customize:

- **Model settings**: BGE-M3 and SpanBERT model names
- **FAISS settings**: Index type, metric, and performance parameters
- **RAG settings**: Similarity thresholds, batch sizes, and limits
- **GPU settings**: Device configuration and optimizations

## Performance Optimizations

The system includes several optimizations for RTX 3090:

- **Mixed Precision**: FP16 for faster inference
- **TensorFloat-32**: Enhanced performance for matrix operations
- **cuDNN Optimizations**: Benchmark mode and memory management
- **Batch Processing**: Efficient batch processing with memory management
- **GPU Memory Management**: Automatic cache clearing and memory monitoring

## Output Format

The system generates detailed results including:

```json
{
  "post_id": 1,
  "post_text": "I feel very depressed and hopeless...",
  "total_matches": 2,
  "processing_time": 1.23,
  "matched_criteria": [
    {
      "criteria_id": "Major_Depressive_Disorder_A.1",
      "diagnosis": "Major Depressive Disorder",
      "criterion_text": "Depressed mood most of the day...",
      "similarity_score": 0.85,
      "spanbert_score": 0.78,
      "is_match": true,
      "supporting_spans": [
        {
          "text": "depressed",
          "start": 0,
          "end": 9,
          "confidence": 0.92,
          "label": "relevant_span"
        }
      ]
    }
  ]
}
```

## Model Details

### BGE-M3 Embedding Model
- **Model**: `BAAI/bge-m3`
- **Dimensions**: 1024
- **Features**: Multilingual, high-quality embeddings
- **Optimization**: FP16 precision for RTX 3090

### SpanBERT Model
- **Model**: `SpanBERT/spanbert-base-cased`
- **Purpose**: Criteria filtering and token extraction
- **Features**: Span extraction with confidence scores
- **Optimization**: Half precision and compiled execution

### FAISS Index
- **Type**: IVFFlat with cosine similarity
- **Features**: GPU acceleration, efficient search
- **Configuration**: 100 clusters, 10 probes

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in settings
   - Enable gradient checkpointing
   - Clear GPU cache periodically

2. **Model Loading Errors**:
   - Check internet connection for model downloads
   - Verify CUDA installation
   - Ensure sufficient disk space

3. **FAISS Index Errors**:
   - Rebuild index if corrupted
   - Check embedding dimensions match
   - Verify file permissions

### Performance Tips

1. **For RTX 3090**:
   - Use mixed precision (FP16)
   - Enable TensorFloat-32
   - Set appropriate batch sizes
   - Monitor GPU memory usage

2. **For Large Datasets**:
   - Process in batches
   - Save intermediate results
   - Use efficient data loading
   - Monitor memory usage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{rag_dsm5_matching,
  title={RAG System for DSM-5 Criteria Matching},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Psy_RAG}
}
```

## Acknowledgments

- BAAI for the BGE-M3 embedding model
- Facebook Research for SpanBERT
- Meta AI for FAISS
- The DSM-5 criteria dataset
