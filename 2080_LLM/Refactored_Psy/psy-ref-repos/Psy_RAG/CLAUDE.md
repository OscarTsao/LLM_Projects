# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a RAG (Retrieval-Augmented Generation) system for matching social media posts against DSM-5 psychiatric criteria using BGE-M3 embeddings, FAISS vector search, and SpanBERT filtering. The system is optimized for RTX 3090 GPU performance.

## Common Commands

### Testing
```bash
# Run all tests with coverage
python run_tests.py

# Run specific test files
pytest tests/test_rag_pipeline.py -v
pytest tests/test_embedding_model.py -v
pytest tests/test_faiss_index.py -v
pytest tests/test_data_loader.py -v

# Run tests with coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
```

### Main Operations
```bash
# Build FAISS index from DSM-5 criteria
python main.py --mode build_index --save_index

# Evaluate posts against criteria
python main.py --mode evaluate --num_posts 100 --load_index data/indices/faiss_index

# Analyze single post
python main.py --mode single_post --post_text "I feel depressed" --load_index data/indices/faiss_index

# Custom thresholds
python main.py --mode evaluate --similarity_threshold 0.8 --spanbert_threshold 0.6 --top_k 15
```

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Install with GPU support
pip install -e .[gpu]

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

## Architecture

### Core Components

The system follows a modular architecture with clear separation of concerns:

1. **RAGPipeline** (`src/models/rag_pipeline.py`): Main orchestrator that coordinates all components
2. **BGEEmbeddingModel** (`src/models/embedding_model.py`): BGE-M3 model wrapper for generating embeddings
3. **FAISSIndex** (`src/models/faiss_index.py`): Vector database for similarity search with GPU acceleration
4. **SpanBERTModel** (`src/models/spanbert_model.py`): Filtering and span extraction model
5. **DataLoader** (`src/utils/data_loader.py`): Handles CSV/JSON data loading and preprocessing

### Data Flow

1. **Index Building**: DSM-5 criteria → BGE-M3 embeddings → FAISS index
2. **Query Processing**: Social media post → BGE-M3 embedding → FAISS search → SpanBERT filtering → Results

### Configuration

All settings are centralized in `src/config/settings.py`:
- Model configurations (BGE-M3, SpanBERT)
- FAISS index settings (type, metric, clustering)
- RAG parameters (thresholds, batch sizes)
- RTX 3090 optimizations (mixed precision, compile flags)

### Data Expectations

- **Posts**: CSV file with social media posts at `Data/translated_posts.csv`
- **Criteria**: JSON file with DSM-5 criteria at `Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json`
- **Generated files**: Saved to `data/embeddings/` and `data/indices/`
- **Results**: Output to `results/` directory with timestamps

### GPU Optimizations

The system includes RTX 3090 specific optimizations:
- Mixed precision (FP16) for 2x speed improvement
- TensorFloat-32 for matrix operations
- cuDNN benchmark mode
- Torch compilation for faster inference
- Automatic memory management

### Error Handling

The system includes comprehensive error handling:
- Graceful GPU fallback to CPU
- Data validation and preprocessing
- Model loading error recovery
- FAISS index corruption handling

## Testing Structure

Tests are organized by component:
- `test_data_loader.py`: CSV/JSON loading, preprocessing validation
- `test_embedding_model.py`: BGE-M3 embedding generation, batch processing
- `test_faiss_index.py`: Vector indexing, similarity search, GPU operations
- `test_rag_pipeline.py`: End-to-end pipeline testing, result validation

All tests include both CPU and GPU test paths where applicable.

## Development Notes

- The system automatically detects CUDA availability and optimizes accordingly
- FAISS indices can be saved/loaded to avoid rebuilding for repeated runs
- Batch processing is used throughout for memory efficiency
- Results include detailed metrics and processing times for performance monitoring