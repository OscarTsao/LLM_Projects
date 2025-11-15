"""
Configuration settings for the RAG system
"""
import os
from pathlib import Path

# Try to import torch, fallback if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "Data"
SRC_DIR = BASE_DIR / "src"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Data paths
POSTS_CSV = DATA_DIR / "mental_health_posts_negative_generated.csv"
CRITERIA_JSON = DATA_DIR / "DSM-5" / "DSM_Criteria_Array_Fixed_Simplify.json"

# Model configurations
BGE_MODEL_NAME = "BAAI/bge-m3"
SPANBERT_MODEL_NAME = "SpanBERT/spanbert-base-cased"

# FAISS settings
FAISS_INDEX_TYPE = "IVFFlat"  # Can be "Flat", "IVFFlat", "IVFPQ"
FAISS_METRIC = "cosine"
FAISS_NPROBE = 10

# RAG settings
TOP_K_RETRIEVAL = 10
SIMILARITY_THRESHOLD = 0.7
MAX_POST_LENGTH = 512
MAX_CRITERIA_LENGTH = 256

# GPU settings
DEVICE = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 4

# SpanBERT settings
SPANBERT_CONFIDENCE_THRESHOLD = 0.5
SPANBERT_MAX_SPANS = 5

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance optimization for RTX 3090
TORCH_COMPILE = True
MIXED_PRECISION = True
GRADIENT_CHECKPOINTING = True

# Advanced RTX 3090 optimizations
RTX_3090_OPTIMIZATIONS = {
    'memory_fraction': 0.85,  # Use 85% of GPU memory
    'batch_size_multiplier': 2,  # Double batch size for RTX 3090
    'enable_tf32': True,
    'enable_cudnn_benchmark': True,
    'enable_flash_attention': True,
    'compile_mode': 'max-autotune',  # Aggressive optimization
    'gradient_checkpointing': True,
    'mixed_precision': 'fp16'
}

# Batch size optimization based on GPU memory
OPTIMAL_BATCH_SIZES = {
    'embedding_batch_size': BATCH_SIZE * 2 if DEVICE == "cuda" else BATCH_SIZE,
    'faiss_search_batch_size': 64 if DEVICE == "cuda" else 16,
    'spanbert_batch_size': 8 if DEVICE == "cuda" else 4
}

# Update batch size based on GPU availability
BATCH_SIZE = OPTIMAL_BATCH_SIZES['embedding_batch_size']

# Ensure directories exist
for dir_path in [DATA_DIR, RESULTS_DIR, LOGS_DIR, DATA_DIR / "indices", DATA_DIR / "embeddings"]:
    dir_path.mkdir(parents=True, exist_ok=True)
