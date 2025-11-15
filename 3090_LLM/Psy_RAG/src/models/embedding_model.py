"""
BGE-M3 embedding model implementation with optimizations for RTX 3090
"""
# Standard library
import gc
import logging
from pathlib import Path
from typing import List, Optional

# Third-party
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Local imports
from ..utils.performance_optimizer import PerformanceOptimizer

logger = logging.getLogger(__name__)


class BGEEmbeddingModel:
    """Optimized BGE-M3 embedding model for RTX 3090"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.optimizer = PerformanceOptimizer()
        self._load_model()
    
    def _load_model(self):
        """Load the BGE-M3 model with optimizations"""
        try:
            logger.info(f"Loading BGE-M3 model: {self.model_name}")
            
            # Load model with optimizations
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True
            )
            
            # Apply performance optimizations for RTX 3090
            if self.device == "cuda":
                self.model = self.optimizer.optimize_model_for_inference(self.model)
            
            logger.info("BGE-M3 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading BGE-M3 model: {e}")
            raise
    
    def encode_texts(
        self, 
        texts: List[str], 
        batch_size: int = 16,
        show_progress: bool = True,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings with optimizations
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            normalize_embeddings: Whether to normalize embeddings
            
        Returns:
            numpy array of embeddings
        """
        try:
            logger.info(f"Encoding {len(texts)} texts with batch size {batch_size}")
            
            # Process in batches to avoid memory issues
            all_embeddings = []
            
            for i in tqdm(range(0, len(texts), batch_size), 
                         desc="Encoding texts", 
                         disable=not show_progress):
                batch_texts = texts[i:i + batch_size]
                
                # Encode batch
                with torch.no_grad():
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_tensor=True,
                        normalize_embeddings=normalize_embeddings,
                        show_progress_bar=False
                    )
                    
                    # Convert to numpy and add to results
                    all_embeddings.append(batch_embeddings.cpu().numpy())
                
                # Clear cache to prevent memory issues
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            
            # Concatenate all embeddings
            embeddings = np.vstack(all_embeddings)
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
    
    def encode_single(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        """Encode a single text"""
        try:
            with torch.no_grad():
                embedding = self.model.encode(
                    text,
                    convert_to_tensor=True,
                    normalize_embeddings=normalize_embeddings
                )
                return embedding.cpu().numpy()
        except Exception as e:
            logger.error(f"Error encoding single text: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        # BGE-M3 has 1024 dimensions
        return 1024
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: Path):
        """Save embeddings to file"""
        try:
            np.save(filepath, embeddings)
            logger.info(f"Embeddings saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            raise
    
    def load_embeddings(self, filepath: Path) -> np.ndarray:
        """Load embeddings from file"""
        try:
            embeddings = np.load(filepath)
            logger.info(f"Embeddings loaded from {filepath}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
