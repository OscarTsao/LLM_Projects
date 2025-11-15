"""
FAISS index implementation for efficient similarity search
"""
import faiss
import numpy as np
import pickle
from typing import List, Tuple, Optional, Dict
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS index for efficient similarity search with optimizations for RTX 3090"""
    
    def __init__(
        self, 
        dimension: int, 
        index_type: str = "IVFFlat",
        metric: str = "cosine",
        nprobe: int = 10
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.nprobe = nprobe
        self.index = None
        self.id_to_text = {}
        self.text_to_id = {}
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index based on configuration"""
        try:
            if self.metric == "cosine":
                # For cosine similarity, we need to normalize vectors
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
            
            # Use GPU if available
            try:
                if faiss.get_num_gpus() > 0:
                    logger.info("Using GPU for FAISS index")
                    gpu_res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.index)
                else:
                    logger.info("Using CPU for FAISS index (GPU not available)")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU FAISS, falling back to CPU: {e}")
                # Index is already CPU-based, no need to change
            
            # Convert to IVFFlat for better performance on large datasets
            if self.index_type == "IVFFlat" and self.metric == "cosine":
                quantizer = faiss.IndexFlatIP(self.dimension)
                # Use fewer clusters for small datasets
                n_clusters = min(100, max(1, self.dimension // 4))
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters)
                
                # Move to GPU if available
                try:
                    if faiss.get_num_gpus() > 0:
                        gpu_res = faiss.StandardGpuResources()
                        self.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.index)
                except Exception as e:
                    logger.warning(f"Failed to move IVFFlat index to GPU, using CPU: {e}")
            
            logger.info(f"Created FAISS index: {self.index_type}, metric: {self.metric}")
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            raise
    
    def add_embeddings(
        self, 
        embeddings: np.ndarray, 
        texts: List[str], 
        ids: Optional[List[str]] = None
    ):
        """
        Add embeddings to the index
        
        Args:
            embeddings: numpy array of embeddings
            texts: List of corresponding texts
            ids: Optional list of IDs for texts
        """
        try:
            if embeddings.shape[1] != self.dimension:
                raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match index dimension {self.dimension}")
            
            # Normalize embeddings for cosine similarity
            if self.metric == "cosine":
                faiss.normalize_L2(embeddings)
            
            # Generate IDs if not provided
            if ids is None:
                ids = [f"text_{i}" for i in range(len(texts))]
            
            # Add to index
            if self.index_type == "IVFFlat" and not self.index.is_trained:
                # Train the index first
                logger.info("Training IVFFlat index...")
                self.index.train(embeddings)
            
            self.index.add(embeddings)
            
            # Store text mappings
            for i, (text, text_id) in enumerate(zip(texts, ids)):
                self.id_to_text[i] = text
                self.text_to_id[text] = i
                # Also store text_id to text mapping
                if not hasattr(self, 'text_id_to_text'):
                    self.text_id_to_text = {}
                self.text_id_to_text[text_id] = text
            
            logger.info(f"Added {len(embeddings)} embeddings to index")
            
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            raise
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[Tuple[str, float, int]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of tuples (text, similarity_score, index)
        """
        try:
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Normalize query for cosine similarity
            if self.metric == "cosine":
                faiss.normalize_L2(query_embedding)
            
            # Set nprobe for IVFFlat
            if self.index_type == "IVFFlat":
                self.index.nprobe = self.nprobe
            
            # Search
            similarities, indices = self.index.search(query_embedding, k)
            
            # Process results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1:  # Invalid index
                    continue
                
                if similarity >= similarity_threshold:
                    text = self.id_to_text.get(idx, f"unknown_{idx}")
                    results.append((text, float(similarity), int(idx)))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching index: {e}")
            return []
    
    def search_batch(
        self, 
        query_embeddings: np.ndarray, 
        k: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[List[Tuple[str, float, int]]]:
        """
        Search for multiple queries at once
        
        Args:
            query_embeddings: numpy array of query embeddings
            k: Number of results per query
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of results for each query
        """
        try:
            # Normalize queries for cosine similarity
            if self.metric == "cosine":
                faiss.normalize_L2(query_embeddings)
            
            # Set nprobe for IVFFlat
            if self.index_type == "IVFFlat":
                self.index.nprobe = self.nprobe
            
            # Search
            similarities, indices = self.index.search(query_embeddings, k)
            
            # Process results
            all_results = []
            for query_idx in range(len(query_embeddings)):
                query_results = []
                for i, (similarity, idx) in enumerate(zip(similarities[query_idx], indices[query_idx])):
                    if idx == -1:  # Invalid index
                        continue
                    
                    if similarity >= similarity_threshold:
                        text = self.id_to_text.get(idx, f"unknown_{idx}")
                        query_results.append((text, float(similarity), int(idx)))
                
                all_results.append(query_results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error in batch search: {e}")
            return []
    
    def get_text_by_id(self, text_id: str) -> Optional[str]:
        """Get text by ID"""
        return getattr(self, 'text_id_to_text', {}).get(text_id)
    
    def get_id_by_text(self, text: str) -> Optional[int]:
        """Get ID by text"""
        return self.text_to_id.get(text)
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "nprobe": self.nprobe
        }
    
    def save_index(self, filepath: Path):
        """Save index and mappings to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(filepath / "faiss_index.bin"))
            
            # Save mappings
            mappings = {
                "id_to_text": self.id_to_text,
                "text_to_id": self.text_to_id,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric": self.metric,
                "nprobe": self.nprobe
            }
            
            with open(filepath / "mappings.json", 'w') as f:
                json.dump(mappings, f, indent=2)
            
            logger.info(f"Index saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    def load_index(self, filepath: Path):
        """Load index and mappings from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(filepath / "faiss_index.bin"))
            
            # Load mappings
            with open(filepath / "mappings.json", 'r') as f:
                mappings = json.load(f)
            
            self.id_to_text = {int(k): v for k, v in mappings["id_to_text"].items()}
            self.text_to_id = mappings["text_to_id"]
            self.dimension = mappings["dimension"]
            self.index_type = mappings["index_type"]
            self.metric = mappings["metric"]
            self.nprobe = mappings["nprobe"]
            
            # Move to GPU if available
            try:
                if faiss.get_num_gpus() > 0:
                    gpu_res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.index)
            except Exception as e:
                logger.warning(f"Failed to move loaded index to GPU, using CPU: {e}")
            
            logger.info(f"Index loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise
