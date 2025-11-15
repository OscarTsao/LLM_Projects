"""
Define the RAG methods used for each Agent, Dense and Sparse retriever.
"""
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Tuple
import string
import re
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import pickle
import os

class SparseRetriever:
    def __init__(self, criteria_texts: list):
        self.criteria_texts = criteria_texts
        self.preprocessed_corpus = [self.preprocess(text.text) for text in criteria_texts]
        self.bm25 = BM25Okapi(self.preprocessed_corpus)
        
    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 - tokenize, lowercase, remove punctuation
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace and split
        tokens = text.split()
        
        # Remove very short tokens (optional)
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Retrieve most relevant documents using BM25
        Returns: List of (document_index, score) tuples
        """
        # Preprocess query
        query_tokens = self.preprocess(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Return results with positive scores
        results = [(int(idx), float(scores[idx])) 
                  for idx in top_indices if scores[idx] > 0]
        
        return results
    
    def get_batch_scores(self, queries: List[str]) -> List[List[float]]:
        """
        Get BM25 scores for multiple queries at once
        """
        batch_results = []
        for query in queries:
            query_tokens = self.preprocess(query)
            scores = self.bm25.get_scores(query_tokens)
            batch_results.append(scores.tolist())
        
        return batch_results

class DenseRetriever:
    def __init__(self, criteria_texts: list, model_name: str = "BAAI/bge-base-en-v1.5", 
                 cache_dir: str = "cache", device: str = None):
        """
        Initialize Dense Retriever with BGE model and FAISS index
        
        Args:
            criteria_texts: List of criteria text objects
            model_name: BGE model name from HuggingFace
            cache_dir: Directory to cache embeddings and index
            device: Device to run model on (auto-detect if None)
        """
        self.criteria_texts = criteria_texts
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize model and tokenizer
        print(f"Loading BGE model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize FAISS index
        self.dimension = 768  # BGE-base dimension
        self.index = None
        self.embeddings = None
        
        # Build or load index
        self._build_index()
        
    def _encode_texts(self, texts: List[str], batch_size: int = 32, 
                     show_progress: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings using BGE model
        """
        embeddings = []
        
        if show_progress:
            from tqdm import tqdm
            text_iter = tqdm(range(0, len(texts), batch_size), desc="Encoding texts")
        else:
            text_iter = range(0, len(texts), batch_size)
            
        with torch.no_grad():
            for i in text_iter:
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get embeddings
                outputs = self.model(**inputs)
                
                # Use [CLS] token embedding (first token)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # L2 normalize embeddings
                batch_embeddings = batch_embeddings / np.linalg.norm(
                    batch_embeddings, axis=1, keepdims=True
                )
                
                embeddings.append(batch_embeddings)
                
                # Clear cache to prevent memory buildup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return np.vstack(embeddings).astype(np.float32)
    
    def _build_index(self):
        """
        Build or load FAISS index from criteria texts
        """
        index_path = os.path.join(self.cache_dir, "faiss_index.bin")
        embeddings_path = os.path.join(self.cache_dir, "embeddings.pkl")
        
        # Try to load existing index
        if os.path.exists(index_path) and os.path.exists(embeddings_path):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(index_path)
            with open(embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            print(f"Loaded index with {self.index.ntotal} vectors")
            return
        
        # Build new index
        print("Building new FAISS index...")
        
        # Extract text content
        texts = [criteria.text for criteria in self.criteria_texts]
        
        # Generate embeddings
        print(f"Encoding {len(texts)} criteria texts...")
        self.embeddings = self._encode_texts(texts)
        
        # Build FAISS index (using IndexFlatIP for cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
        self.index.add(self.embeddings)
        
        # Save index and embeddings
        faiss.write_index(self.index, index_path)
        with open(embeddings_path, 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        print(f"Built and saved index with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Retrieve most relevant documents using dense embeddings
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of (document_index, similarity_score) tuples
        """
        # Encode query
        query_embedding = self._encode_texts([query], show_progress=False)
        
        # Search in FAISS index
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Convert to list of tuples
        results = [
            (int(indices[0][i]), float(similarities[0][i])) 
            for i in range(len(indices[0])) 
            if similarities[0][i] > 0  # Filter positive similarities
        ]
        
        return results
    
    def retrieve_with_texts(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve most relevant documents with actual text content
        
        Returns:
            List of (criteria_text, similarity_score) tuples
        """
        indices_scores = self.retrieve(query, top_k)
        
        return [
            (self.criteria_texts[idx].text, score) 
            for idx, score in indices_scores
        ]
    
    def get_batch_similarities(self, queries: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Get similarities for multiple queries at once
        
        Args:
            queries: List of query strings
            batch_size: Batch size for encoding queries
            
        Returns:
            List of similarity scores for each query
        """
        # Encode all queries
        query_embeddings = self._encode_texts(queries, batch_size=batch_size)
        
        # Get similarities against all documents
        similarities, _ = self.index.search(query_embeddings, self.index.ntotal)
        
        return similarities.tolist()
    
    def add_documents(self, new_texts: List[str]):
        """
        Add new documents to the existing index
        
        Args:
            new_texts: List of new text strings to add
        """
        # Encode new texts
        new_embeddings = self._encode_texts(new_texts)
        
        # Add to index
        self.index.add(new_embeddings)
        
        # Update stored embeddings
        self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        print(f"Added {len(new_texts)} new documents. Total: {self.index.ntotal}")
    
    def save_index(self, path: str = None):
        """Save the current index to disk"""
        if path is None:
            path = os.path.join(self.cache_dir, "faiss_index.bin")
        
        faiss.write_index(self.index, path)
        
        # Save embeddings too
        embeddings_path = path.replace(".bin", "_embeddings.pkl")
        with open(embeddings_path, 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        print(f"Index saved to {path}")

class HybridRetriever:
    def __init__(self, criteria_texts: list, sparse_weight: float = 0.5, 
                 dense_weight: float = 0.5, **kwargs):
        """
        Initialize Hybrid Retriever combining sparse and dense methods
        
        Args:
            criteria_texts: List of criteria text objects
            sparse_weight: Weight for sparse (BM25) scores
            dense_weight: Weight for dense (BGE) scores
            **kwargs: Additional arguments for DenseRetriever
        """
        self.criteria_texts = criteria_texts
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        
        # Initialize both retrievers
        print("Initializing sparse retriever...")
        self.sparse_retriever = SparseRetriever(criteria_texts)
        
        print("Initializing dense retriever...")
        self.dense_retriever = DenseRetriever(criteria_texts, **kwargs)
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve using hybrid approach (combine sparse + dense)
        
        Returns:
            List of (criteria_text, combined_score) tuples
        """
        # Get sparse results
        sparse_results = dict(self.sparse_retriever.retrieve(query, top_k * 2))
        
        # Get dense results  
        dense_results = dict(self.dense_retriever.retrieve(query, top_k * 2))
        
        # Combine scores
        combined_scores = {}
        all_indices = set(sparse_results.keys()) | set(dense_results.keys())
        
        for idx in all_indices:
            sparse_score = sparse_results.get(idx, 0.0)
            dense_score = dense_results.get(idx, 0.0)
            
            # Normalize scores (optional)
            sparse_score = sparse_score / max(sparse_results.values()) if sparse_results else 0
            dense_score = dense_score / max(dense_results.values()) if dense_results else 0
            
            # Combine with weights
            combined_score = (self.sparse_weight * sparse_score + 
                            self.dense_weight * dense_score)
            combined_scores[idx] = combined_score
        
        # Sort by combined score and get top-k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Return with text content
        return [
            (self.criteria_texts[idx].text, score) 
            for idx, score in sorted_results
        ]
    
    def set_weights(self, sparse_weight: float, dense_weight: float):
        """Update the weights for combining sparse and dense scores"""
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        print(f"Updated weights: sparse={sparse_weight}, dense={dense_weight}")