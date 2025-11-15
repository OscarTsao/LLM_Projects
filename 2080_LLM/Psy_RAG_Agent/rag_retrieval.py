#!/usr/bin/env python3
"""
RAG Retrieval System for DSM-5 Criteria
This module handles the retrieval of relevant DSM-5 criteria for given posts.
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import pickle


class DSMCriteriaRetriever:
    """RAG retrieval system for DSM-5 criteria"""

    def __init__(self, model_name='all-MiniLM-L6-v2', index_path='./dsm_criteria_index'):
        """
        Initialize the retriever

        Args:
            model_name: Sentence transformer model for embeddings
            index_path: Path to save/load the FAISS index
        """
        self.model_name = model_name
        self.index_path = index_path
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.criteria_data = []
        self.criteria_texts = []
        self.criteria_metadata = []

    def load_criteria(self, criteria_path: str):
        """Load DSM-5 criteria from JSON file"""
        print("Loading DSM-5 criteria...")

        with open(criteria_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        self.criteria_data = []
        self.criteria_texts = []
        self.criteria_metadata = []

        for disorder in raw_data:
            disorder_name = disorder['diagnosis']
            for criterion in disorder['criteria']:
                criterion_id = criterion['id']
                criterion_text = criterion['text']

                # Store the criterion data
                self.criteria_data.append({
                    'disorder': disorder_name,
                    'criterion_id': criterion_id,
                    'text': criterion_text,
                    'full_key': f"{disorder_name} - {criterion_id}"
                })

                # Store text for embedding
                self.criteria_texts.append(criterion_text)

                # Store metadata for retrieval
                self.criteria_metadata.append({
                    'disorder': disorder_name,
                    'criterion_id': criterion_id,
                    'full_key': f"{disorder_name} - {criterion_id}"
                })

        print(f"Loaded {len(self.criteria_texts)} criteria from {len(raw_data)} disorders")
        return self.criteria_data

    def build_index(self, save_index=True):
        """Build FAISS index from criteria embeddings"""
        print("Building embeddings for criteria...")

        if not self.criteria_texts:
            raise ValueError("No criteria loaded. Call load_criteria() first.")

        # Generate embeddings
        embeddings = self.encoder.encode(self.criteria_texts, show_progress_bar=True)
        embeddings = embeddings.astype('float32')

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        print(f"Built FAISS index with {self.index.ntotal} criteria")

        if save_index:
            self.save_index()

        return self.index

    def save_index(self):
        """Save FAISS index and metadata to disk"""
        os.makedirs(self.index_path, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, os.path.join(self.index_path, 'criteria.index'))

        # Save metadata
        with open(os.path.join(self.index_path, 'criteria_metadata.pkl'), 'wb') as f:
            pickle.dump({
                'criteria_data': self.criteria_data,
                'criteria_metadata': self.criteria_metadata,
                'model_name': self.model_name
            }, f)

        print(f"Saved index to {self.index_path}")

    def load_index(self):
        """Load FAISS index and metadata from disk"""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index path {self.index_path} not found")

        # Load FAISS index
        self.index = faiss.read_index(os.path.join(self.index_path, 'criteria.index'))

        # Load metadata
        with open(os.path.join(self.index_path, 'criteria_metadata.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.criteria_data = data['criteria_data']
            self.criteria_metadata = data['criteria_metadata']
            if data['model_name'] != self.model_name:
                print(f"Warning: Loaded index was built with {data['model_name']}, but using {self.model_name}")

        print(f"Loaded index with {self.index.ntotal} criteria")
        return self.index

    def retrieve(self, query: str, top_k: int = 10, threshold: float = 0.3) -> List[Dict]:
        """
        Retrieve relevant criteria for a given query

        Args:
            query: The post text to search for
            top_k: Number of top criteria to retrieve
            threshold: Similarity threshold (0-1, higher means more similar)

        Returns:
            List of retrieved criteria with metadata and scores
        """
        if self.index is None:
            raise ValueError("Index not built or loaded. Call build_index() or load_index() first.")

        # Encode query
        query_embedding = self.encoder.encode([query])
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        # Filter by threshold and format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold:
                criterion = self.criteria_data[idx]
                results.append({
                    'criterion': criterion,
                    'score': float(score),
                    'text': criterion['text'],
                    'disorder': criterion['disorder'],
                    'criterion_id': criterion['criterion_id'],
                    'full_key': criterion['full_key']
                })

        return results

    def get_retrieved_criteria_for_classification(self, post_text: str, top_k: int = 10,
                                                threshold: float = 0.3) -> Dict[str, str]:
        """
        Get retrieved criteria formatted for classification

        Args:
            post_text: The post text
            top_k: Number of criteria to retrieve
            threshold: Similarity threshold

        Returns:
            Dictionary mapping criterion keys to their texts
        """
        retrieved = self.retrieve(post_text, top_k, threshold)
        return {item['full_key']: item['text'] for item in retrieved}


def main():
    """Test the retrieval system"""
    print("Testing DSM Criteria Retriever...")

    # Initialize retriever
    retriever = DSMCriteriaRetriever()

    # Load criteria
    retriever.load_criteria('Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json')

    # Build index
    retriever.build_index()

    # Test retrieval
    test_post = "I feel very sad and hopeless every day. I can't sleep and have lost my appetite."
    results = retriever.retrieve(test_post, top_k=5, threshold=0.2)

    print(f"\nRetrieved {len(results)} criteria for: '{test_post}'")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['score']:.3f}] {result['disorder']} - {result['criterion_id']}")
        print(f"   Text: {result['text'][:100]}...")
        print()


if __name__ == "__main__":
    main()