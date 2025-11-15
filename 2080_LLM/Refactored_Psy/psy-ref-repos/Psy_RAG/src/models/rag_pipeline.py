"""
Main RAG pipeline for criteria matching using BGE-M3 and SpanBERT
"""
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass
from tqdm import tqdm

from .embedding_model import BGEEmbeddingModel
from .faiss_index import FAISSIndex
from .spanbert_model import SpanBERTModel, SpanResult
from ..utils.data_loader import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class CriteriaMatch:
    """Result of criteria matching"""
    criteria_id: str
    diagnosis: str
    criterion_text: str
    similarity_score: float
    spanbert_score: float
    supporting_spans: List[SpanResult]
    is_match: bool


@dataclass
class RAGResult:
    """Complete RAG result for a post"""
    post_id: int
    post_text: str
    matched_criteria: List[CriteriaMatch]
    total_matches: int
    processing_time: float


class RAGPipeline:
    """Main RAG pipeline for DSM-5 criteria matching"""
    
    def __init__(
        self,
        posts_path: Path,
        criteria_path: Path,
        embedding_model_name: str = "BAAI/bge-m3",
        spanbert_model_name: str = "SpanBERT/spanbert-base-cased",
        device: str = "cuda",
        similarity_threshold: float = 0.7,
        spanbert_threshold: float = 0.5,
        top_k: int = 10
    ):
        self.posts_path = posts_path
        self.criteria_path = criteria_path
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.spanbert_threshold = spanbert_threshold
        self.top_k = top_k
        
        # Initialize models
        self.embedding_model = BGEEmbeddingModel(embedding_model_name, device)
        self.spanbert_model = SpanBERTModel(spanbert_model_name, device)
        
        # Initialize data loader
        self.data_loader = DataLoader(posts_path, criteria_path)
        
        # Initialize FAISS index
        self.faiss_index = None
        self.criteria_data = None
        self.posts_data = None
        
        logger.info("RAG Pipeline initialized")
    
    def build_index(self, save_path: Optional[Path] = None):
        """Build FAISS index from criteria data"""
        try:
            logger.info("Building FAISS index...")
            
            # Load criteria data
            criteria_raw = self.data_loader.load_criteria()
            self.criteria_data = self.data_loader.preprocess_criteria(criteria_raw)
            
            # Extract texts for embedding
            criteria_texts = self.data_loader.get_texts_for_embedding(self.criteria_data)
            
            # Generate embeddings
            logger.info("Generating criteria embeddings...")
            criteria_embeddings = self.embedding_model.encode_texts(
                criteria_texts,
                batch_size=16,
                show_progress=True
            )
            
            # Create FAISS index
            embedding_dim = self.embedding_model.get_embedding_dimension()
            self.faiss_index = FAISSIndex(
                dimension=embedding_dim,
                index_type="IVFFlat",
                metric="cosine",
                nprobe=10
            )
            
            # Add embeddings to index
            criteria_ids = [item['id'] for item in self.criteria_data]
            self.faiss_index.add_embeddings(
                criteria_embeddings,
                criteria_texts,
                criteria_ids
            )
            
            logger.info(f"Index built with {len(criteria_texts)} criteria")
            
            # Save index if path provided
            if save_path:
                save_path.mkdir(parents=True, exist_ok=True)
                self.faiss_index.save_index(save_path)
                logger.info(f"Index saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise
    
    def load_index(self, load_path: Path):
        """Load pre-built FAISS index"""
        try:
            logger.info(f"Loading index from {load_path}")
            
            # Load criteria data
            criteria_raw = self.data_loader.load_criteria()
            self.criteria_data = self.data_loader.preprocess_criteria(criteria_raw)
            
            # Create and load FAISS index
            embedding_dim = self.embedding_model.get_embedding_dimension()
            self.faiss_index = FAISSIndex(
                dimension=embedding_dim,
                index_type="IVFFlat",
                metric="cosine",
                nprobe=10
            )
            self.faiss_index.load_index(load_path)
            
            logger.info("Index loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise
    
    def process_post(self, post_text: str, post_id: int = 0) -> RAGResult:
        """
        Process a single post through the RAG pipeline
        
        Args:
            post_text: The social media post text
            post_id: Optional post ID
            
        Returns:
            RAGResult with matched criteria
        """
        import time
        start_time = time.time()
        
        try:
            # Step 1: Generate embedding for post
            post_embedding = self.embedding_model.encode_single(post_text)
            
            # Step 2: Retrieve similar criteria using FAISS
            similar_criteria = self.faiss_index.search(
                post_embedding,
                k=self.top_k,
                similarity_threshold=self.similarity_threshold
            )
            
            # Step 3: Filter criteria using SpanBERT
            matched_criteria = []
            for criteria_text, similarity_score, idx in similar_criteria:
                # Find the criteria data
                criteria_item = None
                for item in self.criteria_data:
                    if item['text'] == criteria_text:
                        criteria_item = item
                        break
                
                if not criteria_item:
                    continue
                
                # Use SpanBERT to filter and extract supporting evidence
                spanbert_matches = self.spanbert_model.filter_criteria_matches(
                    post_text,
                    [criteria_text],
                    confidence_threshold=self.spanbert_threshold
                )
                
                if spanbert_matches:
                    criteria_text_spanbert, spanbert_score, supporting_spans = spanbert_matches[0]
                    
                    # Determine if it's a match based on both similarity and SpanBERT score
                    is_match = (similarity_score >= self.similarity_threshold and 
                              spanbert_score >= self.spanbert_threshold)
                    
                    criteria_match = CriteriaMatch(
                        criteria_id=criteria_item['id'],
                        diagnosis=criteria_item['diagnosis'],
                        criterion_text=criteria_item['text'],
                        similarity_score=similarity_score,
                        spanbert_score=spanbert_score,
                        supporting_spans=supporting_spans,
                        is_match=is_match
                    )
                    
                    matched_criteria.append(criteria_match)
            
            # Sort by combined score
            matched_criteria.sort(
                key=lambda x: (x.similarity_score + x.spanbert_score) / 2,
                reverse=True
            )
            
            processing_time = time.time() - start_time
            
            return RAGResult(
                post_id=post_id,
                post_text=post_text,
                matched_criteria=matched_criteria,
                total_matches=len([m for m in matched_criteria if m.is_match]),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing post: {e}")
            return RAGResult(
                post_id=post_id,
                post_text=post_text,
                matched_criteria=[],
                total_matches=0,
                processing_time=time.time() - start_time
            )
    
    def process_posts_batch(
        self, 
        posts: List[Dict], 
        batch_size: int = 32
    ) -> List[RAGResult]:
        """
        Process multiple posts in batches
        
        Args:
            posts: List of post dictionaries with 'text' and 'id' keys
            batch_size: Batch size for processing
            
        Returns:
            List of RAGResult objects
        """
        results = []
        
        try:
            logger.info(f"Processing {len(posts)} posts in batches of {batch_size}")
            
            for i in tqdm(range(0, len(posts), batch_size), desc="Processing posts"):
                batch_posts = posts[i:i + batch_size]
                
                # Process batch
                batch_results = []
                for post in batch_posts:
                    result = self.process_post(
                        post['text'], 
                        post.get('id', i)
                    )
                    batch_results.append(result)
                
                results.extend(batch_results)
                
                # Clear cache periodically
                if i % (batch_size * 4) == 0:
                    self.embedding_model.clear_cache()
                    self.spanbert_model.clear_cache()
            
            logger.info(f"Processed {len(results)} posts")
            return results
            
        except Exception as e:
            logger.error(f"Error processing posts batch: {e}")
            return results
    
    def evaluate_posts(self, num_posts: Optional[int] = None) -> List[RAGResult]:
        """
        Evaluate posts from the dataset
        
        Args:
            num_posts: Number of posts to evaluate (None for all)
            
        Returns:
            List of RAGResult objects
        """
        try:
            # Load posts
            posts_df = self.data_loader.load_posts()
            posts_data = self.data_loader.preprocess_posts(posts_df)
            
            # Limit number of posts if specified
            if num_posts:
                posts_data = posts_data[:num_posts]
            
            # Process posts
            results = self.process_posts_batch(posts_data, batch_size=16)
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating posts: {e}")
            return []
    
    def get_statistics(self, results: List[RAGResult]) -> Dict:
        """Get statistics from RAG results"""
        try:
            total_posts = len(results)
            total_matches = sum(r.total_matches for r in results)
            avg_processing_time = np.mean([r.processing_time for r in results])
            
            # Count matches by diagnosis
            diagnosis_counts = {}
            for result in results:
                for match in result.matched_criteria:
                    if match.is_match:
                        diagnosis_counts[match.diagnosis] = diagnosis_counts.get(match.diagnosis, 0) + 1
            
            return {
                "total_posts": total_posts,
                "total_matches": total_matches,
                "avg_matches_per_post": total_matches / total_posts if total_posts > 0 else 0,
                "avg_processing_time": avg_processing_time,
                "diagnosis_counts": diagnosis_counts,
                "posts_with_matches": len([r for r in results if r.total_matches > 0])
            }
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {}
    
    def save_results(self, results: List[RAGResult], filepath: Path):
        """Save results to JSON file"""
        try:
            # Convert results to serializable format
            serializable_results = []
            for result in results:
                serializable_result = {
                    "post_id": result.post_id,
                    "post_text": result.post_text,
                    "total_matches": result.total_matches,
                    "processing_time": result.processing_time,
                    "matched_criteria": []
                }
                
                for match in result.matched_criteria:
                    serializable_match = {
                        "criteria_id": match.criteria_id,
                        "diagnosis": match.diagnosis,
                        "criterion_text": match.criterion_text,
                        "similarity_score": match.similarity_score,
                        "spanbert_score": match.spanbert_score,
                        "is_match": match.is_match,
                        "supporting_spans": [
                            {
                                "text": span.text,
                                "start": span.start,
                                "end": span.end,
                                "confidence": span.confidence,
                                "label": span.label
                            }
                            for span in match.supporting_spans
                        ]
                    }
                    serializable_result["matched_criteria"].append(serializable_match)
                
                serializable_results.append(serializable_result)
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
