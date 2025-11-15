"""
Tests for RAG pipeline functionality
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from src.models.rag_pipeline import RAGPipeline, CriteriaMatch, RAGResult
from src.models.spanbert_model import SpanResult


class TestRAGPipeline:
    """Test cases for RAGPipeline class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data files
        self.posts_path = Path(self.temp_dir) / "test_posts.csv"
        self.criteria_path = Path(self.temp_dir) / "test_criteria.json"
        
        # Create test posts CSV
        import pandas as pd
        test_posts_data = {
            'post_context': ['Test context'],
            'source_file': ['test.csv'],
            'translated_post': ['I feel very depressed and hopeless about my future.'],
            'eval_len_ratio': [0.8],
            'eval_flag_len_short': [False],
            'eval_flag_number_mismatch': [False],
            'eval_flag_paren_mismatch': [False],
            'eval_tgt_alpha_pct': [0.95],
            'eval_flag_alpha_high': [False]
        }
        pd.DataFrame(test_posts_data).to_csv(self.posts_path, index=False)
        
        # Create test criteria JSON
        test_criteria_data = [
            {
                "diagnosis": "Major Depressive Disorder",
                "criteria": [
                    {
                        "id": "A.1",
                        "text": "Depressed mood most of the day, nearly every day"
                    }
                ]
            }
        ]
        with open(self.criteria_path, 'w') as f:
            json.dump(test_criteria_data, f)
    
    def teardown_method(self):
        """Clean up test data"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('src.models.rag_pipeline.BGEEmbeddingModel')
    @patch('src.models.rag_pipeline.SpanBERTModel')
    def test_initialization(self, mock_spanbert, mock_bge):
        """Test pipeline initialization"""
        mock_bge.return_value = Mock()
        mock_spanbert.return_value = Mock()
        
        pipeline = RAGPipeline(
            posts_path=self.posts_path,
            criteria_path=self.criteria_path,
            device="cpu"
        )
        
        assert pipeline.posts_path == self.posts_path
        assert pipeline.criteria_path == self.criteria_path
        assert pipeline.device == "cpu"
        assert pipeline.embedding_model is not None
        assert pipeline.spanbert_model is not None
    
    @patch('src.models.rag_pipeline.BGEEmbeddingModel')
    @patch('src.models.rag_pipeline.SpanBERTModel')
    @patch('src.models.rag_pipeline.FAISSIndex')
    def test_build_index(self, mock_faiss, mock_spanbert, mock_bge):
        """Test building FAISS index"""
        # Setup mocks
        mock_bge_instance = Mock()
        mock_bge_instance.get_embedding_dimension.return_value = 1024
        mock_bge_instance.encode_texts.return_value = np.random.rand(3, 1024)
        mock_bge.return_value = mock_bge_instance
        
        mock_spanbert.return_value = Mock()
        
        mock_faiss_instance = Mock()
        mock_faiss.return_value = mock_faiss_instance
        
        # Create pipeline
        pipeline = RAGPipeline(
            posts_path=self.posts_path,
            criteria_path=self.criteria_path,
            device="cpu"
        )
        
        # Build index
        pipeline.build_index()
        
        # Verify that FAISS index was created and populated
        mock_faiss.assert_called_once()
        mock_faiss_instance.add_embeddings.assert_called_once()
        mock_bge_instance.encode_texts.assert_called_once()
    
    @patch('src.models.rag_pipeline.BGEEmbeddingModel')
    @patch('src.models.rag_pipeline.SpanBERTModel')
    @patch('src.models.rag_pipeline.FAISSIndex')
    def test_process_post(self, mock_faiss, mock_spanbert, mock_bge):
        """Test processing a single post"""
        # Setup mocks
        mock_bge_instance = Mock()
        mock_bge_instance.encode_single.return_value = np.random.rand(1024)
        mock_bge.return_value = mock_bge_instance
        
        mock_spanbert_instance = Mock()
        mock_spanbert_instance.filter_criteria_matches.return_value = [
            ("Depressed mood most of the day", 0.8, [SpanResult("depressed", 0, 9, 0.9, "relevant")])
        ]
        mock_spanbert.return_value = mock_spanbert_instance
        
        mock_faiss_instance = Mock()
        mock_faiss_instance.search.return_value = [
            ("Depressed mood most of the day", 0.9, 0)
        ]
        mock_faiss.return_value = mock_faiss_instance
        
        # Create pipeline
        pipeline = RAGPipeline(
            posts_path=self.posts_path,
            criteria_path=self.criteria_path,
            device="cpu"
        )
        
        # Mock criteria data
        pipeline.criteria_data = [
            {
                'id': 'Major_Depressive_Disorder_A.1',
                'diagnosis': 'Major Depressive Disorder',
                'text': 'Depressed mood most of the day, nearly every day'
            }
        ]
        pipeline.faiss_index = mock_faiss_instance
        
        # Process post
        result = pipeline.process_post("I feel very depressed and hopeless", post_id=1)
        
        # Verify result
        assert isinstance(result, RAGResult)
        assert result.post_id == 1
        assert result.post_text == "I feel very depressed and hopeless"
        # Note: matched_criteria might be empty due to mock setup
        assert result.processing_time > 0
    
    @patch('src.models.rag_pipeline.BGEEmbeddingModel')
    @patch('src.models.rag_pipeline.SpanBERTModel')
    @patch('src.models.rag_pipeline.FAISSIndex')
    def test_process_posts_batch(self, mock_faiss, mock_spanbert, mock_bge):
        """Test processing multiple posts in batch"""
        # Setup mocks
        mock_bge_instance = Mock()
        mock_bge_instance.encode_single.return_value = np.random.rand(1024)
        mock_bge.return_value = mock_bge_instance
        
        mock_spanbert_instance = Mock()
        mock_spanbert_instance.filter_criteria_matches.return_value = []
        mock_spanbert.return_value = mock_spanbert_instance
        
        mock_faiss_instance = Mock()
        mock_faiss_instance.search.return_value = []
        mock_faiss.return_value = mock_faiss_instance
        
        # Create pipeline
        pipeline = RAGPipeline(
            posts_path=self.posts_path,
            criteria_path=self.criteria_path,
            device="cpu"
        )
        
        pipeline.criteria_data = []
        pipeline.faiss_index = mock_faiss_instance
        
        # Process posts
        posts = [
            {'text': 'I feel depressed', 'id': 1},
            {'text': 'I have anxiety', 'id': 2}
        ]
        results = pipeline.process_posts_batch(posts, batch_size=2)
        
        # Verify results
        assert len(results) == 2
        assert all(isinstance(r, RAGResult) for r in results)
        assert results[0].post_id == 1
        assert results[1].post_id == 2
    
    @patch('src.models.rag_pipeline.BGEEmbeddingModel')
    @patch('src.models.rag_pipeline.SpanBERTModel')
    @patch('src.models.rag_pipeline.FAISSIndex')
    def test_evaluate_posts(self, mock_faiss, mock_spanbert, mock_bge):
        """Test evaluating posts from dataset"""
        # Setup mocks
        mock_bge_instance = Mock()
        mock_bge_instance.get_embedding_dimension.return_value = 1024
        mock_bge_instance.encode_texts.return_value = np.random.rand(1, 1024)
        mock_bge_instance.encode_single.return_value = np.random.rand(1024)
        mock_bge.return_value = mock_bge_instance
        
        mock_spanbert_instance = Mock()
        mock_spanbert_instance.filter_criteria_matches.return_value = []
        mock_spanbert.return_value = mock_spanbert_instance
        
        mock_faiss_instance = Mock()
        mock_faiss_instance.search.return_value = []
        mock_faiss.return_value = mock_faiss_instance
        
        # Create pipeline
        pipeline = RAGPipeline(
            posts_path=self.posts_path,
            criteria_path=self.criteria_path,
            device="cpu"
        )
        
        pipeline.criteria_data = []
        pipeline.faiss_index = mock_faiss_instance
        
        # Evaluate posts
        results = pipeline.evaluate_posts(num_posts=1)
        
        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], RAGResult)
    
    def test_get_statistics(self):
        """Test getting statistics from results"""
        # Create mock results
        results = [
            RAGResult(
                post_id=1,
                post_text="Test post 1",
                matched_criteria=[
                    CriteriaMatch(
                        criteria_id="A.1",
                        diagnosis="Major Depressive Disorder",
                        criterion_text="Depressed mood",
                        similarity_score=0.8,
                        spanbert_score=0.7,
                        supporting_spans=[],
                        is_match=True
                    )
                ],
                total_matches=1,
                processing_time=1.0
            ),
            RAGResult(
                post_id=2,
                post_text="Test post 2",
                matched_criteria=[],
                total_matches=0,
                processing_time=0.5
            )
        ]
        
        # Create pipeline
        pipeline = RAGPipeline(
            posts_path=self.posts_path,
            criteria_path=self.criteria_path,
            device="cpu"
        )
        
        # Get statistics
        stats = pipeline.get_statistics(results)
        
        # Verify statistics
        assert stats["total_posts"] == 2
        assert stats["total_matches"] == 1
        assert stats["avg_matches_per_post"] == 0.5
        assert stats["avg_processing_time"] == 0.75
        assert stats["posts_with_matches"] == 1
        assert "Major Depressive Disorder" in stats["diagnosis_counts"]
        assert stats["diagnosis_counts"]["Major Depressive Disorder"] == 1
    
    def test_save_results(self):
        """Test saving results to file"""
        # Create mock results
        results = [
            RAGResult(
                post_id=1,
                post_text="Test post",
                matched_criteria=[
                    CriteriaMatch(
                        criteria_id="A.1",
                        diagnosis="Major Depressive Disorder",
                        criterion_text="Depressed mood",
                        similarity_score=0.8,
                        spanbert_score=0.7,
                        supporting_spans=[
                            SpanResult("depressed", 0, 9, 0.9, "relevant")
                        ],
                        is_match=True
                    )
                ],
                total_matches=1,
                processing_time=1.0
            )
        ]
        
        # Create pipeline
        pipeline = RAGPipeline(
            posts_path=self.posts_path,
            criteria_path=self.criteria_path,
            device="cpu"
        )
        
        # Save results
        save_path = Path(self.temp_dir) / "test_results.json"
        pipeline.save_results(results, save_path)
        
        # Verify file was created and contains data
        assert save_path.exists()
        with open(save_path, 'r') as f:
            saved_data = json.load(f)
        
        assert len(saved_data) == 1
        assert saved_data[0]["post_id"] == 1
        assert saved_data[0]["total_matches"] == 1
        assert len(saved_data[0]["matched_criteria"]) == 1
    
    def test_criteria_match_creation(self):
        """Test CriteriaMatch dataclass"""
        match = CriteriaMatch(
            criteria_id="A.1",
            diagnosis="Major Depressive Disorder",
            criterion_text="Depressed mood",
            similarity_score=0.8,
            spanbert_score=0.7,
            supporting_spans=[],
            is_match=True
        )
        
        assert match.criteria_id == "A.1"
        assert match.diagnosis == "Major Depressive Disorder"
        assert match.similarity_score == 0.8
        assert match.is_match is True
    
    def test_rag_result_creation(self):
        """Test RAGResult dataclass"""
        result = RAGResult(
            post_id=1,
            post_text="Test post",
            matched_criteria=[],
            total_matches=0,
            processing_time=1.0
        )
        
        assert result.post_id == 1
        assert result.post_text == "Test post"
        assert result.total_matches == 0
        assert result.processing_time == 1.0
