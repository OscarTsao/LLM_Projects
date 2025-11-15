"""
Tests for BGE embedding model functionality
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
import torch

from src.models.embedding_model import BGEEmbeddingModel


class TestBGEEmbeddingModel:
    """Test cases for BGEEmbeddingModel class"""
    
    def setup_method(self):
        """Setup test environment"""
        # Mock the SentenceTransformer to avoid loading actual model
        self.mock_model = Mock()
        self.mock_model.encode.return_value = torch.randn(2, 1024)
        self.mock_model.half.return_value = self.mock_model
        
    @patch('src.models.embedding_model.SentenceTransformer')
    def test_initialization(self, mock_sentence_transformer):
        """Test model initialization"""
        mock_sentence_transformer.return_value = self.mock_model
        
        model = BGEEmbeddingModel(device="cpu")
        
        assert model.model_name == "BAAI/bge-m3"
        assert model.device == "cpu"
        assert model.model is not None
    
    @patch('src.models.embedding_model.SentenceTransformer')
    def test_get_embedding_dimension(self, mock_sentence_transformer):
        """Test getting embedding dimension"""
        mock_sentence_transformer.return_value = self.mock_model
        
        model = BGEEmbeddingModel(device="cpu")
        dimension = model.get_embedding_dimension()
        
        assert dimension == 1024
    
    @patch('src.models.embedding_model.SentenceTransformer')
    def test_encode_single(self, mock_sentence_transformer):
        """Test encoding single text"""
        # Mock to return single embedding (1D array)
        self.mock_model.encode.return_value = torch.randn(1024)
        mock_sentence_transformer.return_value = self.mock_model
        
        model = BGEEmbeddingModel(device="cpu")
        embedding = model.encode_single("Test text")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)
    
    @patch('src.models.embedding_model.SentenceTransformer')
    def test_encode_texts(self, mock_sentence_transformer):
        """Test encoding multiple texts"""
        mock_sentence_transformer.return_value = self.mock_model
        
        model = BGEEmbeddingModel(device="cpu")
        texts = ["Test text 1", "Test text 2"]
        embeddings = model.encode_texts(texts, batch_size=2)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 1024)
    
    @patch('src.models.embedding_model.SentenceTransformer')
    def test_encode_texts_batch_processing(self, mock_sentence_transformer):
        """Test batch processing for large number of texts"""
        # Mock to return correct number of embeddings
        def mock_encode(texts, **kwargs):
            return torch.randn(len(texts), 1024)
        
        self.mock_model.encode.side_effect = mock_encode
        mock_sentence_transformer.return_value = self.mock_model
        
        model = BGEEmbeddingModel(device="cpu")
        texts = ["Test text"] * 10  # 10 texts
        embeddings = model.encode_texts(texts, batch_size=3)  # Process in batches of 3
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (10, 1024)
    
    @patch('src.models.embedding_model.SentenceTransformer')
    def test_save_load_embeddings(self, mock_sentence_transformer, tmp_path):
        """Test saving and loading embeddings"""
        mock_sentence_transformer.return_value = self.mock_model
        
        model = BGEEmbeddingModel(device="cpu")
        embeddings = np.random.rand(5, 1024)
        
        # Test saving
        save_path = tmp_path / "test_embeddings.npy"
        model.save_embeddings(embeddings, save_path)
        assert save_path.exists()
        
        # Test loading
        loaded_embeddings = model.load_embeddings(save_path)
        np.testing.assert_array_equal(embeddings, loaded_embeddings)
    
    @patch('src.models.embedding_model.SentenceTransformer')
    def test_normalize_embeddings(self, mock_sentence_transformer):
        """Test embedding normalization"""
        # Mock to return correct number of embeddings
        def mock_encode(texts, **kwargs):
            return torch.randn(len(texts), 1024)
        
        self.mock_model.encode.side_effect = mock_encode
        mock_sentence_transformer.return_value = self.mock_model
        
        model = BGEEmbeddingModel(device="cpu")
        texts = ["Test text"]
        
        # Test with normalization
        embeddings_normalized = model.encode_texts(texts, normalize_embeddings=True)
        
        # Test without normalization
        embeddings_not_normalized = model.encode_texts(texts, normalize_embeddings=False)
        
        assert isinstance(embeddings_normalized, np.ndarray)
        assert isinstance(embeddings_not_normalized, np.ndarray)
        assert embeddings_normalized.shape == (1, 1024)
        assert embeddings_not_normalized.shape == (1, 1024)
    
    @patch('src.models.embedding_model.SentenceTransformer')
    def test_error_handling(self, mock_sentence_transformer):
        """Test error handling"""
        # Test initialization error
        mock_sentence_transformer.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception):
            BGEEmbeddingModel(device="cpu")
    
    @patch('src.models.embedding_model.SentenceTransformer')
    def test_clear_cache(self, mock_sentence_transformer):
        """Test cache clearing"""
        mock_sentence_transformer.return_value = self.mock_model
        
        model = BGEEmbeddingModel(device="cpu")
        # Should not raise any errors
        model.clear_cache()
    
    @patch('src.models.embedding_model.SentenceTransformer')
    def test_gpu_optimizations(self, mock_sentence_transformer):
        """Test GPU optimizations"""
        mock_sentence_transformer.return_value = self.mock_model
        
        with patch('torch.cuda.is_available', return_value=True):
            model = BGEEmbeddingModel(device="cuda")
            
            # Check that half precision is called
            self.mock_model.half.assert_called_once()
    
    @patch('src.models.embedding_model.SentenceTransformer')
    def test_progress_bar(self, mock_sentence_transformer):
        """Test progress bar functionality"""
        # Mock to return correct number of embeddings
        def mock_encode(texts, **kwargs):
            return torch.randn(len(texts), 1024)
        
        self.mock_model.encode.side_effect = mock_encode
        mock_sentence_transformer.return_value = self.mock_model
        
        model = BGEEmbeddingModel(device="cpu")
        texts = ["Test text"] * 5
        
        # Test with progress bar
        embeddings = model.encode_texts(texts, show_progress=True)
        
        # Test without progress bar
        embeddings = model.encode_texts(texts, show_progress=False)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (5, 1024)
