"""Unit tests for HF encoder wrapper."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.dataaug_multi_both.models.encoders.hf_encoder import (
    HFEncoder,
    HFEncoderConfig,
    ModelLoadingError,
    ModelAuthenticationError
)


class TestHFEncoderConfig:
    """Test suite for HFEncoderConfig."""
    
    def test_config_defaults(self):
        """Test that config has correct defaults."""
        config = HFEncoderConfig(model_id="bert-base-uncased")
        assert config.model_id == "bert-base-uncased"
        assert config.revision is None
        assert config.cache_dir is None
        assert config.trust_remote_code is False
        assert config.use_auth_token is None
        assert config.gradient_checkpointing is False
    
    def test_config_custom_values(self):
        """Test that config accepts custom values."""
        config = HFEncoderConfig(
            model_id="mental/mental-bert-base-uncased",
            revision="v1.0",
            cache_dir="/tmp/cache",
            trust_remote_code=True,
            use_auth_token="hf_token",
            gradient_checkpointing=True
        )
        assert config.model_id == "mental/mental-bert-base-uncased"
        assert config.revision == "v1.0"
        assert config.cache_dir == "/tmp/cache"
        assert config.trust_remote_code is True
        assert config.use_auth_token == "hf_token"
        assert config.gradient_checkpointing is True


class TestHFEncoder:
    """Test suite for HFEncoder."""
    
    def test_encoder_initialization(self):
        """Test that encoder can be initialized."""
        config = HFEncoderConfig(model_id="bert-base-uncased")
        encoder = HFEncoder(config)
        assert encoder.config == config
        assert encoder.model is None
        assert encoder.tokenizer is None
    
    @patch('src.dataaug_multi_both.models.encoders.hf_encoder.AutoModel')
    @patch('src.dataaug_multi_both.models.encoders.hf_encoder.AutoTokenizer')
    def test_load_success(self, mock_tokenizer_class, mock_model_class):
        """Test successful model loading."""
        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        config = HFEncoderConfig(model_id="bert-base-uncased")
        encoder = HFEncoder(config)
        model, tokenizer = encoder.load()
        
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        assert encoder.model == mock_model
        assert encoder.tokenizer == mock_tokenizer
    
    @patch('src.dataaug_multi_both.models.encoders.hf_encoder.AutoModel')
    @patch('src.dataaug_multi_both.models.encoders.hf_encoder.AutoTokenizer')
    def test_load_with_revision(self, mock_tokenizer_class, mock_model_class):
        """Test loading with specific revision."""
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        config = HFEncoderConfig(
            model_id="bert-base-uncased",
            revision="v1.0"
        )
        encoder = HFEncoder(config)
        encoder.load()
        
        # Verify revision was passed
        mock_model_class.from_pretrained.assert_called_once()
        call_kwargs = mock_model_class.from_pretrained.call_args[1]
        assert call_kwargs['revision'] == "v1.0"
    
    @patch('src.dataaug_multi_both.models.encoders.hf_encoder.AutoModel')
    @patch('src.dataaug_multi_both.models.encoders.hf_encoder.AutoTokenizer')
    def test_load_authentication_error(self, mock_tokenizer_class, mock_model_class):
        """Test that authentication errors are handled properly."""
        # Simulate 401 error
        error = Exception("401 Unauthorized")
        mock_tokenizer_class.from_pretrained.side_effect = error
        
        config = HFEncoderConfig(model_id="private/model")
        encoder = HFEncoder(config)
        
        with pytest.raises(ModelAuthenticationError, match="Authentication failed"):
            encoder.load()
    
    @patch('src.dataaug_multi_both.models.encoders.hf_encoder.AutoModel')
    @patch('src.dataaug_multi_both.models.encoders.hf_encoder.AutoTokenizer')
    def test_load_network_error(self, mock_tokenizer_class, mock_model_class):
        """Test that network errors produce actionable error."""
        error = Exception("Network unreachable")
        mock_tokenizer_class.from_pretrained.side_effect = error
        
        config = HFEncoderConfig(model_id="bert-base-uncased")
        encoder = HFEncoder(config)
        
        with pytest.raises(ModelLoadingError, match="Failed to load model"):
            encoder.load()
    
    @patch('src.dataaug_multi_both.models.encoders.hf_encoder.AutoModel')
    @patch('src.dataaug_multi_both.models.encoders.hf_encoder.AutoTokenizer')
    def test_enable_gradient_checkpointing(self, mock_tokenizer_class, mock_model_class):
        """Test gradient checkpointing enablement."""
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_model.gradient_checkpointing_enable = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        config = HFEncoderConfig(model_id="bert-base-uncased")
        encoder = HFEncoder(config)
        encoder.load()
        encoder.enable_gradient_checkpointing()
        
        mock_model.gradient_checkpointing_enable.assert_called_once()
    
    def test_enable_gradient_checkpointing_before_load_fails(self):
        """Test that gradient checkpointing fails if model not loaded."""
        config = HFEncoderConfig(model_id="bert-base-uncased")
        encoder = HFEncoder(config)
        
        with pytest.raises(ModelLoadingError, match="Model must be loaded"):
            encoder.enable_gradient_checkpointing()
    
    @patch('src.dataaug_multi_both.models.encoders.hf_encoder.AutoModel')
    @patch('src.dataaug_multi_both.models.encoders.hf_encoder.AutoTokenizer')
    def test_configure_for_training(self, mock_tokenizer_class, mock_model_class):
        """Test configure_for_training with gradient checkpointing."""
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_model.gradient_checkpointing_enable = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        config = HFEncoderConfig(
            model_id="bert-base-uncased",
            gradient_checkpointing=True
        )
        encoder = HFEncoder(config)
        model, tokenizer = encoder.configure_for_training()
        
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        mock_model.gradient_checkpointing_enable.assert_called_once()
    
    def test_is_auth_error_401(self):
        """Test that 401 errors are recognized as auth errors."""
        config = HFEncoderConfig(model_id="bert-base-uncased")
        encoder = HFEncoder(config)
        
        error = Exception("401 Unauthorized")
        assert encoder._is_auth_error(error) is True
    
    def test_is_auth_error_403(self):
        """Test that 403 errors are recognized as auth errors."""
        config = HFEncoderConfig(model_id="bert-base-uncased")
        encoder = HFEncoder(config)
        
        error = Exception("403 Forbidden")
        assert encoder._is_auth_error(error) is True
    
    def test_is_auth_error_token_keyword(self):
        """Test that token-related errors are recognized as auth errors."""
        config = HFEncoderConfig(model_id="bert-base-uncased")
        encoder = HFEncoder(config)
        
        error = Exception("Invalid token provided")
        assert encoder._is_auth_error(error) is True
    
    def test_is_auth_error_network_error(self):
        """Test that network errors are not auth errors."""
        config = HFEncoderConfig(model_id="bert-base-uncased")
        encoder = HFEncoder(config)
        
        error = Exception("Network unreachable")
        assert encoder._is_auth_error(error) is False
    
    @patch('src.dataaug_multi_both.models.encoders.hf_encoder.AutoModel')
    @patch('src.dataaug_multi_both.models.encoders.hf_encoder.AutoTokenizer')
    def test_load_with_cache_dir(self, mock_tokenizer_class, mock_model_class):
        """Test loading with custom cache directory."""
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        config = HFEncoderConfig(
            model_id="bert-base-uncased",
            cache_dir="/custom/cache"
        )
        encoder = HFEncoder(config)
        encoder.load()
        
        # Verify cache_dir was passed
        call_kwargs = mock_model_class.from_pretrained.call_args[1]
        assert call_kwargs['cache_dir'] == "/custom/cache"

