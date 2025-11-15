"""Unit tests for criteria matching head."""

import pytest
import torch
from src.dataaug_multi_both.models.heads.criteria_matching import (
    CriteriaMatchingHead,
    CriteriaMatchingConfig,
    create_criteria_matching_head
)


class TestCriteriaMatchingConfig:
    """Test suite for CriteriaMatchingConfig."""
    
    def test_config_defaults(self):
        """Test that config has correct defaults."""
        config = CriteriaMatchingConfig()
        assert config.hidden_size == 768
        assert config.num_labels == 9
        assert config.dropout == 0.1
        assert config.pooling_strategy == "cls"
        assert config.threshold == 0.5
    
    def test_config_custom_values(self):
        """Test that config accepts custom values."""
        config = CriteriaMatchingConfig(
            hidden_size=1024,
            num_labels=12,
            dropout=0.2,
            pooling_strategy="mean",
            threshold=0.6
        )
        assert config.hidden_size == 1024
        assert config.num_labels == 12
        assert config.dropout == 0.2
        assert config.pooling_strategy == "mean"
        assert config.threshold == 0.6
    
    def test_config_invalid_dropout(self):
        """Test that invalid dropout raises error."""
        with pytest.raises(ValueError, match="Dropout must be in"):
            CriteriaMatchingConfig(dropout=1.5)
    
    def test_config_invalid_pooling(self):
        """Test that invalid pooling strategy raises error."""
        with pytest.raises(ValueError, match="Invalid pooling strategy"):
            CriteriaMatchingConfig(pooling_strategy="invalid")
    
    def test_config_invalid_threshold(self):
        """Test that invalid threshold raises error."""
        with pytest.raises(ValueError, match="Threshold must be in"):
            CriteriaMatchingConfig(threshold=1.5)
    
    def test_config_to_dict(self):
        """Test config serialization to dict."""
        config = CriteriaMatchingConfig(hidden_size=768, num_labels=9)
        config_dict = config.to_dict()
        
        assert config_dict["hidden_size"] == 768
        assert config_dict["num_labels"] == 9
        assert "dropout" in config_dict
        assert "pooling_strategy" in config_dict


class TestCriteriaMatchingHead:
    """Test suite for CriteriaMatchingHead."""
    
    def test_head_initialization(self):
        """Test that head can be initialized."""
        head = CriteriaMatchingHead(hidden_size=768, num_labels=9)
        assert head.hidden_size == 768
        assert head.num_labels == 9
        assert head.pooling_strategy == "cls"
    
    def test_forward_cls_pooling(self):
        """Test forward pass with CLS pooling."""
        batch_size, seq_len, hidden_size = 2, 10, 768
        num_labels = 9
        
        head = CriteriaMatchingHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            pooling_strategy="cls"
        )
        
        # Create dummy encoder outputs
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        encoder_outputs = {"last_hidden_state": hidden_states}
        
        # Forward pass
        logits = head(encoder_outputs)
        
        assert logits.shape == (batch_size, num_labels)
    
    def test_forward_mean_pooling(self):
        """Test forward pass with mean pooling."""
        batch_size, seq_len, hidden_size = 2, 10, 768
        num_labels = 9
        
        head = CriteriaMatchingHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            pooling_strategy="mean"
        )
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        encoder_outputs = {"last_hidden_state": hidden_states}
        attention_mask = torch.ones(batch_size, seq_len)
        
        logits = head(encoder_outputs, attention_mask)
        
        assert logits.shape == (batch_size, num_labels)
    
    def test_forward_max_pooling(self):
        """Test forward pass with max pooling."""
        batch_size, seq_len, hidden_size = 2, 10, 768
        num_labels = 9
        
        head = CriteriaMatchingHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            pooling_strategy="max"
        )
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        encoder_outputs = {"last_hidden_state": hidden_states}
        
        logits = head(encoder_outputs)
        
        assert logits.shape == (batch_size, num_labels)
    
    def test_pool_encoder_outputs_cls(self):
        """Test CLS pooling extracts first token."""
        batch_size, seq_len, hidden_size = 2, 10, 768
        
        head = CriteriaMatchingHead(
            hidden_size=hidden_size,
            pooling_strategy="cls"
        )
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        pooled = head.pool_encoder_outputs(hidden_states)
        
        assert pooled.shape == (batch_size, hidden_size)
        # Verify it's the first token
        assert torch.allclose(pooled, hidden_states[:, 0, :])
    
    def test_pool_encoder_outputs_mean_with_mask(self):
        """Test mean pooling respects attention mask."""
        batch_size, seq_len, hidden_size = 2, 10, 768
        
        head = CriteriaMatchingHead(
            hidden_size=hidden_size,
            pooling_strategy="mean"
        )
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        # Mask out last 5 tokens
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, 5:] = 0
        
        pooled = head.pool_encoder_outputs(hidden_states, attention_mask)
        
        assert pooled.shape == (batch_size, hidden_size)
        # Pooled output should not be all zeros
        assert not torch.allclose(pooled, torch.zeros_like(pooled))
    
    def test_get_predictions(self):
        """Test converting logits to binary predictions."""
        batch_size, num_labels = 2, 9
        
        head = CriteriaMatchingHead(hidden_size=768, num_labels=num_labels)
        
        # Create logits that will produce known predictions
        logits = torch.tensor([
            [2.0, -2.0, 0.5, -0.5, 1.0, -1.0, 0.0, 3.0, -3.0],
            [-2.0, 2.0, -0.5, 0.5, -1.0, 1.0, 0.0, -3.0, 3.0]
        ])
        
        predictions = head.get_predictions(logits, threshold=0.5)
        
        assert predictions.shape == (batch_size, num_labels)
        assert predictions.dtype == torch.long
        # Check some predictions (sigmoid(2.0) > 0.5, sigmoid(-2.0) < 0.5)
        assert predictions[0, 0] == 1
        assert predictions[0, 1] == 0
    
    def test_get_probabilities(self):
        """Test converting logits to probabilities."""
        batch_size, num_labels = 2, 9
        
        head = CriteriaMatchingHead(hidden_size=768, num_labels=num_labels)
        
        logits = torch.randn(batch_size, num_labels)
        probabilities = head.get_probabilities(logits)
        
        assert probabilities.shape == (batch_size, num_labels)
        # All probabilities should be in [0, 1]
        assert torch.all(probabilities >= 0.0)
        assert torch.all(probabilities <= 1.0)
    
    def test_invalid_pooling_strategy_raises_error(self):
        """Test that invalid pooling strategy raises error."""
        head = CriteriaMatchingHead(
            hidden_size=768,
            pooling_strategy="invalid"
        )
        
        hidden_states = torch.randn(2, 10, 768)
        
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            head.pool_encoder_outputs(hidden_states)
    
    def test_dropout_applied(self):
        """Test that dropout is applied during training."""
        head = CriteriaMatchingHead(
            hidden_size=768,
            num_labels=9,
            dropout=0.5
        )
        head.train()  # Set to training mode
        
        hidden_states = torch.randn(2, 10, 768)
        encoder_outputs = {"last_hidden_state": hidden_states}
        
        # Run multiple times and check for variation (due to dropout)
        logits1 = head(encoder_outputs)
        logits2 = head(encoder_outputs)
        
        # With dropout, outputs should differ
        # (This test may occasionally fail due to randomness, but very unlikely)
        assert not torch.allclose(logits1, logits2, atol=1e-6)


class TestCreateCriteriaMatchingHead:
    """Test suite for factory function."""
    
    def test_create_head_defaults(self):
        """Test creating head with defaults."""
        head = create_criteria_matching_head(hidden_size=768)
        assert head.hidden_size == 768
        assert head.num_labels == 9
    
    def test_create_head_custom_params(self):
        """Test creating head with custom parameters."""
        head = create_criteria_matching_head(
            hidden_size=1024,
            num_labels=12,
            dropout=0.2,
            pooling_strategy="mean"
        )
        assert head.hidden_size == 1024
        assert head.num_labels == 12
        assert head.pooling_strategy == "mean"

