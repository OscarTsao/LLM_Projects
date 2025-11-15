"""Unit tests for evidence binding head."""

import pytest
import torch
from src.dataaug_multi_both.models.heads.evidence_binding import (
    EvidenceBindingHead,
    EvidenceBindingConfig,
    create_evidence_binding_head
)


class TestEvidenceBindingConfig:
    """Test suite for EvidenceBindingConfig."""
    
    def test_config_defaults(self):
        """Test that config has correct defaults."""
        config = EvidenceBindingConfig()
        assert config.hidden_size == 768
        assert config.max_span_length == 512
        assert config.dropout == 0.1
        assert config.top_k_spans == 1
    
    def test_config_custom_values(self):
        """Test that config accepts custom values."""
        config = EvidenceBindingConfig(
            hidden_size=1024,
            max_span_length=256,
            dropout=0.2,
            top_k_spans=3
        )
        assert config.hidden_size == 1024
        assert config.max_span_length == 256
        assert config.dropout == 0.2
        assert config.top_k_spans == 3
    
    def test_config_invalid_dropout(self):
        """Test that invalid dropout raises error."""
        with pytest.raises(ValueError, match="Dropout must be in"):
            EvidenceBindingConfig(dropout=1.5)
    
    def test_config_invalid_max_span_length(self):
        """Test that invalid max_span_length raises error."""
        with pytest.raises(ValueError, match="max_span_length must be positive"):
            EvidenceBindingConfig(max_span_length=0)
    
    def test_config_invalid_top_k(self):
        """Test that invalid top_k_spans raises error."""
        with pytest.raises(ValueError, match="top_k_spans must be positive"):
            EvidenceBindingConfig(top_k_spans=0)
    
    def test_config_to_dict(self):
        """Test config serialization to dict."""
        config = EvidenceBindingConfig(hidden_size=768)
        config_dict = config.to_dict()
        
        assert config_dict["hidden_size"] == 768
        assert config_dict["max_span_length"] == 512
        assert "dropout" in config_dict


class TestEvidenceBindingHead:
    """Test suite for EvidenceBindingHead."""
    
    def test_head_initialization(self):
        """Test that head can be initialized."""
        head = EvidenceBindingHead(hidden_size=768, max_span_length=512)
        assert head.hidden_size == 768
        assert head.max_span_length == 512
    
    def test_forward_output_shape(self):
        """Test forward pass output shapes."""
        batch_size, seq_len, hidden_size = 2, 10, 768
        
        head = EvidenceBindingHead(hidden_size=hidden_size)
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        encoder_outputs = {"last_hidden_state": hidden_states}
        
        start_logits, end_logits = head(encoder_outputs)
        
        assert start_logits.shape == (batch_size, seq_len)
        assert end_logits.shape == (batch_size, seq_len)
    
    def test_forward_with_attention_mask(self):
        """Test forward pass with attention mask."""
        batch_size, seq_len, hidden_size = 2, 10, 768
        
        head = EvidenceBindingHead(hidden_size=hidden_size)
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        encoder_outputs = {"last_hidden_state": hidden_states}
        
        # Mask out last 3 tokens
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, 7:] = 0
        
        start_logits, end_logits = head(encoder_outputs, attention_mask)
        
        # Masked positions should have very negative logits
        assert torch.all(start_logits[:, 7:] < -1e8)
        assert torch.all(end_logits[:, 7:] < -1e8)
    
    def test_get_span_predictions(self):
        """Test extracting most likely span."""
        batch_size, seq_len = 2, 10
        
        head = EvidenceBindingHead(hidden_size=768)
        
        # Create logits with known argmax
        start_logits = torch.randn(batch_size, seq_len)
        start_logits[0, 2] = 10.0  # Highest for batch 0
        start_logits[1, 5] = 10.0  # Highest for batch 1
        
        end_logits = torch.randn(batch_size, seq_len)
        end_logits[0, 7] = 10.0  # Highest for batch 0
        end_logits[1, 8] = 10.0  # Highest for batch 1
        
        start_pos, end_pos = head.get_span_predictions(start_logits, end_logits)
        
        assert start_pos[0] == 2
        assert start_pos[1] == 5
        assert end_pos[0] == 7
        assert end_pos[1] == 8
    
    def test_get_span_predictions_ensures_end_after_start(self):
        """Test that end position is always >= start position."""
        batch_size, seq_len = 2, 10
        
        head = EvidenceBindingHead(hidden_size=768)
        
        # Create logits where end < start
        start_logits = torch.randn(batch_size, seq_len)
        start_logits[0, 7] = 10.0  # Start at position 7
        
        end_logits = torch.randn(batch_size, seq_len)
        end_logits[0, 3] = 10.0  # End at position 3 (before start)
        
        start_pos, end_pos = head.get_span_predictions(start_logits, end_logits)
        
        # End should be adjusted to be >= start
        assert end_pos[0] >= start_pos[0]
    
    def test_extract_spans_top_1(self):
        """Test extracting top-1 span."""
        batch_size, seq_len = 2, 10
        
        head = EvidenceBindingHead(hidden_size=768, max_span_length=5)
        
        start_logits = torch.randn(batch_size, seq_len)
        end_logits = torch.randn(batch_size, seq_len)
        
        start_pos, end_pos, scores = head.extract_spans(
            start_logits, end_logits, top_k=1
        )
        
        assert start_pos.shape == (batch_size, 1)
        assert end_pos.shape == (batch_size, 1)
        assert scores.shape == (batch_size, 1)
        
        # Verify span length constraint
        for i in range(batch_size):
            span_length = end_pos[i, 0] - start_pos[i, 0] + 1
            assert span_length <= 5
    
    def test_extract_spans_top_k(self):
        """Test extracting top-k spans."""
        batch_size, seq_len = 2, 10
        top_k = 3
        
        head = EvidenceBindingHead(hidden_size=768)
        
        start_logits = torch.randn(batch_size, seq_len)
        end_logits = torch.randn(batch_size, seq_len)
        
        start_pos, end_pos, scores = head.extract_spans(
            start_logits, end_logits, top_k=top_k
        )
        
        assert start_pos.shape == (batch_size, top_k)
        assert end_pos.shape == (batch_size, top_k)
        assert scores.shape == (batch_size, top_k)
        
        # Scores should be in descending order
        for i in range(batch_size):
            for j in range(top_k - 1):
                assert scores[i, j] >= scores[i, j + 1]
    
    def test_extract_spans_respects_max_length(self):
        """Test that extracted spans respect max_span_length."""
        batch_size, seq_len = 2, 20
        max_span_length = 5
        
        head = EvidenceBindingHead(hidden_size=768, max_span_length=max_span_length)
        
        start_logits = torch.randn(batch_size, seq_len)
        end_logits = torch.randn(batch_size, seq_len)
        
        start_pos, end_pos, scores = head.extract_spans(
            start_logits, end_logits, top_k=5
        )
        
        # All spans should be <= max_span_length
        for i in range(batch_size):
            for j in range(5):
                span_length = end_pos[i, j] - start_pos[i, j] + 1
                assert span_length <= max_span_length
    
    def test_dropout_applied_during_training(self):
        """Test that dropout is applied during training."""
        head = EvidenceBindingHead(hidden_size=768, dropout=0.5)
        head.train()  # Set to training mode
        
        hidden_states = torch.randn(2, 10, 768)
        encoder_outputs = {"last_hidden_state": hidden_states}
        
        # Run multiple times and check for variation
        start1, end1 = head(encoder_outputs)
        start2, end2 = head(encoder_outputs)
        
        # With dropout, outputs should differ
        assert not torch.allclose(start1, start2, atol=1e-6)
        assert not torch.allclose(end1, end2, atol=1e-6)


class TestCreateEvidenceBindingHead:
    """Test suite for factory function."""
    
    def test_create_head_defaults(self):
        """Test creating head with defaults."""
        head = create_evidence_binding_head(hidden_size=768)
        assert head.hidden_size == 768
        assert head.max_span_length == 512
    
    def test_create_head_custom_params(self):
        """Test creating head with custom parameters."""
        head = create_evidence_binding_head(
            hidden_size=1024,
            max_span_length=256,
            dropout=0.2
        )
        assert head.hidden_size == 1024
        assert head.max_span_length == 256

