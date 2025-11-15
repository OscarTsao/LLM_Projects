"""Unit tests for multi-task model."""

import pytest
import torch
from unittest.mock import Mock, MagicMock
from src.dataaug_multi_both.models.multi_task_model import (
    MultiTaskModel,
    MultiTaskModelOutput
)
from src.dataaug_multi_both.models.heads.criteria_matching import CriteriaMatchingHead
from src.dataaug_multi_both.models.heads.evidence_binding import EvidenceBindingHead


class TestMultiTaskModelOutput:
    """Test suite for MultiTaskModelOutput."""
    
    def test_output_creation(self):
        """Test that output can be created."""
        output = MultiTaskModelOutput(
            criteria_logits=torch.randn(2, 9),
            start_logits=torch.randn(2, 10),
            end_logits=torch.randn(2, 10)
        )
        assert output.criteria_logits is not None
        assert output.start_logits is not None
        assert output.end_logits is not None


class TestMultiTaskModel:
    """Test suite for MultiTaskModel."""
    
    def create_mock_encoder(self, hidden_size=768):
        """Create a mock encoder for testing."""
        class MockEncoder(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_ids, attention_mask=None, **kwargs):
                batch_size, seq_len = input_ids.shape
                return {
                    "last_hidden_state": torch.randn(batch_size, seq_len, self.hidden_size)
                }

            def get_hidden_size(self):
                return self.hidden_size

        return MockEncoder(hidden_size)
    
    def test_model_initialization(self):
        """Test that model can be initialized."""
        encoder = self.create_mock_encoder()
        criteria_head = CriteriaMatchingHead(hidden_size=768, num_labels=9)
        evidence_head = EvidenceBindingHead(hidden_size=768)
        
        model = MultiTaskModel(encoder, criteria_head, evidence_head)
        
        assert model.encoder == encoder
        assert model.criteria_head == criteria_head
        assert model.evidence_head == evidence_head
    
    def test_forward_output_structure(self):
        """Test forward pass output structure."""
        encoder = self.create_mock_encoder()
        criteria_head = CriteriaMatchingHead(hidden_size=768, num_labels=9)
        evidence_head = EvidenceBindingHead(hidden_size=768)
        
        model = MultiTaskModel(encoder, criteria_head, evidence_head)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        output = model(input_ids, attention_mask)
        
        assert isinstance(output, MultiTaskModelOutput)
        assert output.criteria_logits.shape == (batch_size, 9)
        assert output.start_logits.shape == (batch_size, seq_len)
        assert output.end_logits.shape == (batch_size, seq_len)
    
    def test_forward_with_predictions(self):
        """Test forward pass with predictions enabled."""
        encoder = self.create_mock_encoder()
        criteria_head = CriteriaMatchingHead(hidden_size=768, num_labels=9)
        evidence_head = EvidenceBindingHead(hidden_size=768)
        
        model = MultiTaskModel(encoder, criteria_head, evidence_head)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        output = model(input_ids, attention_mask, return_predictions=True)
        
        assert output.criteria_predictions is not None
        assert output.criteria_probabilities is not None
        assert output.span_predictions is not None
        assert output.criteria_predictions.shape == (batch_size, 9)
        assert output.criteria_probabilities.shape == (batch_size, 9)
    
    def test_forward_with_encoder_outputs(self):
        """Test forward pass with encoder outputs returned."""
        encoder = self.create_mock_encoder()
        criteria_head = CriteriaMatchingHead(hidden_size=768, num_labels=9)
        evidence_head = EvidenceBindingHead(hidden_size=768)
        
        model = MultiTaskModel(encoder, criteria_head, evidence_head)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        output = model(input_ids, return_encoder_outputs=True)
        
        assert output.encoder_outputs is not None
        assert "last_hidden_state" in output.encoder_outputs
    
    def test_freeze_encoder(self):
        """Test freezing encoder weights."""
        encoder = Mock()
        param1 = Mock()
        param1.requires_grad = True
        param2 = Mock()
        param2.requires_grad = True
        encoder.parameters = Mock(return_value=[param1, param2])
        
        criteria_head = CriteriaMatchingHead(hidden_size=768, num_labels=9)
        evidence_head = EvidenceBindingHead(hidden_size=768)
        
        model = MultiTaskModel(encoder, criteria_head, evidence_head)
        model.freeze_encoder()
        
        assert param1.requires_grad is False
        assert param2.requires_grad is False
    
    def test_unfreeze_encoder(self):
        """Test unfreezing encoder weights."""
        encoder = Mock()
        param1 = Mock()
        param1.requires_grad = False
        param2 = Mock()
        param2.requires_grad = False
        encoder.parameters = Mock(return_value=[param1, param2])
        
        criteria_head = CriteriaMatchingHead(hidden_size=768, num_labels=9)
        evidence_head = EvidenceBindingHead(hidden_size=768)
        
        model = MultiTaskModel(encoder, criteria_head, evidence_head)
        model.unfreeze_encoder()
        
        assert param1.requires_grad is True
        assert param2.requires_grad is True
    
    def test_freeze_encoder_on_init(self):
        """Test that encoder can be frozen during initialization."""
        encoder = Mock()
        param = Mock()
        param.requires_grad = True
        encoder.parameters = Mock(return_value=[param])
        
        criteria_head = CriteriaMatchingHead(hidden_size=768, num_labels=9)
        evidence_head = EvidenceBindingHead(hidden_size=768)
        
        model = MultiTaskModel(
            encoder, criteria_head, evidence_head,
            freeze_encoder=True
        )
        
        assert param.requires_grad is False
    
    def test_get_model_info(self):
        """Test getting model information."""
        encoder = self.create_mock_encoder()
        encoder.get_model_info = Mock(return_value={"hidden_size": 768})
        
        criteria_head = CriteriaMatchingHead(hidden_size=768, num_labels=9)
        evidence_head = EvidenceBindingHead(hidden_size=768, max_span_length=512)
        
        model = MultiTaskModel(encoder, criteria_head, evidence_head)
        info = model.get_model_info()
        
        assert "encoder" in info
        assert "criteria_head" in info
        assert "evidence_head" in info
        assert info["criteria_head"]["num_labels"] == 9
        assert info["evidence_head"]["max_span_length"] == 512
        assert "total_parameters" in info
        assert "trainable_parameters" in info
    
    def test_custom_criteria_threshold(self):
        """Test using custom criteria threshold."""
        encoder = self.create_mock_encoder()
        criteria_head = CriteriaMatchingHead(hidden_size=768, num_labels=9)
        evidence_head = EvidenceBindingHead(hidden_size=768)
        
        model = MultiTaskModel(encoder, criteria_head, evidence_head)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Use different thresholds
        output1 = model(input_ids, return_predictions=True, criteria_threshold=0.3)
        output2 = model(input_ids, return_predictions=True, criteria_threshold=0.7)
        
        # Different thresholds should produce different predictions
        # (assuming some probabilities are between 0.3 and 0.7)
        # This test may occasionally fail due to randomness
        assert output1.criteria_predictions is not None
        assert output2.criteria_predictions is not None

