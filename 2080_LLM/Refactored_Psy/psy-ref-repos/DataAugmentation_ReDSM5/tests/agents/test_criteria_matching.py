"""Tests for criteria matching agent."""

import pytest
import torch
from transformers import AutoTokenizer

from src.agents.base import CriteriaMatchingConfig
from src.agents.criteria_matching import CriteriaMatchingAgent


@pytest.fixture
def criteria_config():
    """Create a test configuration for criteria matching agent."""
    return CriteriaMatchingConfig(
        model_name="google-bert/bert-base-uncased",
        max_seq_length=128,
        dropout=0.1,
        classifier_hidden_sizes=[64],
        loss_type="bce",
        learning_rate=1e-4,
        weight_decay=0.01
    )


@pytest.fixture
def criteria_agent(criteria_config):
    """Create a criteria matching agent for testing."""
    return CriteriaMatchingAgent(criteria_config)


@pytest.fixture
def sample_inputs():
    """Create sample inputs for testing."""
    posts = [
        "I feel very sad and depressed all the time.",
        "I love going out and having fun with friends."
    ]
    criteria = [
        "Depressed mood most of the day, nearly every day.",
        "Depressed mood most of the day, nearly every day."
    ]
    return posts, criteria


def test_criteria_agent_initialization(criteria_agent, criteria_config):
    """Test that the criteria matching agent initializes correctly."""
    assert criteria_agent.config == criteria_config
    assert hasattr(criteria_agent, 'encoder')
    assert hasattr(criteria_agent, 'classifier')
    assert hasattr(criteria_agent, 'tokenizer')


def test_criteria_agent_forward_pass(criteria_agent, sample_inputs):
    """Test forward pass through the criteria matching agent."""
    posts, criteria = sample_inputs
    
    # Tokenize inputs
    inputs = criteria_agent.tokenize_inputs(posts, criteria)
    
    # Forward pass
    outputs = criteria_agent(**inputs)
    
    # Check output structure
    assert hasattr(outputs, 'predictions')
    assert hasattr(outputs, 'confidence')
    assert hasattr(outputs, 'logits')
    assert hasattr(outputs, 'probabilities')
    
    # Check tensor shapes
    batch_size = len(posts)
    assert outputs.predictions.shape == (batch_size,)
    assert outputs.confidence.shape == (batch_size,)
    assert outputs.logits.shape == (batch_size,)
    assert outputs.probabilities.shape == (batch_size,)


def test_criteria_agent_predict(criteria_agent, sample_inputs):
    """Test prediction without gradients."""
    posts, criteria = sample_inputs
    
    # Make predictions
    outputs = criteria_agent.predict_batch(posts, criteria)
    
    # Check that predictions are valid
    assert torch.all((outputs.predictions >= 0) & (outputs.predictions <= 1))
    assert torch.all((outputs.probabilities >= 0) & (outputs.probabilities <= 1))
    assert torch.all((outputs.confidence >= 0) & (outputs.confidence <= 1))


def test_criteria_agent_loss_computation(criteria_agent, sample_inputs):
    """Test loss computation."""
    posts, criteria = sample_inputs
    
    # Create inputs and targets
    inputs = criteria_agent.tokenize_inputs(posts, criteria)
    targets = torch.tensor([1.0, 0.0])  # Binary labels
    
    # Forward pass
    outputs = criteria_agent(**inputs)
    
    # Compute loss
    loss = criteria_agent.get_loss(outputs, targets)
    
    # Check loss properties
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert loss.item() >= 0  # Non-negative


def test_criteria_agent_tokenization(criteria_agent, sample_inputs):
    """Test input tokenization."""
    posts, criteria = sample_inputs
    
    # Tokenize
    inputs = criteria_agent.tokenize_inputs(posts, criteria)
    
    # Check required keys
    assert 'input_ids' in inputs
    assert 'attention_mask' in inputs
    
    # Check tensor properties
    assert isinstance(inputs['input_ids'], torch.Tensor)
    assert isinstance(inputs['attention_mask'], torch.Tensor)
    assert inputs['input_ids'].shape == inputs['attention_mask'].shape
    
    # Check sequence length
    assert inputs['input_ids'].size(1) <= criteria_agent.config.max_seq_length


def test_criteria_agent_gradient_checkpointing(criteria_agent):
    """Test gradient checkpointing functionality."""
    # Enable gradient checkpointing
    criteria_agent.enable_gradient_checkpointing()
    
    # Check that it doesn't raise an error
    # (Actual functionality testing would require training)
    assert True


def test_criteria_agent_device_handling(criteria_agent):
    """Test device handling."""
    # Test moving to CPU (should work regardless of CUDA availability)
    criteria_agent = criteria_agent.to('cpu')
    assert next(criteria_agent.parameters()).device.type == 'cpu'
    
    # Test device utility function
    sample_dict = {'tensor': torch.tensor([1, 2, 3])}
    moved_dict = criteria_agent.to_device(sample_dict)
    assert moved_dict['tensor'].device == criteria_agent.device


@pytest.mark.parametrize("loss_type", ["bce", "focal", "adaptive_focal"])
def test_criteria_agent_loss_types(loss_type):
    """Test different loss function types."""
    config = CriteriaMatchingConfig(
        model_name="google-bert/bert-base-uncased",
        max_seq_length=128,
        loss_type=loss_type,
        alpha=0.25,
        gamma=2.0,
        delta=1.0
    )
    
    agent = CriteriaMatchingAgent(config)
    
    # Test that the agent initializes with the correct loss function
    assert agent.config.loss_type == loss_type
    
    # Test loss computation
    dummy_logits = torch.randn(2)
    dummy_targets = torch.tensor([1.0, 0.0])
    
    from src.agents.base import AgentOutput
    outputs = AgentOutput(
        predictions=torch.sigmoid(dummy_logits),
        confidence=torch.ones(2),
        logits=dummy_logits,
        probabilities=torch.sigmoid(dummy_logits)
    )
    
    loss = agent.get_loss(outputs, dummy_targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_criteria_agent_batch_processing(criteria_agent):
    """Test batch processing with different batch sizes."""
    posts = ["Test post 1", "Test post 2", "Test post 3"]
    criteria = ["Test criterion"] * 3
    
    # Test batch prediction
    outputs = criteria_agent.predict_batch(posts, criteria)
    
    assert outputs.predictions.shape[0] == 3
    assert outputs.confidence.shape[0] == 3
    assert outputs.logits.shape[0] == 3
    assert outputs.probabilities.shape[0] == 3
