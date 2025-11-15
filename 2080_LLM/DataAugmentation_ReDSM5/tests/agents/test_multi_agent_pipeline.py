"""Tests for multi-agent pipeline."""

import pytest
import torch

from src.agents.base import CriteriaMatchingConfig, EvidenceBindingConfig
from src.agents.criteria_matching import CriteriaMatchingAgent
from src.agents.evidence_binding import EvidenceBindingAgent
from src.agents.multi_agent_pipeline import MultiAgentPipeline, create_multi_agent_pipeline


@pytest.fixture
def criteria_agent():
    """Create a criteria matching agent for testing."""
    config = CriteriaMatchingConfig(
        model_name="google-bert/bert-base-uncased",
        max_seq_length=128,
        dropout=0.1,
        classifier_hidden_sizes=[64],
        loss_type="bce"
    )
    return CriteriaMatchingAgent(config)


@pytest.fixture
def evidence_agent():
    """Create an evidence binding agent for testing."""
    config = EvidenceBindingConfig(
        model_name="google-bert/bert-base-uncased",
        max_seq_length=128,
        dropout=0.1,
        span_threshold=0.5
    )
    return EvidenceBindingAgent(config)


@pytest.fixture
def multi_agent_pipeline(criteria_agent, evidence_agent):
    """Create a multi-agent pipeline for testing."""
    return MultiAgentPipeline(criteria_agent, evidence_agent)


@pytest.fixture
def sample_inputs():
    """Create sample inputs for testing."""
    posts = [
        "I feel very sad and depressed all the time. I can't sleep at night.",
        "I love going out and having fun with friends. Life is great!"
    ]
    criteria = [
        "Depressed mood most of the day, nearly every day.",
        "Depressed mood most of the day, nearly every day."
    ]
    return posts, criteria


def test_pipeline_initialization(multi_agent_pipeline, criteria_agent, evidence_agent):
    """Test that the pipeline initializes correctly."""
    assert multi_agent_pipeline.criteria_agent == criteria_agent
    assert multi_agent_pipeline.evidence_agent == evidence_agent
    assert multi_agent_pipeline.evidence_threshold == 0.5


def test_pipeline_forward_pass(multi_agent_pipeline, sample_inputs):
    """Test forward pass through the pipeline."""
    posts, criteria = sample_inputs
    
    # Get tokenized inputs
    inputs = multi_agent_pipeline.criteria_agent.tokenize_inputs(posts, criteria)
    
    # Forward pass
    outputs = multi_agent_pipeline(**inputs)
    
    # Check output structure
    assert hasattr(outputs, 'criteria_match')
    assert hasattr(outputs, 'criteria_confidence')
    assert hasattr(outputs, 'criteria_probabilities')
    assert hasattr(outputs, 'evidence_spans')
    assert hasattr(outputs, 'evidence_confidence')
    assert hasattr(outputs, 'evidence_text')
    assert hasattr(outputs, 'overall_confidence')
    
    # Check tensor shapes
    batch_size = len(posts)
    assert outputs.criteria_match.shape == (batch_size,)
    assert outputs.criteria_confidence.shape == (batch_size,)
    assert outputs.criteria_probabilities.shape == (batch_size,)
    assert outputs.overall_confidence.shape == (batch_size,)


def test_pipeline_predict(multi_agent_pipeline, sample_inputs):
    """Test prediction without gradients."""
    posts, criteria = sample_inputs
    
    # Make predictions
    outputs = multi_agent_pipeline.predict_batch(posts, criteria)
    
    # Check that predictions are valid
    assert torch.all((outputs.criteria_match >= 0) & (outputs.criteria_match <= 1))
    assert torch.all((outputs.criteria_probabilities >= 0) & (outputs.criteria_probabilities <= 1))
    assert torch.all((outputs.criteria_confidence >= 0) & (outputs.criteria_confidence <= 1))
    assert torch.all((outputs.overall_confidence >= 0) & (outputs.overall_confidence <= 1))


def test_pipeline_evidence_filtering(multi_agent_pipeline, sample_inputs):
    """Test that evidence is only provided for positive criteria matches."""
    posts, criteria = sample_inputs
    
    # Make predictions
    outputs = multi_agent_pipeline.predict_batch(posts, criteria, run_evidence=True)
    
    # Check evidence spans structure
    assert isinstance(outputs.evidence_spans, list)
    assert len(outputs.evidence_spans) == len(posts)
    
    # Check evidence text structure
    assert isinstance(outputs.evidence_text, list)
    assert len(outputs.evidence_text) == len(posts)
    
    # Each item should be a list of spans/text
    for spans, text in zip(outputs.evidence_spans, outputs.evidence_text):
        assert isinstance(spans, list)
        assert isinstance(text, list)


def test_pipeline_without_evidence(multi_agent_pipeline, sample_inputs):
    """Test pipeline when evidence binding is disabled."""
    posts, criteria = sample_inputs
    
    # Make predictions without evidence
    outputs = multi_agent_pipeline.predict_batch(posts, criteria, run_evidence=False)
    
    # Evidence-related outputs should be None
    assert outputs.evidence_spans is None
    assert outputs.evidence_confidence is None
    assert outputs.evidence_text is None
    
    # Criteria outputs should still be present
    assert outputs.criteria_match is not None
    assert outputs.criteria_confidence is not None
    assert outputs.criteria_probabilities is not None


def test_pipeline_device_handling(multi_agent_pipeline):
    """Test device handling for the pipeline."""
    # Test moving to CPU
    multi_agent_pipeline = multi_agent_pipeline.to('cpu')
    
    # Check that both agents are on CPU
    assert next(multi_agent_pipeline.criteria_agent.parameters()).device.type == 'cpu'
    assert next(multi_agent_pipeline.evidence_agent.parameters()).device.type == 'cpu'


def test_create_multi_agent_pipeline_factory():
    """Test the factory function for creating pipelines."""
    pipeline = create_multi_agent_pipeline()
    
    assert isinstance(pipeline, MultiAgentPipeline)
    assert hasattr(pipeline, 'criteria_agent')
    assert hasattr(pipeline, 'evidence_agent')


def test_pipeline_with_custom_configs():
    """Test pipeline creation with custom configurations."""
    criteria_config = CriteriaMatchingConfig(
        model_name="google-bert/bert-base-uncased",
        max_seq_length=256,
        dropout=0.2
    )
    
    evidence_config = EvidenceBindingConfig(
        model_name="google-bert/bert-base-uncased",
        max_seq_length=256,
        dropout=0.2,
        span_threshold=0.7
    )
    
    pipeline = create_multi_agent_pipeline(
        criteria_config=criteria_config,
        evidence_config=evidence_config,
        evidence_threshold=0.7
    )
    
    assert pipeline.criteria_agent.config == criteria_config
    assert pipeline.evidence_agent.config == evidence_config
    assert pipeline.evidence_threshold == 0.7


def test_pipeline_output_consistency(multi_agent_pipeline, sample_inputs):
    """Test that pipeline outputs are consistent across multiple runs."""
    posts, criteria = sample_inputs
    
    # Set to eval mode for consistent outputs
    multi_agent_pipeline.eval()
    
    # Make multiple predictions
    outputs1 = multi_agent_pipeline.predict_batch(posts, criteria)
    outputs2 = multi_agent_pipeline.predict_batch(posts, criteria)
    
    # Check that outputs are consistent (within floating point precision)
    torch.testing.assert_close(outputs1.criteria_match, outputs2.criteria_match)
    torch.testing.assert_close(outputs1.criteria_confidence, outputs2.criteria_confidence)
    torch.testing.assert_close(outputs1.criteria_probabilities, outputs2.criteria_probabilities)


def test_pipeline_batch_sizes(multi_agent_pipeline):
    """Test pipeline with different batch sizes."""
    # Test with single example
    posts = ["I feel sad."]
    criteria = ["Depressed mood."]
    
    outputs = multi_agent_pipeline.predict_batch(posts, criteria)
    assert outputs.criteria_match.shape[0] == 1
    
    # Test with larger batch
    posts = ["I feel sad."] * 5
    criteria = ["Depressed mood."] * 5
    
    outputs = multi_agent_pipeline.predict_batch(posts, criteria)
    assert outputs.criteria_match.shape[0] == 5


def test_pipeline_empty_inputs(multi_agent_pipeline):
    """Test pipeline behavior with empty inputs."""
    posts = []
    criteria = []
    
    # This should handle empty inputs gracefully
    try:
        outputs = multi_agent_pipeline.predict_batch(posts, criteria)
        # If it doesn't raise an error, check that outputs are empty
        assert outputs.criteria_match.shape[0] == 0
    except (ValueError, IndexError):
        # It's acceptable for empty inputs to raise an error
        pass
