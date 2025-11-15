"""
Test LLM model building with LoRA/QLoRA and classification head.
"""

import pytest
import torch

from src.models import build_model


@pytest.fixture
def tiny_model_config():
    """Configuration for tiny model testing."""
    return {
        'model_id': 'hf-internal-testing/tiny-random-LlamaForSequenceClassification',
        'num_labels': 9,
        'method': 'full_ft'
    }


def test_build_model_full_ft(tiny_model_config):
    """Test building full fine-tuning model."""
    model = build_model(**tiny_model_config)

    assert model is not None
    assert hasattr(model, 'config')
    assert model.config.num_labels == 9

    # All parameters should be trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    assert trainable_params == total_params


def test_build_model_lora(tiny_model_config):
    """Test building LoRA model."""
    tiny_model_config['method'] = 'lora'
    tiny_model_config['lora_r'] = 8
    tiny_model_config['lora_alpha'] = 16

    model = build_model(**tiny_model_config)

    assert model is not None

    # Count trainable vs total parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    # LoRA should reduce trainable parameters
    assert trainable_params < total_params


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="QLoRA requires CUDA"
)
def test_build_model_qlora(tiny_model_config):
    """Test building QLoRA model."""
    try:
        import bitsandbytes
    except ImportError:
        pytest.skip("bitsandbytes not available")

    tiny_model_config['method'] = 'qlora'
    tiny_model_config['lora_r'] = 8
    tiny_model_config['lora_alpha'] = 16

    model = build_model(**tiny_model_config)

    assert model is not None

    # QLoRA should have even fewer trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    assert trainable_params < total_params


def test_model_forward_pass(tiny_model_config):
    """Test forward pass through model."""
    model = build_model(**tiny_model_config)
    model.eval()

    # Create dummy input
    batch_size = 4
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs.logits is not None
    assert outputs.logits.shape == (batch_size, 9)


def test_model_training_mode(tiny_model_config):
    """Test that model can be set to training mode."""
    model = build_model(**tiny_model_config)

    # Switch to training mode
    model.train()
    assert model.training

    # Switch to eval mode
    model.eval()
    assert not model.training


def test_model_gradient_flow(tiny_model_config):
    """Test that gradients flow through the model."""
    model = build_model(**tiny_model_config)
    model.train()

    # Create dummy input and labels
    batch_size = 2
    seq_length = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, 2, (batch_size, 9)).float()

    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    # Backward pass
    loss.backward()

    # Check that at least some parameters have gradients
    has_grad = False
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all()
            has_grad = True

    assert has_grad, "No parameters received gradients"


def test_model_save_and_load(tiny_model_config, tmp_path):
    """Test that model can be saved and loaded."""
    model = build_model(**tiny_model_config)

    # Save model
    save_path = tmp_path / "test_model"
    model.save_pretrained(save_path)

    # Check that files were created
    assert save_path.exists()
    assert (save_path / "config.json").exists()

    # Load model
    from transformers import AutoModelForSequenceClassification
    loaded_model = AutoModelForSequenceClassification.from_pretrained(save_path)

    assert loaded_model is not None
    assert loaded_model.config.num_labels == 9
