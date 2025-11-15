"""
Edge case and robustness tests.

Tests handling of extreme values, empty batches, and boundary conditions.
"""

import numpy as np
import pytest
import torch

from src.losses import build_loss_fn
from src.thresholds import sigmoid, grid_search_thresholds


def test_empty_batch_handling():
    """Test handling of empty prediction/label arrays."""
    probs = np.array([]).reshape(0, 9)
    labels = np.array([]).reshape(0, 9)

    # Should not crash
    try:
        thresholds = grid_search_thresholds(probs, labels)
        # If it returns, check that result is reasonable
        assert len(thresholds) == 9
    except (ValueError, ZeroDivisionError):
        # It's okay to raise an error for empty input
        pass


def test_single_sample_batch():
    """Test loss computation with single sample."""
    loss_fn = build_loss_fn(loss_type='bce', num_labels=9)

    logits = torch.randn(1, 9)
    labels = torch.randint(0, 2, (1, 9)).float()

    loss = loss_fn(logits, labels)

    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_very_large_logits():
    """Test loss with very large logits."""
    loss_fn = build_loss_fn(loss_type='bce', num_labels=9)

    logits = torch.tensor([[100.0] * 9])
    labels = torch.ones(1, 9)

    loss = loss_fn(logits, labels)

    # Should still be finite due to numerical stability
    assert torch.isfinite(loss)


def test_very_small_logits():
    """Test loss with very small (negative) logits."""
    loss_fn = build_loss_fn(loss_type='bce', num_labels=9)

    logits = torch.tensor([[-100.0] * 9])
    labels = torch.zeros(1, 9)

    loss = loss_fn(logits, labels)

    assert torch.isfinite(loss)


def test_all_zero_predictions():
    """Test threshold search when all predictions are zero."""
    probs = np.zeros((10, 5))
    labels = np.random.randint(0, 2, (10, 5))

    thresholds = grid_search_thresholds(probs, labels)

    assert len(thresholds) == 5
    assert np.isfinite(thresholds).all()


def test_all_one_predictions():
    """Test threshold search when all predictions are one."""
    probs = np.ones((10, 5))
    labels = np.random.randint(0, 2, (10, 5))

    thresholds = grid_search_thresholds(probs, labels)

    assert len(thresholds) == 5
    assert np.isfinite(thresholds).all()


def test_sigmoid_extreme_values():
    """Test sigmoid with extreme input values."""
    # Very large positive
    result_large = sigmoid(np.array([1000.0]))
    assert np.isclose(result_large[0], 1.0, atol=1e-6)

    # Very large negative
    result_small = sigmoid(np.array([-1000.0]))
    assert np.isclose(result_small[0], 0.0, atol=1e-6)


def test_mixed_extreme_values():
    """Test loss with mixed extreme positive and negative logits."""
    loss_fn = build_loss_fn(loss_type='bce', num_labels=9)

    logits = torch.tensor([
        [100.0, -100.0, 0.0, 50.0, -50.0, 10.0, -10.0, 1.0, -1.0]
    ])
    labels = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])

    loss = loss_fn(logits, labels)

    assert torch.isfinite(loss)


def test_gradient_with_zero_loss():
    """Test gradient computation when loss is very small."""
    loss_fn = build_loss_fn(loss_type='bce', num_labels=9)

    # Perfect predictions
    logits = torch.tensor([[10.0] * 9], requires_grad=True)
    labels = torch.ones(1, 9)

    loss = loss_fn(logits, labels)
    loss.backward()

    # Gradients should exist and be finite (though small)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
