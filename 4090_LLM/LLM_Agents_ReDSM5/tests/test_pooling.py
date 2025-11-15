"""
Test pooling strategies for multi-window document aggregation.

Tests max, mean, and logit_sum pooling methods.
"""

import numpy as np
import pytest


def test_max_pooling():
    """Test max pooling across windows."""
    logits = np.array([
        [1.0, 2.0],
        [3.0, 1.0],
        [2.0, 4.0]
    ])

    pooled = logits.max(axis=0)

    assert pooled.shape == (2,)
    assert np.allclose(pooled, [3.0, 4.0])


def test_mean_pooling():
    """Test mean pooling across windows."""
    logits = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])

    pooled = logits.mean(axis=0)

    assert pooled.shape == (2,)
    assert np.allclose(pooled, [3.0, 4.0])


def test_logit_sum_pooling():
    """Test logit sum (probability product) pooling."""
    from scipy.special import expit, logit

    logits = np.array([
        [1.0, -1.0],
        [2.0, -2.0]
    ])

    # Convert to probabilities, multiply, convert back to logits
    probs = expit(logits)
    pooled_prob = probs.prod(axis=0)
    pooled_logit = logit(np.clip(pooled_prob, 1e-7, 1 - 1e-7))

    assert pooled_logit.shape == (2,)
    assert np.isfinite(pooled_logit).all()


def test_pooling_shape_preservation():
    """Test that pooling preserves label dimension."""
    num_windows = 5
    num_labels = 9

    logits = np.random.randn(num_windows, num_labels)

    # Max pooling
    pooled_max = logits.max(axis=0)
    assert pooled_max.shape == (num_labels,)

    # Mean pooling
    pooled_mean = logits.mean(axis=0)
    assert pooled_mean.shape == (num_labels,)


def test_single_window_pooling():
    """Test pooling with single window (should return same values)."""
    logits = np.array([[1.0, 2.0, 3.0]])

    pooled_max = logits.max(axis=0)
    pooled_mean = logits.mean(axis=0)

    assert np.allclose(pooled_max, [1.0, 2.0, 3.0])
    assert np.allclose(pooled_mean, [1.0, 2.0, 3.0])
