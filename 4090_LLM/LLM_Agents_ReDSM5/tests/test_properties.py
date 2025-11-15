"""
Property-based tests using Hypothesis.

Tests mathematical properties of loss functions, sigmoid, and pooling.
"""

import numpy as np
import pytest
import torch
from hypothesis import given, strategies as st, settings

from src.thresholds import sigmoid
from src.losses import build_loss_fn


@given(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_sigmoid_range(x):
    """Property: sigmoid always returns values in [0, 1]."""
    result = sigmoid(np.array([x]))
    assert 0 <= result[0] <= 1


@given(
    st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=50)
def test_sigmoid_monotonic(x1, x2):
    """Property: sigmoid is monotonically increasing."""
    if x1 < x2:
        s1 = sigmoid(np.array([x1]))[0]
        s2 = sigmoid(np.array([x2]))[0]
        assert s1 <= s2


@given(
    st.integers(min_value=2, max_value=20),
    st.integers(min_value=2, max_value=15)
)
@settings(max_examples=20)
def test_pooling_preserves_shape(num_windows, num_labels):
    """Property: pooling preserves label dimension."""
    logits = np.random.randn(num_windows, num_labels)

    # Max pooling
    pooled_max = logits.max(axis=0)
    assert pooled_max.shape == (num_labels,)

    # Mean pooling
    pooled_mean = logits.mean(axis=0)
    assert pooled_mean.shape == (num_labels,)


@given(st.integers(min_value=1, max_value=100))
@settings(max_examples=30)
def test_loss_non_negativity(batch_size):
    """Property: loss is always non-negative."""
    loss_fn = build_loss_fn(loss_type='bce', num_labels=9)

    logits = torch.randn(batch_size, 9)
    labels = torch.randint(0, 2, (batch_size, 9)).float()

    loss = loss_fn(logits, labels)

    assert loss.item() >= 0
    assert torch.isfinite(loss)
