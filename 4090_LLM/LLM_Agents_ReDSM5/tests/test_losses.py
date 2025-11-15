"""
Test loss functions for multi-label classification.

Tests BCE and Focal loss with class weights, label smoothing, and gradient flow.
"""

import numpy as np
import pytest
import torch

from src.losses import build_loss_fn


def test_build_loss_fn_bce():
    """Test building BCE loss function."""
    loss_fn = build_loss_fn(loss_type='bce', num_labels=9)

    logits = torch.randn(4, 9)
    labels = torch.randint(0, 2, (4, 9)).float()

    loss = loss_fn(logits, labels)

    assert loss.item() >= 0
    assert torch.isfinite(loss)


def test_build_loss_fn_focal():
    """Test building Focal loss function."""
    loss_fn = build_loss_fn(loss_type='focal', num_labels=9, focal_gamma=2.0)

    logits = torch.randn(4, 9)
    labels = torch.randint(0, 2, (4, 9)).float()

    loss = loss_fn(logits, labels)

    assert loss.item() >= 0
    assert torch.isfinite(loss)


def test_loss_with_class_weights():
    """Test loss function with class weights."""
    class_weights = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 1.0, 2.0, 1.5, 3.0])
    loss_fn = build_loss_fn(loss_type='bce', num_labels=9, class_weights=class_weights)

    logits = torch.randn(4, 9)
    labels = torch.randint(0, 2, (4, 9)).float()

    loss = loss_fn(logits, labels)

    assert loss.item() >= 0
    assert torch.isfinite(loss)


def test_loss_with_label_smoothing():
    """Test loss function with label smoothing."""
    loss_fn = build_loss_fn(loss_type='bce', num_labels=9, label_smoothing=0.1)

    logits = torch.randn(4, 9)
    labels = torch.randint(0, 2, (4, 9)).float()

    loss = loss_fn(logits, labels)

    assert loss.item() >= 0
    assert torch.isfinite(loss)


def test_loss_gradients_finite():
    """Test that loss gradients are finite."""
    loss_fn = build_loss_fn(loss_type='bce', num_labels=9)

    logits = torch.randn(4, 9, requires_grad=True)
    labels = torch.randint(0, 2, (4, 9)).float()

    loss = loss_fn(logits, labels)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_loss_reduction_mean():
    """Test that loss reduction to mean works correctly."""
    loss_fn = build_loss_fn(loss_type='bce', num_labels=9)

    # Small batch
    logits_small = torch.randn(2, 9)
    labels_small = torch.randint(0, 2, (2, 9)).float()
    loss_small = loss_fn(logits_small, labels_small)

    # Large batch with same logits repeated
    logits_large = logits_small.repeat(4, 1)
    labels_large = labels_small.repeat(4, 1)
    loss_large = loss_fn(logits_large, labels_large)

    # Losses should be approximately equal (mean reduction)
    assert abs(loss_small.item() - loss_large.item()) < 0.1
