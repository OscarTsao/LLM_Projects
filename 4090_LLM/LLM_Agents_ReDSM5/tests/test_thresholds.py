"""
Test threshold optimization for multi-label classification.

Tests grid search for per-label thresholds and temperature scaling.
"""

import numpy as np
import pytest

from src.thresholds import grid_search_thresholds, sigmoid


def test_sigmoid():
    """Test sigmoid function."""
    logits = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    probs = sigmoid(logits)

    assert (probs >= 0).all() and (probs <= 1).all()
    assert np.isclose(probs[2], 0.5)
    assert probs[0] < probs[1] < probs[2] < probs[3] < probs[4]


def test_grid_search_basic():
    """Test basic threshold grid search."""
    # Create simple predictions and labels
    probs = np.array([
        [0.9, 0.1, 0.8],
        [0.2, 0.8, 0.3],
        [0.7, 0.3, 0.9]
    ])
    labels = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])

    thresholds = grid_search_thresholds(probs, labels)

    assert len(thresholds) == 3
    assert (thresholds > 0).all() and (thresholds < 1).all()


def test_grid_search_all_positive():
    """Test threshold search when all labels are positive."""
    probs = np.random.rand(10, 5)
    labels = np.ones((10, 5))

    thresholds = grid_search_thresholds(probs, labels)

    assert len(thresholds) == 5
    # Thresholds should be relatively low to capture all positives
    assert (thresholds < 0.8).any()


def test_grid_search_all_negative():
    """Test threshold search when all labels are negative."""
    probs = np.random.rand(10, 5)
    labels = np.zeros((10, 5))

    thresholds = grid_search_thresholds(probs, labels)

    assert len(thresholds) == 5
    # Thresholds should be relatively high to avoid false positives
    assert (thresholds > 0.2).any()
