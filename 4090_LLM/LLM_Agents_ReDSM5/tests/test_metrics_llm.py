"""
Test multi-label classification metrics for LLM outputs.
"""

import numpy as np
import pytest
from sklearn.metrics import f1_score


def test_multilabel_f1_matches_sklearn():
    """Test that our metrics match sklearn's implementation."""
    from src.metrics import compute_metrics_bundle

    # Create sample predictions and targets
    y_true = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0]
    ])
    y_pred = np.array([
        [0.9, 0.1, 0.8],
        [0.2, 0.9, 0.1],
        [0.7, 0.6, 0.3]
    ])

    # Compute metrics with our function
    result = compute_metrics_bundle(y_pred, y_true, threshold=0.5)

    # Compute sklearn baseline
    preds_binary = (y_pred > 0.5).astype(int)
    sklearn_macro_f1 = f1_score(y_true, preds_binary, average='macro', zero_division=0)

    # Check that macro F1 matches
    assert abs(result['macro_f1'] - sklearn_macro_f1) < 0.01


def test_metrics_shape():
    """Test that metrics are computed for correct shape inputs."""
    from src.metrics import compute_metrics_bundle

    y_true = np.random.randint(0, 2, size=(10, 9))
    y_pred = np.random.rand(10, 9)

    result = compute_metrics_bundle(y_pred, y_true, threshold=0.5)

    assert 'macro_f1' in result
    assert 'micro_f1' in result
    assert 'weighted_f1' in result
