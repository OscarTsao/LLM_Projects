from __future__ import annotations

import numpy as np

from src.dataaug_multi_both.utils.thresholding import tune_threshold


def test_tune_threshold_global() -> None:
    logits = np.array([[2.0, -1.0], [1.0, 1.5], [-0.5, -2.0], [0.2, 0.4]])
    labels = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])

    macro_f1, tau = tune_threshold(logits, labels, per_class=False)
    assert 0.0 <= tau <= 1.0
    assert 0 <= macro_f1 <= 1.0


def test_tune_threshold_per_class() -> None:
    logits = np.array([[2.0, -2.0], [1.0, 0.0], [-1.0, 1.5], [0.5, -0.3]])
    labels = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])

    macro_f1, thresholds = tune_threshold(logits, labels, per_class=True)
    assert isinstance(thresholds, list)
    assert len(thresholds) == logits.shape[1]
    assert 0.0 <= thresholds[0] <= 1.0
    assert 0.0 <= thresholds[1] <= 1.0
    assert 0 <= macro_f1 <= 1.0


def test_tune_threshold_shape_mismatch() -> None:
    logits = np.zeros((4, 2))
    labels = np.zeros((3, 2))
    try:
        tune_threshold(logits, labels)
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError for mismatched shapes.")
