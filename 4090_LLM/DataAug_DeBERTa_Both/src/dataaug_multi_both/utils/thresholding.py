"""Threshold tuning utilities for macro-F1 optimisation."""

from __future__ import annotations

import numpy as np
from typing import Iterable, Sequence, Tuple

__all__ = ["tune_threshold"]


def tune_threshold(
    logits: np.ndarray | Sequence[Sequence[float]],
    labels: np.ndarray | Sequence[Sequence[int]],
    *,
    per_class: bool = False,
) -> Tuple[float, float | Sequence[float]]:
    """Tune decision thresholds to maximise macro-F1."""

    logits_arr = np.asarray(logits, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int8)

    if logits_arr.ndim != 2 or labels_arr.ndim != 2:
        raise ValueError("logits and labels must be 2D arrays.")
    if logits_arr.shape != labels_arr.shape:
        raise ValueError("logits and labels must have the same shape.")

    probabilities = _sigmoid(logits_arr)

    if per_class:
        thresholds = []
        per_class_f1 = []
        for column in range(probabilities.shape[1]):
            best_f1, best_tau = _best_threshold(probabilities[:, column], labels_arr[:, column])
            thresholds.append(best_tau)
            per_class_f1.append(best_f1)
        macro_f1 = float(np.mean(per_class_f1))
        return macro_f1, thresholds

    best_f1, best_tau = _best_threshold(probabilities, labels_arr)
    return best_f1, best_tau


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _best_threshold(probabilities: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    if probabilities.ndim == 1:
        probabilities = probabilities[:, None]
        labels = labels[:, None]

    candidate_grid = _build_candidate_grid(probabilities)
    best_tau = 0.5
    best_score = -1.0

    for tau in candidate_grid:
        predictions = probabilities >= tau
        score = _macro_f1(predictions, labels)
        if score > best_score + 1e-9:
            best_score = score
            best_tau = float(tau)
    return float(best_score), float(best_tau)


def _build_candidate_grid(probabilities: np.ndarray) -> np.ndarray:
    base_grid = np.linspace(0.05, 0.95, num=19, dtype=np.float64)
    unique_probs = np.unique(probabilities)
    if unique_probs.size <= 256:
        combined = np.unique(np.concatenate([base_grid, unique_probs]))
    else:
        quantiles = np.quantile(unique_probs, np.linspace(0.05, 0.95, num=32))
        combined = np.unique(np.concatenate([base_grid, quantiles]))
    return np.clip(combined, 0.0, 1.0)


def _macro_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    labels_bool = labels.astype(bool)
    preds_bool = predictions.astype(bool)

    tp = np.logical_and(preds_bool, labels_bool).sum(axis=0, dtype=float)
    fp = np.logical_and(preds_bool, np.logical_not(labels_bool)).sum(axis=0, dtype=float)
    fn = np.logical_and(np.logical_not(preds_bool), labels_bool).sum(axis=0, dtype=float)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)

    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(np.mean(f1))
