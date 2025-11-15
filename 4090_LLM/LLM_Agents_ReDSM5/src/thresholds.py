from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from sklearn.metrics import f1_score, log_loss


@dataclass
class ThresholdResult:
    thresholds: np.ndarray
    per_label_f1: np.ndarray
    temperatures: Optional[np.ndarray] = None


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_grid(start: float = 0.01, end: float = 0.99, step: float = 0.01) -> np.ndarray:
    return np.arange(start, end + 1e-9, step, dtype=np.float32)


def grid_search_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    grid: Optional[Iterable[float]] = None,
) -> ThresholdResult:
    num_labels = labels.shape[1]
    grid = np.asarray(list(grid or make_grid()), dtype=np.float32)
    best_thresholds = np.full(num_labels, 0.5, dtype=np.float32)
    per_label_scores = np.zeros(num_labels, dtype=np.float32)

    for idx in range(num_labels):
        label_truth = labels[:, idx]
        best_score = -1.0
        best_t = 0.5
        for t in grid:
            preds = (probs[:, idx] >= t).astype(int)
            score = f1_score(label_truth, preds, zero_division=0)
            if score > best_score:
                best_score = score
                best_t = float(t)
        best_thresholds[idx] = best_t
        per_label_scores[idx] = best_score if best_score >= 0 else 0.0

    return ThresholdResult(thresholds=best_thresholds, per_label_f1=per_label_scores)


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (probs >= thresholds[None, :]).astype(int)


def temperature_grid_search(
    logits: np.ndarray,
    labels: np.ndarray,
    grid: Iterable[float],
) -> np.ndarray:
    temperatures = np.ones(logits.shape[1], dtype=np.float32)
    grid = np.asarray(list(grid), dtype=np.float32)
    if grid.size == 0:
        return temperatures

    for idx in range(logits.shape[1]):
        best_temp = 1.0
        best_loss = float("inf")
        for temp in grid:
            scaled = logits[:, idx] / max(temp, 1e-6)
            probs = sigmoid(scaled)
            eps = 1e-6
            probs = np.clip(probs, eps, 1 - eps)
            loss = log_loss(labels[:, idx], probs, labels=[0, 1])
            if loss < best_loss:
                best_loss = loss
                best_temp = float(temp)
        temperatures[idx] = best_temp
    return temperatures


def apply_temperature_scaling(logits: np.ndarray, temperatures: Optional[np.ndarray]) -> np.ndarray:
    if temperatures is None:
        return logits
    temps = np.asarray(temperatures, dtype=np.float32)
    temps = np.clip(temps, 1e-6, None)
    return logits / temps[None, :]


__all__ = [
    "ThresholdResult",
    "grid_search_thresholds",
    "apply_thresholds",
    "temperature_grid_search",
    "apply_temperature_scaling",
    "sigmoid",
    "make_grid",
]
