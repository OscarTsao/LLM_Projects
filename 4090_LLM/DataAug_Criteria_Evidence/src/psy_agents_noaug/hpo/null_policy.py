"""Utilities for applying QA/null threshold policies."""

from __future__ import annotations

import numpy as np
import torch


def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to ``logits``."""

    temperature = max(1e-3, float(temperature))
    return logits / temperature


def apply_null_policy(
    probs: np.ndarray,
    *,
    strategy: str,
    threshold: float,
    ratio: float,
) -> np.ndarray:
    """Adjust binary predictions according to a null strategy."""

    strategy = strategy.lower()
    preds = probs >= 0.5

    if strategy == "none":
        return preds.astype(int)

    if strategy == "threshold":
        return (probs >= threshold).astype(int)

    if strategy == "ratio":
        # Keep top-K predictions determined by ratio of positives
        k = max(1, int(len(probs) * ratio))
        top_idx = np.argsort(probs)[-k:]
        mask = np.zeros_like(probs, dtype=int)
        mask[top_idx] = 1
        return mask

    if strategy == "calibrated":
        # Simple Platt-like calibration: adjust threshold using moving average
        running_mean = probs.mean()
        calibrated_threshold = (running_mean + threshold) / 2.0
        return (probs >= calibrated_threshold).astype(int)

    raise ValueError(f"Unsupported null strategy: {strategy}")
