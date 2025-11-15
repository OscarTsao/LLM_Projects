from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def optimize_global_threshold(y_true: np.ndarray, y_prob: np.ndarray, lo: float = 0.2, hi: float = 0.6, steps: int = 101) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(lo, hi, steps):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


__all__ = ["optimize_global_threshold"]
