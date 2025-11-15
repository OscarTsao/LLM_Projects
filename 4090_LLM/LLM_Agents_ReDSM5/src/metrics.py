from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class MetricsResult:
    metrics: Dict[str, float]
    per_label: List[Dict[str, float]]
    confusion: List[Dict[str, int]]


def compute_global_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def compute_per_label_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    label_names: Sequence[str],
) -> List[Dict[str, float]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    pr_auc_values = []
    for idx in range(labels.shape[1]):
        try:
            pr_auc = average_precision_score(labels[:, idx], probs[:, idx])
        except ValueError:
            pr_auc = float("nan")
        pr_auc_values.append(pr_auc)
    rows: List[Dict[str, float]] = []
    for idx, name in enumerate(label_names):
        rows.append(
            {
                "label": name,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
                "pr_auc": float(pr_auc_values[idx]),
            }
        )
    return rows


def compute_confusion_counts(labels: np.ndarray, preds: np.ndarray, label_names: Sequence[str]) -> List[Dict[str, int]]:
    confusion: List[Dict[str, int]] = []
    for idx, name in enumerate(label_names):
        y_true = labels[:, idx]
        y_pred = preds[:, idx]
        tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
        tn = int(np.logical_and(y_true == 0, y_pred == 0).sum())
        fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
        fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
        confusion.append({"label": name, "tp": tp, "tn": tn, "fp": fp, "fn": fn})
    return confusion


def compute_metrics_bundle(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    label_names: Sequence[str],
) -> MetricsResult:
    metrics = compute_global_metrics(labels, preds)
    per_label = compute_per_label_metrics(labels, preds, probs, label_names)
    confusion = compute_confusion_counts(labels, preds, label_names)
    return MetricsResult(metrics=metrics, per_label=per_label, confusion=confusion)


__all__ = [
    "MetricsResult",
    "compute_metrics_bundle",
    "compute_global_metrics",
    "compute_per_label_metrics",
    "compute_confusion_counts",
]
