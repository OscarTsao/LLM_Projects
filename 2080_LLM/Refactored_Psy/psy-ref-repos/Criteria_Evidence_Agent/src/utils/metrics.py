"""Metrics computation utilities."""

from typing import Dict, List

import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import f1_score, roc_auc_score


def prepare_thresholds(head_cfg: DictConfig) -> np.ndarray:
    """Prepare classification thresholds from configuration.

    Args:
        head_cfg: Head configuration containing labels and thresholds

    Returns:
        Array of thresholds for each label
    """
    labels = list(head_cfg.labels)
    thresholds_cfg = head_cfg.get("thresholds", {})
    thresholds = [thresholds_cfg.get(label, 0.5) for label in labels]
    return np.asarray(thresholds, dtype=np.float32)


def compute_multi_label_metrics(
    head_name: str,
    y_true: np.ndarray,
    y_probs: np.ndarray,
    thresholds: np.ndarray,
) -> Dict[str, float]:
    """Compute multi-label classification metrics.

    Args:
        head_name: Name of the classification head
        y_true: Ground truth labels (batch_size, num_labels)
        y_probs: Predicted probabilities (batch_size, num_labels)
        thresholds: Classification thresholds (num_labels,)

    Returns:
        Dictionary of computed metrics
    """
    thresholds = thresholds.reshape(1, -1)
    y_pred = (y_probs >= thresholds).astype(int)

    metrics: Dict[str, float] = {}
    metrics[f"val_{head_name}_micro_f1"] = f1_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    metrics[f"val_{head_name}_macro_f1"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    try:
        metrics[f"val_{head_name}_roc_auc"] = roc_auc_score(y_true, y_probs, average="macro")
    except ValueError:
        metrics[f"val_{head_name}_roc_auc"] = float("nan")

    return metrics


def compute_token_metrics(
    head_name: str,
    logits: List[np.ndarray],
    labels: List[np.ndarray],
    ignore_index: int,
) -> Dict[str, float]:
    """Compute token classification metrics.

    Args:
        head_name: Name of the classification head
        logits: List of logit arrays
        labels: List of label arrays
        ignore_index: Index to ignore in loss computation

    Returns:
        Dictionary of computed metrics
    """
    if not logits or not labels:
        return {}

    logits_concat = np.concatenate(logits, axis=0)
    labels_concat = np.concatenate(labels, axis=0)
    preds = logits_concat.argmax(axis=-1)
    mask = labels_concat != ignore_index

    if not np.any(mask):
        return {}

    y_true = labels_concat[mask]
    y_pred = preds[mask]

    metrics: Dict[str, float] = {}
    metrics[f"val_{head_name}_token_micro_f1"] = f1_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    metrics[f"val_{head_name}_token_macro_f1"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    return metrics


def compute_span_metrics(
    head_name: str,
    start_logits: List[np.ndarray],
    end_logits: List[np.ndarray],
    start_positions: List[np.ndarray],
    end_positions: List[np.ndarray],
    ignore_index: int,
) -> Dict[str, float]:
    """Compute span classification metrics.

    Args:
        head_name: Name of the classification head
        start_logits: List of start logit arrays
        end_logits: List of end logit arrays
        start_positions: List of start position arrays
        end_positions: List of end position arrays
        ignore_index: Index to ignore in loss computation

    Returns:
        Dictionary of computed metrics
    """
    if not start_logits or not start_positions:
        return {}

    start_preds = np.concatenate(
        [logits.argmax(axis=-1, keepdims=True) for logits in start_logits], axis=0
    ).squeeze(-1)
    end_preds = np.concatenate(
        [logits.argmax(axis=-1, keepdims=True) for logits in end_logits], axis=0
    ).squeeze(-1)
    start_targets = np.concatenate(start_positions, axis=0)
    end_targets = np.concatenate(end_positions, axis=0)

    valid_start = start_targets != ignore_index
    valid_end = end_targets != ignore_index

    metrics: Dict[str, float] = {}

    if np.any(valid_start):
        metrics[f"val_{head_name}_start_acc"] = float(
            (start_preds[valid_start] == start_targets[valid_start]).mean()
        )

    if np.any(valid_end):
        metrics[f"val_{head_name}_end_acc"] = float(
            (end_preds[valid_end] == end_targets[valid_end]).mean()
        )

    if np.any(valid_start & valid_end):
        joint_correct = (
            (start_preds == start_targets) & (end_preds == end_targets) & valid_start & valid_end
        )
        metrics[f"val_{head_name}_span_acc"] = float(joint_correct[valid_start & valid_end].mean())

    return metrics
