from __future__ import annotations

from collections import Counter
from typing import Dict

from .metrics import normalize_text


def _tokenize(text: str) -> list[str]:
    """Normalize and split text into whitespace-separated tokens."""

    return normalize_text(text).split()


def token_precision(prediction: str, ground_truth: str) -> float:
    """Compute token-level precision with SQuAD-style normalization."""

    pred_tokens = _tokenize(prediction)
    truth_tokens = _tokenize(ground_truth)

    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    return num_same / len(pred_tokens)


def token_recall(prediction: str, ground_truth: str) -> float:
    """Compute token-level recall with SQuAD-style normalization."""

    pred_tokens = _tokenize(prediction)
    truth_tokens = _tokenize(ground_truth)

    if not pred_tokens and not truth_tokens:
        return 1.0
    if not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    return num_same / len(truth_tokens)


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute the harmonic mean of precision and recall at the token level."""

    precision = token_precision(prediction, ground_truth)
    recall = token_recall(prediction, ground_truth)

    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    """Return 1.0 when normalized prediction and ground truth match exactly."""

    return float(normalize_text(prediction) == normalize_text(ground_truth))


def token_level_scores(prediction: str, ground_truth: str) -> Dict[str, float]:
    """Compute token-level precision, recall, F1, and exact match for one pair."""

    precision = token_precision(prediction, ground_truth)
    recall = token_recall(prediction, ground_truth)
    f1 = token_f1(prediction, ground_truth)
    em = exact_match(prediction, ground_truth)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": em,
    }


def aggregate_token_metrics(
    predictions: Dict[str, str], references: Dict[str, str]
) -> Dict[str, float]:
    """Average token-level metrics across a dataset keyed by example id."""

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_em = 0.0
    count = max(len(references), 1)

    for example_id, reference in references.items():
        prediction = predictions.get(example_id, "")
        scores = token_level_scores(prediction, reference)
        total_precision += scores["precision"]
        total_recall += scores["recall"]
        total_f1 += scores["f1"]
        total_em += scores["exact_match"]

    return {
        "precision": total_precision / count,
        "recall": total_recall / count,
        "f1": total_f1 / count,
        "exact_match": total_em / count,
    }


__all__ = [
    "token_precision",
    "token_recall",
    "token_f1",
    "exact_match",
    "token_level_scores",
    "aggregate_token_metrics",
]
