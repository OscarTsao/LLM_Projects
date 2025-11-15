from __future__ import annotations

import json
import math
import string
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation/articles for fair F1 comparison."""

    text = text.lower()
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    tokens = text.split()
    return " ".join(tokens)


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score mirroring the SQuAD evaluation style."""

    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    """Exact match score using the normalized representation."""

    return float(normalize_text(prediction) == normalize_text(ground_truth))


def aggregate_metrics(predictions: Dict[str, str], references: Dict[str, str]) -> Dict[str, float]:
    """Compute corpus-level EM and F1."""

    f1_scores: List[float] = []
    em_scores: List[float] = []
    for example_id, reference in references.items():
        prediction = predictions.get(example_id, "")
        f1_scores.append(f1_score(prediction, reference))
        em_scores.append(exact_match(prediction, reference))
    return {
        "f1": float(sum(f1_scores) / max(len(f1_scores), 1)),
        "exact_match": float(sum(em_scores) / max(len(em_scores), 1)),
    }


def save_metrics(metrics: Dict[str, float], path: Path) -> None:
    """Persist metrics dictionary as json."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


__all__ = [
    "f1_score",
    "exact_match",
    "aggregate_metrics",
    "save_metrics",
]
