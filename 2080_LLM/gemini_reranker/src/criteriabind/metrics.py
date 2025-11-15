"""Metric utilities for the reranker and span extractor."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
from sklearn import metrics


# ---------------------------------------------------------------------------
# Classification / retrieval metrics
# ---------------------------------------------------------------------------

def classification_metrics(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute precision/recall/F1/ROC-AUC/PR-AUC."""

    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)
    y_pred = (y_scores_arr >= threshold).astype(int)
    precision = metrics.precision_score(y_true_arr, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true_arr, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true_arr, y_pred, zero_division=0)
    try:
        roc_auc = metrics.roc_auc_score(y_true_arr, y_scores_arr)
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = metrics.average_precision_score(y_true_arr, y_scores_arr)
    except ValueError:
        pr_auc = float("nan")
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


def pairwise_accuracy(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> float:
    """Compute pairwise accuracy for positive vs negative scores."""

    if len(scores_pos) != len(scores_neg):
        raise ValueError("scores_pos and scores_neg must have the same length")
    comparisons = np.asarray(scores_pos) > np.asarray(scores_neg)
    return float(np.mean(comparisons))


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def mean_average_precision(
    grouped_scores: dict[str, List[Tuple[int, float]]],
) -> float:
    """Compute MAP over grouped (label, score) lists keyed by query id."""

    ap_values: list[float] = []
    for _, items in grouped_scores.items():
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        correct = 0
        precision_at_k: list[float] = []
        for k, (label, _) in enumerate(sorted_items, start=1):
            if label > 0:
                correct += 1
                precision_at_k.append(correct / k)
        if precision_at_k:
            ap_values.append(sum(precision_at_k) / len(precision_at_k))
    return float(np.mean(ap_values)) if ap_values else 0.0


def precision_at_k(labels_scores: List[Tuple[int, float]], k: int = 1) -> float:
    if not labels_scores:
        return 0.0
    top_k = sorted(labels_scores, key=lambda x: x[1], reverse=True)[:k]
    hits = sum(1 for label, _ in top_k if label > 0)
    return hits / len(top_k)


def mean_reciprocal_rank(labels_scores: List[Tuple[int, float]]) -> float:
    if not labels_scores:
        return 0.0
    sorted_items = sorted(labels_scores, key=lambda x: x[1], reverse=True)
    for idx, (label, _) in enumerate(sorted_items, start=1):
        if label > 0:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(labels_scores: List[Tuple[int, float]], k: int = 10) -> float:
    if not labels_scores:
        return 0.0
    sorted_items = sorted(labels_scores, key=lambda x: x[1], reverse=True)[:k]
    dcg = 0.0
    for rank, (label, _) in enumerate(sorted_items, start=1):
        if label > 0:
            dcg += (2 ** label - 1) / math.log2(rank + 1)
    ideal_items = sorted(labels_scores, key=lambda x: x[0], reverse=True)[:k]
    idcg = 0.0
    for rank, (label, _) in enumerate(ideal_items, start=1):
        if label > 0:
            idcg += (2 ** label - 1) / math.log2(rank + 1)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def group_scores(
    ids: Sequence[str],
    labels: Sequence[int],
    scores: Sequence[float],
) -> dict[str, List[Tuple[int, float]]]:
    """Group scores by item id for MAP calculations."""

    grouped: dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for key, label, score in zip(ids, labels, scores, strict=False):
        grouped[key].append((label, score))
    return grouped


def compute_ranking_metrics(
    grouped_scores: dict[str, List[Tuple[int, float]]],
    k: int = 10,
) -> dict[str, float]:
    """Compute MAP, precision@1, MRR, and nDCG@k over grouped scores."""

    if not grouped_scores:
        return {"map@10": 0.0, "p@1": 0.0, "mrr": 0.0, f"ndcg@{k}": 0.0}
    map_value = mean_average_precision(grouped_scores)
    p_at_1 = []
    mrr_values = []
    ndcg_values = []
    for items in grouped_scores.values():
        p_at_1.append(precision_at_k(items, k=1))
        mrr_values.append(mean_reciprocal_rank(items))
        ndcg_values.append(ndcg_at_k(items, k=k))
    metrics_dict = {
        "map@10": map_value,
        "p@1": float(np.mean(p_at_1) if p_at_1 else 0.0),
        "mrr": float(np.mean(mrr_values) if mrr_values else 0.0),
        f"ndcg@{k}": float(np.mean(ndcg_values) if ndcg_values else 0.0),
    }
    return metrics_dict


# ---------------------------------------------------------------------------
# QA helpers
# ---------------------------------------------------------------------------

def qa_exact_match(pred: str, reference: str) -> float:
    """Compute exact match metric (normalized)."""

    return 1.0 if _normalize_answer(pred) == _normalize_answer(reference) else 0.0


def qa_f1(pred: str, reference: str) -> float:
    """Compute token-level F1 between prediction and reference."""

    pred_tokens = _normalize_answer(pred).split()
    ref_tokens = _normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    num_same = sum(min(pred_tokens.count(tok), ref_tokens.count(tok)) for tok in common)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_em_f1(
    predictions: Iterable[str],
    references: Iterable[str],
) -> dict[str, float]:
    """Compute aggregated EM/F1 across examples."""

    em_scores: list[float] = []
    f1_scores: list[float] = []
    for pred, ref in zip(predictions, references, strict=False):
        em_scores.append(qa_exact_match(pred, ref))
        f1_scores.append(qa_f1(pred, ref))
    return {
        "exact_match": float(np.mean(em_scores) if em_scores else 0.0),
        "f1": float(np.mean(f1_scores) if f1_scores else 0.0),
    }


def hit_at_k(
    ranked_ids: Sequence[Sequence[str]],
    ground_truth_ids: Sequence[str],
    k: int = 3,
) -> float:
    """Compute Hit@k for evidence retrieval."""

    hits = 0
    total = 0
    for rankings, gold in zip(ranked_ids, ground_truth_ids, strict=False):
        total += 1
        if gold in rankings[:k]:
            hits += 1
    return hits / total if total else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_answer(text: str) -> str:
    """Lower, strip, and collapse whitespace."""

    import re

    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ").strip()
    return text
