# File: src/utils/metrics.py
"""Metrics computation for span and classification evaluation."""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging

from .decode import compute_span_iou

logger = logging.getLogger(__name__)


class SpanMetrics:
    """Metrics for span-based evaluation."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        """Initialize span metrics.

        Args:
            iou_threshold: IoU threshold for considering a span match
        """
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.total_predictions = 0
        self.total_gold = 0
        self.total_matches = 0
        self.exact_matches = 0
        self.total_examples = 0

    def update(
        self,
        predicted_spans: List[List[Tuple[int, int]]],
        gold_spans: List[List[Tuple[int, int]]],
    ) -> None:
        """Update metrics with a batch of predictions and gold spans.

        Args:
            predicted_spans: List of predicted span lists (one per example)
            gold_spans: List of gold span lists (one per example)
        """
        if len(predicted_spans) != len(gold_spans):
            raise ValueError("Predicted and gold spans must have same length")

        for pred_spans, gold_spans_ex in zip(predicted_spans, gold_spans):
            self._update_single_example(pred_spans, gold_spans_ex)

    def _update_single_example(
        self,
        pred_spans: List[Tuple[int, int]],
        gold_spans: List[Tuple[int, int]],
    ) -> None:
        """Update metrics for a single example."""
        self.total_examples += 1
        self.total_predictions += len(pred_spans)
        self.total_gold += len(gold_spans)

        # Check for exact match (same number of spans, all match exactly)
        if len(pred_spans) == len(gold_spans):
            exact_match = True
            for pred_span in pred_spans:
                if pred_span not in gold_spans:
                    exact_match = False
                    break
            if exact_match:
                self.exact_matches += 1

        # Count IoU-based matches
        matched_gold = set()
        for pred_span in pred_spans:
            for i, gold_span in enumerate(gold_spans):
                if i not in matched_gold:
                    iou = compute_span_iou(pred_span, gold_span)
                    if iou >= self.iou_threshold:
                        self.total_matches += 1
                        matched_gold.add(i)
                        break

    def compute(self) -> Dict[str, float]:
        """Compute final metrics.

        Returns:
            Dictionary of metric values
        """
        if self.total_predictions == 0:
            precision = 0.0
        else:
            precision = self.total_matches / self.total_predictions

        if self.total_gold == 0:
            recall = 0.0
        else:
            recall = self.total_matches / self.total_gold

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        if self.total_examples == 0:
            exact_match_rate = 0.0
        else:
            exact_match_rate = self.exact_matches / self.total_examples

        return {
            "span_precision": precision,
            "span_recall": recall,
            "span_f1": f1,
            "span_exact_match": exact_match_rate,
            "span_total_predictions": self.total_predictions,
            "span_total_gold": self.total_gold,
            "span_total_matches": self.total_matches,
        }


class ClassificationMetrics:
    """Metrics for classification evaluation."""

    def __init__(self, num_labels: int, label_names: Optional[List[str]] = None) -> None:
        """Initialize classification metrics.

        Args:
            num_labels: Number of classification labels
            label_names: Optional label names for reporting
        """
        self.num_labels = num_labels
        self.label_names = label_names or [f"label_{i}" for i in range(num_labels)]
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.predictions = []
        self.gold_labels = []

    def update(
        self,
        predicted_labels: List[int],
        gold_labels: List[int],
    ) -> None:
        """Update metrics with predictions and gold labels.

        Args:
            predicted_labels: List of predicted label indices
            gold_labels: List of gold label indices
        """
        self.predictions.extend(predicted_labels)
        self.gold_labels.extend(gold_labels)

    def compute(self) -> Dict[str, float]:
        """Compute classification metrics.

        Returns:
            Dictionary of metric values
        """
        if not self.predictions or not self.gold_labels:
            return {
                "cls_accuracy": 0.0,
                "cls_macro_precision": 0.0,
                "cls_macro_recall": 0.0,
                "cls_macro_f1": 0.0,
                "cls_micro_precision": 0.0,
                "cls_micro_recall": 0.0,
                "cls_micro_f1": 0.0,
            }

        # Basic accuracy
        accuracy = accuracy_score(self.gold_labels, self.predictions)

        # Precision, recall, F1 for each class and averaged
        precision, recall, f1, support = precision_recall_fscore_support(
            self.gold_labels,
            self.predictions,
            average=None,
            zero_division=0,
        )

        # Macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)

        # Micro averages
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            self.gold_labels,
            self.predictions,
            average='micro',
            zero_division=0,
        )

        results = {
            "cls_accuracy": accuracy,
            "cls_macro_precision": macro_precision,
            "cls_macro_recall": macro_recall,
            "cls_macro_f1": macro_f1,
            "cls_micro_precision": micro_precision,
            "cls_micro_recall": micro_recall,
            "cls_micro_f1": micro_f1,
        }

        # Per-class metrics
        for i, label_name in enumerate(self.label_names):
            if i < len(precision):
                results[f"cls_{label_name}_precision"] = precision[i]
                results[f"cls_{label_name}_recall"] = recall[i]
                results[f"cls_{label_name}_f1"] = f1[i]
                results[f"cls_{label_name}_support"] = support[i]

        return results


class CombinedMetrics:
    """Combined metrics for span and classification tasks."""

    def __init__(
        self,
        span_iou_threshold: float = 0.5,
        num_labels: int = 2,
        label_names: Optional[List[str]] = None,
        span_weight: float = 0.5,
        cls_weight: float = 0.5,
    ) -> None:
        """Initialize combined metrics.

        Args:
            span_iou_threshold: IoU threshold for span matching
            num_labels: Number of classification labels
            label_names: Optional label names
            span_weight: Weight for span F1 in combined score
            cls_weight: Weight for classification F1 in combined score
        """
        self.span_metrics = SpanMetrics(span_iou_threshold)
        self.cls_metrics = ClassificationMetrics(num_labels, label_names)
        self.span_weight = span_weight
        self.cls_weight = cls_weight

    def reset(self) -> None:
        """Reset all metrics."""
        self.span_metrics.reset()
        self.cls_metrics.reset()

    def update(
        self,
        predicted_spans: List[List[Tuple[int, int]]],
        gold_spans: List[List[Tuple[int, int]]],
        predicted_labels: Optional[List[int]] = None,
        gold_labels: Optional[List[int]] = None,
    ) -> None:
        """Update all metrics.

        Args:
            predicted_spans: Predicted span lists
            gold_spans: Gold span lists
            predicted_labels: Optional predicted labels
            gold_labels: Optional gold labels
        """
        self.span_metrics.update(predicted_spans, gold_spans)

        if predicted_labels is not None and gold_labels is not None:
            self.cls_metrics.update(predicted_labels, gold_labels)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics including combined score.

        Returns:
            Dictionary of all metric values
        """
        span_results = self.span_metrics.compute()
        cls_results = self.cls_metrics.compute()

        # Combine results
        results = {**span_results, **cls_results}

        # Compute combined score
        span_f1 = span_results["span_f1"]
        cls_f1 = cls_results["cls_macro_f1"]

        combined_score = self.span_weight * span_f1 + self.cls_weight * cls_f1
        results["combined_score"] = combined_score

        return results


def evaluate_predictions(
    predictions: Dict[str, List[Dict[str, Any]]],
    gold_data: List[Dict[str, Any]],
    span_iou_threshold: float = 0.5,
    label_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Evaluate predictions against gold data.

    Args:
        predictions: Dictionary mapping example IDs to prediction lists
        gold_data: List of gold examples
        span_iou_threshold: IoU threshold for span matching
        label_names: Optional label names

    Returns:
        Dictionary of evaluation metrics
    """
    # Extract gold spans and labels
    gold_spans_by_id = {}
    gold_labels_by_id = {}

    for example in gold_data:
        example_id = example["id"]
        gold_spans_by_id[example_id] = example.get("evidence_char_spans", [])
        if "label" in example:
            gold_labels_by_id[example_id] = example["label"]

    # Determine number of labels
    if gold_labels_by_id:
        num_labels = max(gold_labels_by_id.values()) + 1
    else:
        num_labels = 2

    # Initialize metrics
    metrics = CombinedMetrics(
        span_iou_threshold=span_iou_threshold,
        num_labels=num_labels,
        label_names=label_names,
    )

    # Collect predictions and gold for evaluation
    pred_spans_list = []
    gold_spans_list = []
    pred_labels_list = []
    gold_labels_list = []

    for example_id in gold_spans_by_id:
        # Get predicted spans
        if example_id in predictions:
            pred_spans = [(span["start_char"], span["end_char"])
                         for span in predictions[example_id]]
        else:
            pred_spans = []

        # Get gold spans
        gold_spans = gold_spans_by_id[example_id]

        pred_spans_list.append(pred_spans)
        gold_spans_list.append(gold_spans)

        # Handle labels if available
        if example_id in gold_labels_by_id:
            gold_labels_list.append(gold_labels_by_id[example_id])

            # For now, assume binary classification based on span presence
            # In practice, you'd get this from model's classification head
            pred_label = 1 if pred_spans else 0
            pred_labels_list.append(pred_label)

    # Update metrics
    if pred_labels_list and gold_labels_list:
        metrics.update(
            pred_spans_list,
            gold_spans_list,
            pred_labels_list,
            gold_labels_list,
        )
    else:
        metrics.update(pred_spans_list, gold_spans_list)

    return metrics.compute()


def compute_span_overlap_stats(
    predicted_spans: List[Tuple[int, int]],
    gold_spans: List[Tuple[int, int]],
) -> Dict[str, Any]:
    """Compute detailed overlap statistics between predicted and gold spans.

    Args:
        predicted_spans: List of predicted spans
        gold_spans: List of gold spans

    Returns:
        Dictionary with overlap statistics
    """
    if not predicted_spans and not gold_spans:
        return {
            "total_overlap": 0,
            "avg_iou": 0.0,
            "max_iou": 0.0,
            "best_matches": [],
        }

    if not predicted_spans or not gold_spans:
        return {
            "total_overlap": 0,
            "avg_iou": 0.0,
            "max_iou": 0.0,
            "best_matches": [],
        }

    # Compute all pairwise IoUs
    ious = []
    best_matches = []

    for pred_span in predicted_spans:
        best_iou = 0.0
        best_gold_span = None

        for gold_span in gold_spans:
            iou = compute_span_iou(pred_span, gold_span)
            ious.append(iou)

            if iou > best_iou:
                best_iou = iou
                best_gold_span = gold_span

        if best_gold_span is not None:
            best_matches.append({
                "predicted": pred_span,
                "gold": best_gold_span,
                "iou": best_iou,
            })

    return {
        "total_overlap": sum(1 for iou in ious if iou > 0),
        "avg_iou": np.mean(ious) if ious else 0.0,
        "max_iou": max(ious) if ious else 0.0,
        "best_matches": best_matches,
    }