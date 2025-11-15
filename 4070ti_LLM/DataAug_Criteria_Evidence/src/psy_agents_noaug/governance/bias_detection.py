#!/usr/bin/env python
"""Bias detection and fairness metrics (Phase 25).

This module provides tools for detecting bias and measuring fairness
in ML models across protected attributes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


class FairnessMetric(str, Enum):
    """Fairness metrics."""

    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    EQUALIZED_ODDS = "equalized_odds"
    DISPARATE_IMPACT = "disparate_impact"


@dataclass
class BiasMetrics:
    """Bias and fairness metrics."""

    protected_attribute: str
    reference_group: str
    comparison_group: str

    # Demographic parity (equal positive prediction rates)
    demographic_parity_difference: float
    demographic_parity_ratio: float

    # Equal opportunity (equal TPR)
    equal_opportunity_difference: float

    # Equalized odds (equal TPR and FPR)
    equalized_odds_difference: float

    # Disparate impact
    disparate_impact_ratio: float

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_fair(
        self,
        metric: FairnessMetric = FairnessMetric.DEMOGRAPHIC_PARITY,
        threshold: float = 0.1,
    ) -> bool:
        """Check if model is fair according to metric.

        Args:
            metric: Fairness metric to check
            threshold: Fairness threshold

        Returns:
            True if fair, False otherwise
        """
        if metric == FairnessMetric.DEMOGRAPHIC_PARITY:
            return abs(self.demographic_parity_difference) <= threshold
        if metric == FairnessMetric.EQUAL_OPPORTUNITY:
            return abs(self.equal_opportunity_difference) <= threshold
        if metric == FairnessMetric.EQUALIZED_ODDS:
            return abs(self.equalized_odds_difference) <= threshold
        if metric == FairnessMetric.DISPARATE_IMPACT:
            # 80% rule: ratio should be between 0.8 and 1.25
            return 0.8 <= self.disparate_impact_ratio <= 1.25
        return False

    def get_summary(self) -> dict[str, Any]:
        """Get bias metrics summary.

        Returns:
            Summary dictionary
        """
        return {
            "protected_attribute": self.protected_attribute,
            "reference_group": self.reference_group,
            "comparison_group": self.comparison_group,
            "demographic_parity_diff": self.demographic_parity_difference,
            "equal_opportunity_diff": self.equal_opportunity_difference,
            "disparate_impact_ratio": self.disparate_impact_ratio,
            "is_fair_demographic_parity": self.is_fair(
                FairnessMetric.DEMOGRAPHIC_PARITY
            ),
            "is_fair_equal_opportunity": self.is_fair(FairnessMetric.EQUAL_OPPORTUNITY),
            "is_fair_disparate_impact": self.is_fair(FairnessMetric.DISPARATE_IMPACT),
        }


class BiasDetector:
    """Detector for bias and fairness issues."""

    def __init__(self):
        """Initialize bias detector."""
        LOGGER.info("Initialized BiasDetector")

    def calculate_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        group_mask: np.ndarray,
    ) -> dict[str, float]:
        """Calculate metrics for a protected group.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            group_mask: Boolean mask for group members

        Returns:
            Dictionary of metrics
        """
        # Filter to group
        y_true_group = y_true[group_mask]
        y_pred_group = y_pred[group_mask]

        if len(y_true_group) == 0:
            return {
                "positive_rate": 0.0,
                "true_positive_rate": 0.0,
                "false_positive_rate": 0.0,
            }

        # Positive prediction rate
        positive_rate = np.mean(y_pred_group)

        # True positive rate (sensitivity/recall)
        positive_mask = y_true_group == 1
        if np.sum(positive_mask) > 0:
            tpr = np.mean(y_pred_group[positive_mask])
        else:
            tpr = 0.0

        # False positive rate
        negative_mask = y_true_group == 0
        if np.sum(negative_mask) > 0:
            fpr = np.mean(y_pred_group[negative_mask])
        else:
            fpr = 0.0

        return {
            "positive_rate": float(positive_rate),
            "true_positive_rate": float(tpr),
            "false_positive_rate": float(fpr),
        }

    def detect_bias(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attribute: np.ndarray,
        reference_value: Any,
        comparison_value: Any,
        attribute_name: str = "protected_attribute",
    ) -> BiasMetrics:
        """Detect bias between groups.

        Args:
            y_true: True labels
            y_pred: Predicted labels (binary)
            protected_attribute: Protected attribute values
            reference_value: Reference group value
            comparison_value: Comparison group value
            attribute_name: Name of protected attribute

        Returns:
            Bias metrics
        """
        # Create group masks
        ref_mask = protected_attribute == reference_value
        comp_mask = protected_attribute == comparison_value

        # Calculate metrics for each group
        ref_metrics = self.calculate_group_metrics(y_true, y_pred, ref_mask)
        comp_metrics = self.calculate_group_metrics(y_true, y_pred, comp_mask)

        # Demographic parity (equal positive prediction rates)
        dp_diff = comp_metrics["positive_rate"] - ref_metrics["positive_rate"]
        dp_ratio = (
            comp_metrics["positive_rate"] / ref_metrics["positive_rate"]
            if ref_metrics["positive_rate"] > 0
            else 0.0
        )

        # Equal opportunity (equal TPR)
        eo_diff = (
            comp_metrics["true_positive_rate"] - ref_metrics["true_positive_rate"]
        )

        # Equalized odds (equal TPR and FPR)
        tpr_diff = (
            comp_metrics["true_positive_rate"] - ref_metrics["true_positive_rate"]
        )
        fpr_diff = (
            comp_metrics["false_positive_rate"] - ref_metrics["false_positive_rate"]
        )
        eq_odds_diff = max(abs(tpr_diff), abs(fpr_diff))

        # Disparate impact ratio
        di_ratio = dp_ratio

        return BiasMetrics(
            protected_attribute=attribute_name,
            reference_group=str(reference_value),
            comparison_group=str(comparison_value),
            demographic_parity_difference=dp_diff,
            demographic_parity_ratio=dp_ratio,
            equal_opportunity_difference=eo_diff,
            equalized_odds_difference=eq_odds_diff,
            disparate_impact_ratio=di_ratio,
            metadata={
                "reference_metrics": ref_metrics,
                "comparison_metrics": comp_metrics,
            },
        )

    def analyze_multiple_groups(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attribute: np.ndarray,
        reference_value: Any,
        attribute_name: str = "protected_attribute",
    ) -> list[BiasMetrics]:
        """Analyze bias across multiple groups.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attribute: Protected attribute values
            reference_value: Reference group value
            attribute_name: Name of protected attribute

        Returns:
            List of bias metrics for each comparison
        """
        unique_values = np.unique(protected_attribute)
        results = []

        for value in unique_values:
            if value == reference_value:
                continue

            metrics = self.detect_bias(
                y_true,
                y_pred,
                protected_attribute,
                reference_value,
                value,
                attribute_name,
            )
            results.append(metrics)

        return results

    def generate_fairness_report(
        self,
        metrics_list: list[BiasMetrics],
        threshold: float = 0.1,
    ) -> dict[str, Any]:
        """Generate comprehensive fairness report.

        Args:
            metrics_list: List of bias metrics
            threshold: Fairness threshold

        Returns:
            Fairness report
        """
        if not metrics_list:
            return {
                "num_comparisons": 0,
                "overall_fair": True,
                "violations": [],
            }

        violations = []
        for metrics in metrics_list:
            # Check each fairness criterion
            if not metrics.is_fair(FairnessMetric.DEMOGRAPHIC_PARITY, threshold):
                violations.append(
                    {
                        "group": metrics.comparison_group,
                        "metric": "demographic_parity",
                        "difference": metrics.demographic_parity_difference,
                    }
                )

            if not metrics.is_fair(FairnessMetric.EQUAL_OPPORTUNITY, threshold):
                violations.append(
                    {
                        "group": metrics.comparison_group,
                        "metric": "equal_opportunity",
                        "difference": metrics.equal_opportunity_difference,
                    }
                )

            if not metrics.is_fair(FairnessMetric.DISPARATE_IMPACT):
                violations.append(
                    {
                        "group": metrics.comparison_group,
                        "metric": "disparate_impact",
                        "ratio": metrics.disparate_impact_ratio,
                    }
                )

        return {
            "num_comparisons": len(metrics_list),
            "overall_fair": len(violations) == 0,
            "num_violations": len(violations),
            "violations": violations,
            "threshold": threshold,
        }


def detect_bias(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attribute: np.ndarray,
    reference_value: Any,
    comparison_value: Any,
) -> BiasMetrics:
    """Detect bias (convenience function).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        protected_attribute: Protected attribute values
        reference_value: Reference group value
        comparison_value: Comparison group value

    Returns:
        Bias metrics
    """
    detector = BiasDetector()
    return detector.detect_bias(
        y_true,
        y_pred,
        protected_attribute,
        reference_value,
        comparison_value,
    )
