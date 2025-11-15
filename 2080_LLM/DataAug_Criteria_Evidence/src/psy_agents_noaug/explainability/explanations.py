#!/usr/bin/env python
"""Explanation aggregation and comparison (Phase 27).

This module provides tools for aggregating explanations from multiple
methods and comparing explanations across models or instances.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Single model explanation."""

    instance_id: str
    prediction: Any
    method: str
    feature_attributions: dict[str, float]
    confidence: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_top_features(self, k: int = 5, absolute: bool = True) -> list[tuple[str, float]]:
        """Get top k features by attribution.

        Args:
            k: Number of features to return
            absolute: Use absolute values for ranking

        Returns:
            List of (feature, attribution) tuples
        """
        if absolute:
            sorted_features = sorted(
                self.feature_attributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
        else:
            sorted_features = sorted(
                self.feature_attributions.items(),
                key=lambda x: x[1],
                reverse=True,
            )

        return sorted_features[:k]

    def get_summary(self) -> dict[str, Any]:
        """Get explanation summary.

        Returns:
            Summary dictionary
        """
        attributions = list(self.feature_attributions.values())

        return {
            "instance_id": self.instance_id,
            "prediction": self.prediction,
            "method": self.method,
            "confidence": self.confidence,
            "num_features": len(self.feature_attributions),
            "top_features": self.get_top_features(5),
            "mean_attribution": float(np.mean(attributions)),
            "max_attribution": float(np.max(np.abs(attributions))),
            "timestamp": self.timestamp.isoformat(),
        }


class ExplanationAggregator:
    """Aggregator for multiple explanations."""

    def __init__(self):
        """Initialize explanation aggregator."""
        self.explanations: list[Explanation] = []

        LOGGER.info("Initialized ExplanationAggregator")

    def add_explanation(self, explanation: Explanation) -> None:
        """Add an explanation.

        Args:
            explanation: Explanation to add
        """
        self.explanations.append(explanation)

    def get_by_instance(self, instance_id: str) -> list[Explanation]:
        """Get all explanations for an instance.

        Args:
            instance_id: Instance ID

        Returns:
            List of explanations
        """
        return [e for e in self.explanations if e.instance_id == instance_id]

    def get_by_method(self, method: str) -> list[Explanation]:
        """Get all explanations by method.

        Args:
            method: Explanation method

        Returns:
            List of explanations
        """
        return [e for e in self.explanations if e.method == method]

    def aggregate_feature_importance(
        self,
        method: str | None = None,
        aggregation: str = "mean",
    ) -> dict[str, float]:
        """Aggregate feature importance across explanations.

        Args:
            method: Filter by method (None for all)
            aggregation: Aggregation method ("mean", "median", "max")

        Returns:
            Aggregated feature importance
        """
        # Filter explanations
        explanations = self.explanations if method is None else self.get_by_method(method)

        if not explanations:
            return {}

        # Collect all features
        all_features = set()
        for exp in explanations:
            all_features.update(exp.feature_attributions.keys())

        # Aggregate
        aggregated = {}
        for feature in all_features:
            values = [
                exp.feature_attributions.get(feature, 0.0) for exp in explanations
            ]

            if aggregation == "mean":
                aggregated[feature] = float(np.mean(values))
            elif aggregation == "median":
                aggregated[feature] = float(np.median(values))
            elif aggregation == "max":
                aggregated[feature] = float(np.max(np.abs(values)))
            else:
                msg = f"Unknown aggregation: {aggregation}"
                raise ValueError(msg)

        return aggregated

    def compare_explanations(
        self,
        instance_id: str,
        method1: str,
        method2: str,
    ) -> dict[str, Any]:
        """Compare two explanation methods for an instance.

        Args:
            instance_id: Instance ID
            method1: First method
            method2: Second method

        Returns:
            Comparison results
        """
        # Get explanations
        exp1 = next(
            (e for e in self.explanations if e.instance_id == instance_id and e.method == method1),
            None,
        )
        exp2 = next(
            (e for e in self.explanations if e.instance_id == instance_id and e.method == method2),
            None,
        )

        if exp1 is None or exp2 is None:
            return {"error": "Explanations not found"}

        # Get common features
        common_features = set(exp1.feature_attributions.keys()) & set(
            exp2.feature_attributions.keys()
        )

        if not common_features:
            return {"error": "No common features"}

        # Calculate correlation
        values1 = [exp1.feature_attributions[f] for f in common_features]
        values2 = [exp2.feature_attributions[f] for f in common_features]

        correlation = float(np.corrcoef(values1, values2)[0, 1])

        # Top features agreement
        top1 = set(f for f, _ in exp1.get_top_features(5))
        top2 = set(f for f, _ in exp2.get_top_features(5))

        agreement = len(top1 & top2) / max(len(top1), len(top2))

        return {
            "instance_id": instance_id,
            "method1": method1,
            "method2": method2,
            "correlation": correlation,
            "top_features_agreement": agreement,
            "num_common_features": len(common_features),
        }

    def get_consensus_features(
        self,
        instance_id: str,
        k: int = 5,
        min_agreement: float = 0.5,
    ) -> list[str]:
        """Get consensus important features across methods.

        Args:
            instance_id: Instance ID
            k: Number of top features per method
            min_agreement: Minimum fraction of methods agreeing

        Returns:
            List of consensus features
        """
        # Get all explanations for instance
        explanations = self.get_by_instance(instance_id)

        if not explanations:
            return []

        # Get top features from each method
        all_top_features = []
        for exp in explanations:
            top_features = [f for f, _ in exp.get_top_features(k)]
            all_top_features.extend(top_features)

        # Count occurrences
        from collections import Counter

        feature_counts = Counter(all_top_features)

        # Filter by agreement threshold
        min_count = int(len(explanations) * min_agreement)
        consensus = [
            feature
            for feature, count in feature_counts.most_common()
            if count >= min_count
        ]

        return consensus


def create_explanation(
    instance_id: str,
    prediction: Any,
    method: str,
    feature_names: list[str],
    attributions: np.ndarray,
    **kwargs: Any,
) -> Explanation:
    """Create explanation from attributions (convenience function).

    Args:
        instance_id: Instance identifier
        prediction: Model prediction
        method: Explanation method
        feature_names: Feature names
        attributions: Feature attribution scores
        **kwargs: Additional explanation properties

    Returns:
        Explanation object
    """
    feature_attributions = dict(zip(feature_names, attributions))

    return Explanation(
        instance_id=instance_id,
        prediction=prediction,
        method=method,
        feature_attributions=feature_attributions,
        **kwargs,
    )
