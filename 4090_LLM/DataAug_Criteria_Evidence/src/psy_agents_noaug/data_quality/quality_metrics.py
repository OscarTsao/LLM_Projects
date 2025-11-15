#!/usr/bin/env python
"""Quality metrics calculation (Phase 23).

This module provides tools for calculating data quality metrics including
completeness, validity, consistency, and timeliness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Data quality metrics."""

    completeness: float  # % of non-null values
    validity: float  # % of values passing validation
    uniqueness: float  # % of unique values
    consistency: float  # % of values consistent with expectations
    accuracy: float  # Overall accuracy score
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_overall_score(self) -> float:
        """Calculate overall quality score.

        Returns:
            Overall quality score (0-1)
        """
        return np.mean(
            [
                self.completeness,
                self.validity,
                self.uniqueness,
                self.consistency,
                self.accuracy,
            ]
        )


@dataclass
class QualityReport:
    """Comprehensive quality report."""

    feature_metrics: dict[str, QualityMetrics]
    overall_metrics: QualityMetrics
    num_features: int
    num_samples: int
    timestamp: datetime = field(default_factory=datetime.now)

    def get_summary(self) -> dict[str, Any]:
        """Get quality summary.

        Returns:
            Summary dictionary
        """
        return {
            "overall_score": self.overall_metrics.get_overall_score(),
            "completeness": self.overall_metrics.completeness,
            "validity": self.overall_metrics.validity,
            "uniqueness": self.overall_metrics.uniqueness,
            "consistency": self.overall_metrics.consistency,
            "accuracy": self.overall_metrics.accuracy,
            "num_features": self.num_features,
            "num_samples": self.num_samples,
            "timestamp": self.timestamp.isoformat(),
        }


class QualityAnalyzer:
    """Analyzer for data quality metrics."""

    def __init__(self):
        """Initialize quality analyzer."""
        LOGGER.info("Initialized QualityAnalyzer")

    def calculate_completeness(self, data: np.ndarray) -> float:
        """Calculate completeness (% non-null).

        Args:
            data: Data array

        Returns:
            Completeness score (0-1)
        """
        if len(data) == 0:
            return 1.0

        null_count = 0
        for value in data:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                null_count += 1

        return 1.0 - (null_count / len(data))

    def calculate_validity(
        self,
        data: np.ndarray,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> float:
        """Calculate validity (% within valid range).

        Args:
            data: Data array
            min_value: Minimum valid value
            max_value: Maximum valid value

        Returns:
            Validity score (0-1)
        """
        if len(data) == 0:
            return 1.0

        valid_count = 0

        for value in data:
            try:
                val = float(value)

                is_valid = True
                if min_value is not None and val < min_value:
                    is_valid = False
                if max_value is not None and val > max_value:
                    is_valid = False

                if is_valid:
                    valid_count += 1
            except (ValueError, TypeError):
                # Non-numeric values are invalid
                continue

        return valid_count / len(data)

    def calculate_uniqueness(self, data: np.ndarray) -> float:
        """Calculate uniqueness (% unique values).

        Args:
            data: Data array

        Returns:
            Uniqueness score (0-1)
        """
        if len(data) == 0:
            return 1.0

        unique_values = set()
        for value in data:
            try:
                hashable_value = (
                    value if isinstance(value, (str, int, float)) else str(value)
                )
                unique_values.add(hashable_value)
            except Exception:
                unique_values.add(str(value))

        return len(unique_values) / len(data)

    def calculate_consistency(
        self,
        data: np.ndarray,
        expected_mean: float | None = None,
        expected_std: float | None = None,
        tolerance: float = 0.2,
    ) -> float:
        """Calculate consistency with expected statistics.

        Args:
            data: Data array
            expected_mean: Expected mean
            expected_std: Expected standard deviation
            tolerance: Tolerance for deviation (fraction)

        Returns:
            Consistency score (0-1)
        """
        if len(data) == 0:
            return 1.0

        # Filter numeric values
        numeric_values = []
        for value in data:
            try:
                numeric_values.append(float(value))
            except (ValueError, TypeError):
                continue

        if not numeric_values:
            return 0.0

        actual_mean = np.mean(numeric_values)
        actual_std = np.std(numeric_values)

        score = 1.0

        # Check mean consistency
        if expected_mean is not None:
            mean_diff = abs(actual_mean - expected_mean)
            mean_threshold = max(
                abs(expected_mean) * tolerance, tolerance
            )  # Avoid zero
            if mean_diff > mean_threshold:
                score *= max(0.0, 1.0 - (mean_diff / (mean_threshold * 2)))

        # Check std consistency
        if expected_std is not None:
            std_diff = abs(actual_std - expected_std)
            std_threshold = max(abs(expected_std) * tolerance, tolerance)  # Avoid zero
            if std_diff > std_threshold:
                score *= max(0.0, 1.0 - (std_diff / (std_threshold * 2)))

        return score

    def calculate_metrics(
        self,
        data: np.ndarray,
        feature_name: str = "feature",
        expected_stats: dict[str, float] | None = None,
    ) -> QualityMetrics:
        """Calculate all quality metrics for a feature.

        Args:
            data: Data array
            feature_name: Feature name
            expected_stats: Expected statistics

        Returns:
            Quality metrics
        """
        expected_stats = expected_stats or {}

        completeness = self.calculate_completeness(data)
        validity = self.calculate_validity(
            data,
            expected_stats.get("min"),
            expected_stats.get("max"),
        )
        uniqueness = self.calculate_uniqueness(data)
        consistency = self.calculate_consistency(
            data,
            expected_stats.get("mean"),
            expected_stats.get("std"),
        )

        # Accuracy is overall average
        accuracy = np.mean([completeness, validity, consistency])

        return QualityMetrics(
            completeness=completeness,
            validity=validity,
            uniqueness=uniqueness,
            consistency=consistency,
            accuracy=accuracy,
            metadata={"feature_name": feature_name},
        )

    def generate_report(
        self,
        data: dict[str, np.ndarray],
        expected_stats: dict[str, dict[str, float]] | None = None,
    ) -> QualityReport:
        """Generate comprehensive quality report.

        Args:
            data: Dictionary of feature_name -> data_array
            expected_stats: Expected statistics per feature

        Returns:
            Quality report
        """
        expected_stats = expected_stats or {}

        # Calculate metrics per feature
        feature_metrics = {}
        for feature_name, feature_data in data.items():
            metrics = self.calculate_metrics(
                feature_data,
                feature_name,
                expected_stats.get(feature_name),
            )
            feature_metrics[feature_name] = metrics

        # Calculate overall metrics
        if feature_metrics:
            overall_completeness = np.mean(
                [m.completeness for m in feature_metrics.values()]
            )
            overall_validity = np.mean([m.validity for m in feature_metrics.values()])
            overall_uniqueness = np.mean(
                [m.uniqueness for m in feature_metrics.values()]
            )
            overall_consistency = np.mean(
                [m.consistency for m in feature_metrics.values()]
            )
            overall_accuracy = np.mean([m.accuracy for m in feature_metrics.values()])
        else:
            overall_completeness = 1.0
            overall_validity = 1.0
            overall_uniqueness = 1.0
            overall_consistency = 1.0
            overall_accuracy = 1.0

        overall_metrics = QualityMetrics(
            completeness=overall_completeness,
            validity=overall_validity,
            uniqueness=overall_uniqueness,
            consistency=overall_consistency,
            accuracy=overall_accuracy,
        )

        # Calculate totals
        num_features = len(data)
        num_samples = len(next(iter(data.values()))) if data else 0

        return QualityReport(
            feature_metrics=feature_metrics,
            overall_metrics=overall_metrics,
            num_features=num_features,
            num_samples=num_samples,
        )


def calculate_quality_metrics(
    data: np.ndarray,
    expected_stats: dict[str, float] | None = None,
) -> QualityMetrics:
    """Calculate quality metrics (convenience function).

    Args:
        data: Data array
        expected_stats: Expected statistics

    Returns:
        Quality metrics
    """
    analyzer = QualityAnalyzer()
    return analyzer.calculate_metrics(data, expected_stats=expected_stats)
