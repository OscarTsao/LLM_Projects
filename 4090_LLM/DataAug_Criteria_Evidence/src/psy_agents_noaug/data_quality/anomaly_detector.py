#!/usr/bin/env python
"""Anomaly detection (Phase 23).

This module provides tools for detecting anomalies and outliers in data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""

    feature_name: str
    num_anomalies: int
    total_samples: int
    anomaly_rate: float
    anomaly_indices: list[int]
    anomaly_scores: list[float]
    method: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class AnomalyDetector:
    """Base class for anomaly detectors."""

    def __init__(self):
        """Initialize anomaly detector."""
        LOGGER.info(f"Initialized {self.__class__.__name__}")

    def detect(
        self,
        data: np.ndarray,
        feature_name: str = "feature",
    ) -> AnomalyResult:
        """Detect anomalies in data.

        Args:
            data: Data array
            feature_name: Feature name

        Returns:
            Anomaly detection result
        """
        raise NotImplementedError


class IQRDetector(AnomalyDetector):
    """Interquartile Range (IQR) anomaly detector."""

    def __init__(self, threshold: float = 1.5):
        """Initialize IQR detector.

        Args:
            threshold: IQR multiplier for outlier detection
        """
        super().__init__()
        self.threshold = threshold

    def detect(
        self,
        data: np.ndarray,
        feature_name: str = "feature",
    ) -> AnomalyResult:
        """Detect anomalies using IQR method.

        Args:
            data: Data array
            feature_name: Feature name

        Returns:
            Anomaly detection result
        """
        # Filter numeric values
        numeric_values = []
        numeric_indices = []

        for i, value in enumerate(data):
            try:
                numeric_values.append(float(value))
                numeric_indices.append(i)
            except (ValueError, TypeError):
                continue

        if not numeric_values:
            return AnomalyResult(
                feature_name=feature_name,
                num_anomalies=0,
                total_samples=len(data),
                anomaly_rate=0.0,
                anomaly_indices=[],
                anomaly_scores=[],
                method="IQR",
            )

        # Calculate IQR
        q1 = np.percentile(numeric_values, 25)
        q3 = np.percentile(numeric_values, 75)
        iqr = q3 - q1

        # Define bounds
        lower_bound = q1 - self.threshold * iqr
        upper_bound = q3 + self.threshold * iqr

        # Find anomalies
        anomaly_indices = []
        anomaly_scores = []

        for i, value in zip(numeric_indices, numeric_values):
            if value < lower_bound or value > upper_bound:
                anomaly_indices.append(i)

                # Score is distance from nearest bound
                if value < lower_bound:
                    score = (lower_bound - value) / (iqr if iqr > 0 else 1.0)
                else:
                    score = (value - upper_bound) / (iqr if iqr > 0 else 1.0)

                anomaly_scores.append(float(score))

        return AnomalyResult(
            feature_name=feature_name,
            num_anomalies=len(anomaly_indices),
            total_samples=len(data),
            anomaly_rate=len(anomaly_indices) / len(data),
            anomaly_indices=anomaly_indices,
            anomaly_scores=anomaly_scores,
            method="IQR",
            metadata={
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
        )


class ZScoreDetector(AnomalyDetector):
    """Z-score based anomaly detector."""

    def __init__(self, threshold: float = 3.0):
        """Initialize Z-score detector.

        Args:
            threshold: Z-score threshold for outliers
        """
        super().__init__()
        self.threshold = threshold

    def detect(
        self,
        data: np.ndarray,
        feature_name: str = "feature",
    ) -> AnomalyResult:
        """Detect anomalies using Z-score method.

        Args:
            data: Data array
            feature_name: Feature name

        Returns:
            Anomaly detection result
        """
        # Filter numeric values
        numeric_values = []
        numeric_indices = []

        for i, value in enumerate(data):
            try:
                numeric_values.append(float(value))
                numeric_indices.append(i)
            except (ValueError, TypeError):
                continue

        if not numeric_values:
            return AnomalyResult(
                feature_name=feature_name,
                num_anomalies=0,
                total_samples=len(data),
                anomaly_rate=0.0,
                anomaly_indices=[],
                anomaly_scores=[],
                method="Z-Score",
            )

        # Calculate mean and std
        mean = np.mean(numeric_values)
        std = np.std(numeric_values)

        if std == 0:
            # No variance, no outliers
            return AnomalyResult(
                feature_name=feature_name,
                num_anomalies=0,
                total_samples=len(data),
                anomaly_rate=0.0,
                anomaly_indices=[],
                anomaly_scores=[],
                method="Z-Score",
                metadata={"mean": mean, "std": std},
            )

        # Calculate Z-scores
        anomaly_indices = []
        anomaly_scores = []

        for i, value in zip(numeric_indices, numeric_values):
            z_score = abs((value - mean) / std)

            if z_score > self.threshold:
                anomaly_indices.append(i)
                anomaly_scores.append(float(z_score))

        return AnomalyResult(
            feature_name=feature_name,
            num_anomalies=len(anomaly_indices),
            total_samples=len(data),
            anomaly_rate=len(anomaly_indices) / len(data),
            anomaly_indices=anomaly_indices,
            anomaly_scores=anomaly_scores,
            method="Z-Score",
            metadata={"mean": mean, "std": std, "threshold": self.threshold},
        )


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest anomaly detector (simplified version)."""

    def __init__(self, contamination: float = 0.1, n_trees: int = 10):
        """Initialize Isolation Forest detector.

        Args:
            contamination: Expected proportion of outliers
            n_trees: Number of trees (simplified, not used in basic version)
        """
        super().__init__()
        self.contamination = contamination
        self.n_trees = n_trees

    def detect(
        self,
        data: np.ndarray,
        feature_name: str = "feature",
    ) -> AnomalyResult:
        """Detect anomalies using simplified isolation approach.

        Args:
            data: Data array
            feature_name: Feature name

        Returns:
            Anomaly detection result
        """
        # Filter numeric values
        numeric_values = []
        numeric_indices = []

        for i, value in enumerate(data):
            try:
                numeric_values.append(float(value))
                numeric_indices.append(i)
            except (ValueError, TypeError):
                continue

        if not numeric_values:
            return AnomalyResult(
                feature_name=feature_name,
                num_anomalies=0,
                total_samples=len(data),
                anomaly_rate=0.0,
                anomaly_indices=[],
                anomaly_scores=[],
                method="IsolationForest",
            )

        # Simple isolation: use distance from median
        # (Simplified version - real isolation forest is more complex)
        median = np.median(numeric_values)
        mad = np.median([abs(v - median) for v in numeric_values])

        # Score based on distance from median
        scores = []
        for value in numeric_values:
            if mad > 0:
                score = abs(value - median) / mad
            else:
                score = 0.0
            scores.append(score)

        # Select top outliers based on contamination
        threshold_idx = int((1 - self.contamination) * len(scores))
        sorted_indices = np.argsort(scores)
        outlier_local_indices = sorted_indices[threshold_idx:]

        anomaly_indices = [numeric_indices[i] for i in outlier_local_indices]
        anomaly_scores = [scores[i] for i in outlier_local_indices]

        return AnomalyResult(
            feature_name=feature_name,
            num_anomalies=len(anomaly_indices),
            total_samples=len(data),
            anomaly_rate=len(anomaly_indices) / len(data),
            anomaly_indices=anomaly_indices,
            anomaly_scores=anomaly_scores,
            method="IsolationForest",
            metadata={
                "median": median,
                "mad": mad,
                "contamination": self.contamination,
            },
        )


def detect_anomalies(
    data: np.ndarray,
    method: str = "iqr",
    feature_name: str = "feature",
    **kwargs,
) -> AnomalyResult:
    """Detect anomalies (convenience function).

    Args:
        data: Data array
        method: Detection method (iqr, zscore, isolation_forest)
        feature_name: Feature name
        **kwargs: Additional arguments for detector

    Returns:
        Anomaly detection result
    """
    if method == "iqr":
        detector = IQRDetector(**kwargs)
    elif method == "zscore":
        detector = ZScoreDetector(**kwargs)
    elif method == "isolation_forest":
        detector = IsolationForestDetector(**kwargs)
    else:
        msg = f"Unknown method: {method}"
        raise ValueError(msg)

    return detector.detect(data, feature_name)
