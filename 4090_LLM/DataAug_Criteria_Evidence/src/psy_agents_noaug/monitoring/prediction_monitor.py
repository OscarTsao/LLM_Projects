#!/usr/bin/env python
"""Prediction monitoring and drift detection (Phase 26).

This module provides tools for monitoring prediction distributions,
detecting data drift, and tracking prediction quality over time.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from scipy import stats

LOGGER = logging.getLogger(__name__)


@dataclass
class DriftMetrics:
    """Data drift metrics."""

    # Statistical tests
    ks_statistic: float  # Kolmogorov-Smirnov test
    ks_pvalue: float
    js_divergence: float  # Jensen-Shannon divergence

    # Distribution metrics
    mean_shift: float
    std_shift: float

    # Drift detection
    is_drift_detected: bool
    drift_severity: str  # "none", "low", "medium", "high"

    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> dict[str, Any]:
        """Get drift metrics summary.

        Returns:
            Summary dictionary
        """
        return {
            "is_drift_detected": self.is_drift_detected,
            "drift_severity": self.drift_severity,
            "statistics": {
                "ks_statistic": self.ks_statistic,
                "ks_pvalue": self.ks_pvalue,
                "js_divergence": self.js_divergence,
            },
            "distribution_shifts": {
                "mean_shift": self.mean_shift,
                "std_shift": self.std_shift,
            },
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PredictionStats:
    """Prediction statistics."""

    # Distribution
    mean: float
    std: float
    min: float
    max: float

    # Class distribution (for classification)
    class_distribution: dict[int, float] = field(default_factory=dict)

    # Confidence metrics
    mean_confidence: float = 0.0
    low_confidence_rate: float = 0.0  # Rate of predictions with confidence < threshold

    # Sample size
    num_predictions: int = 0

    timestamp: datetime = field(default_factory=datetime.now)

    def get_summary(self) -> dict[str, Any]:
        """Get prediction statistics summary.

        Returns:
            Summary dictionary
        """
        return {
            "distribution": {
                "mean": self.mean,
                "std": self.std,
                "min": self.min,
                "max": self.max,
            },
            "class_distribution": self.class_distribution,
            "confidence": {
                "mean": self.mean_confidence,
                "low_rate": self.low_confidence_rate,
            },
            "num_predictions": self.num_predictions,
            "timestamp": self.timestamp.isoformat(),
        }


class PredictionMonitor:
    """Monitor prediction distributions and detect drift."""

    def __init__(
        self,
        reference_window_size: int = 1000,
        monitoring_window_size: int = 100,
        drift_threshold: float = 0.05,  # p-value threshold
    ):
        """Initialize prediction monitor.

        Args:
            reference_window_size: Size of reference distribution window
            monitoring_window_size: Size of monitoring window for drift detection
            drift_threshold: p-value threshold for drift detection
        """
        self.reference_window_size = reference_window_size
        self.monitoring_window_size = monitoring_window_size
        self.drift_threshold = drift_threshold

        # Reference distribution (baseline)
        self.reference_predictions: deque[float] = deque(maxlen=reference_window_size)
        self.reference_labels: deque[int] = deque(maxlen=reference_window_size)

        # Monitoring window (current)
        self.monitoring_predictions: deque[float] = deque(maxlen=monitoring_window_size)
        self.monitoring_labels: deque[int] = deque(maxlen=monitoring_window_size)

        # Stats history
        self.drift_history: list[DriftMetrics] = []

        LOGGER.info(
            f"Initialized PredictionMonitor "
            f"(ref_window={reference_window_size}, "
            f"mon_window={monitoring_window_size}, "
            f"threshold={drift_threshold})"
        )

    def record_prediction(
        self,
        prediction: float,
        label: int | None = None,
        confidence: float | None = None,
    ) -> None:
        """Record a prediction.

        Args:
            prediction: Predicted value/probability
            label: Predicted label (for classification)
            confidence: Prediction confidence
        """
        # Add to reference if not full, otherwise to monitoring
        if len(self.reference_predictions) < self.reference_window_size:
            self.reference_predictions.append(prediction)
            if label is not None:
                self.reference_labels.append(label)
        else:
            self.monitoring_predictions.append(prediction)
            if label is not None:
                self.monitoring_labels.append(label)

    def get_reference_stats(self) -> PredictionStats | None:
        """Get statistics for reference distribution.

        Returns:
            Prediction statistics or None if insufficient data
        """
        if len(self.reference_predictions) < 10:
            return None

        preds = np.array(self.reference_predictions)

        # Class distribution
        class_dist = {}
        if self.reference_labels:
            label_counts = Counter(self.reference_labels)
            total = len(self.reference_labels)
            class_dist = {k: v / total for k, v in label_counts.items()}

        return PredictionStats(
            mean=float(np.mean(preds)),
            std=float(np.std(preds)),
            min=float(np.min(preds)),
            max=float(np.max(preds)),
            class_distribution=class_dist,
            num_predictions=len(preds),
        )

    def get_monitoring_stats(self) -> PredictionStats | None:
        """Get statistics for monitoring window.

        Returns:
            Prediction statistics or None if insufficient data
        """
        if len(self.monitoring_predictions) < 10:
            return None

        preds = np.array(self.monitoring_predictions)

        # Class distribution
        class_dist = {}
        if self.monitoring_labels:
            label_counts = Counter(self.monitoring_labels)
            total = len(self.monitoring_labels)
            class_dist = {k: v / total for k, v in label_counts.items()}

        return PredictionStats(
            mean=float(np.mean(preds)),
            std=float(np.std(preds)),
            min=float(np.min(preds)),
            max=float(np.max(preds)),
            class_distribution=class_dist,
            num_predictions=len(preds),
        )

    def detect_drift(self) -> DriftMetrics | None:
        """Detect drift between reference and monitoring distributions.

        Returns:
            Drift metrics or None if insufficient data
        """
        if (
            len(self.reference_predictions) < self.reference_window_size // 2
            or len(self.monitoring_predictions) < self.monitoring_window_size // 2
        ):
            LOGGER.debug("Insufficient data for drift detection")
            return None

        ref_data = np.array(self.reference_predictions)
        mon_data = np.array(self.monitoring_predictions)

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(ref_data, mon_data)

        # Jensen-Shannon divergence
        js_div = self._calculate_js_divergence(ref_data, mon_data)

        # Distribution shifts
        mean_shift = abs(np.mean(mon_data) - np.mean(ref_data))
        std_shift = abs(np.std(mon_data) - np.std(ref_data))

        # Determine drift
        is_drift = ks_pvalue < self.drift_threshold

        # Severity based on KS statistic
        if ks_stat < 0.1:
            severity = "none"
        elif ks_stat < 0.2:
            severity = "low"
        elif ks_stat < 0.4:
            severity = "medium"
        else:
            severity = "high"

        metrics = DriftMetrics(
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_pvalue),
            js_divergence=float(js_div),
            mean_shift=float(mean_shift),
            std_shift=float(std_shift),
            is_drift_detected=is_drift,
            drift_severity=severity,
        )

        self.drift_history.append(metrics)

        if is_drift:
            LOGGER.warning(
                f"Drift detected! KS={ks_stat:.3f}, p={ks_pvalue:.3f}, "
                f"severity={severity}"
            )
        else:
            LOGGER.debug(f"No drift detected (KS={ks_stat:.3f}, p={ks_pvalue:.3f})")

        return metrics

    def _calculate_js_divergence(
        self,
        dist1: np.ndarray,
        dist2: np.ndarray,
    ) -> float:
        """Calculate Jensen-Shannon divergence between distributions.

        Args:
            dist1: First distribution
            dist2: Second distribution

        Returns:
            JS divergence value
        """
        # Create histograms
        bins = 50
        range_min = min(dist1.min(), dist2.min())
        range_max = max(dist1.max(), dist2.max())

        hist1, _ = np.histogram(dist1, bins=bins, range=(range_min, range_max))
        hist2, _ = np.histogram(dist2, bins=bins, range=(range_min, range_max))

        # Normalize to probabilities
        p = hist1 / hist1.sum()
        q = hist2 / hist2.sum()

        # Avoid division by zero
        p = np.where(p == 0, 1e-10, p)
        q = np.where(q == 0, 1e-10, q)

        # Calculate JS divergence
        m = 0.5 * (p + q)
        js_div = 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))

        return float(js_div)

    def get_drift_history(
        self,
        since: datetime | None = None,
    ) -> list[DriftMetrics]:
        """Get drift detection history.

        Args:
            since: Get history since this time

        Returns:
            List of drift metrics
        """
        if since is None:
            return self.drift_history

        return [m for m in self.drift_history if m.timestamp >= since]

    def reset_monitoring_window(self) -> None:
        """Reset monitoring window (keeps reference)."""
        self.monitoring_predictions.clear()
        self.monitoring_labels.clear()

        LOGGER.info("Reset monitoring window")

    def update_reference_distribution(self) -> None:
        """Update reference distribution from current monitoring window."""
        if len(self.monitoring_predictions) < self.monitoring_window_size // 2:
            LOGGER.warning("Insufficient monitoring data to update reference")
            return

        # Move monitoring data to reference
        for pred in self.monitoring_predictions:
            self.reference_predictions.append(pred)

        for label in self.monitoring_labels:
            self.reference_labels.append(label)

        # Clear monitoring window
        self.monitoring_predictions.clear()
        self.monitoring_labels.clear()

        LOGGER.info("Updated reference distribution from monitoring window")

    def get_summary(self) -> dict[str, Any]:
        """Get monitoring summary.

        Returns:
            Summary dictionary
        """
        ref_stats = self.get_reference_stats()
        mon_stats = self.get_monitoring_stats()

        # Recent drift
        recent_drift = None
        if self.drift_history:
            recent_drift = self.drift_history[-1].get_summary()

        return {
            "reference": ref_stats.get_summary() if ref_stats else None,
            "monitoring": mon_stats.get_summary() if mon_stats else None,
            "recent_drift": recent_drift,
            "drift_detections": len(
                [m for m in self.drift_history if m.is_drift_detected]
            ),
            "total_checks": len(self.drift_history),
        }


def calculate_prediction_entropy(probabilities: np.ndarray) -> float:
    """Calculate entropy of prediction probabilities.

    Args:
        probabilities: Array of probability distributions

    Returns:
        Mean entropy across predictions

    Example:
        >>> probs = np.array([[0.9, 0.1], [0.6, 0.4]])
        >>> entropy = calculate_prediction_entropy(probs)
    """
    # Avoid log(0)
    probs = np.clip(probabilities, 1e-10, 1.0)

    # Calculate entropy for each prediction
    entropies = -np.sum(probs * np.log(probs), axis=1)

    return float(np.mean(entropies))
