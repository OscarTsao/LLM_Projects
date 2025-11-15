#!/usr/bin/env python
"""Drift detection for model monitoring (Phase 17).

This module provides drift detection including:
- Data drift detection (input distribution changes)
- Prediction drift detection (output distribution changes)
- Statistical tests (KS test, PSI, etc.)
- Drift scoring and alerting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats

LOGGER = logging.getLogger(__name__)


class DriftStatus(Enum):
    """Status of drift detection."""

    NO_DRIFT = "no_drift"
    WARNING = "warning"
    DRIFT_DETECTED = "drift_detected"


@dataclass
class DriftResult:
    """Result of drift detection."""

    status: DriftStatus
    drift_score: float
    p_value: float
    threshold: float
    timestamp: datetime
    message: str = ""


class DriftDetector:
    """Detect data and prediction drift."""

    def __init__(
        self,
        warning_threshold: float = 0.05,
        drift_threshold: float = 0.01,
    ):
        """Initialize drift detector.

        Args:
            warning_threshold: P-value threshold for warnings
            drift_threshold: P-value threshold for drift detection
        """
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold

        # Baseline distributions
        self.baseline_data: np.ndarray | None = None
        self.baseline_predictions: np.ndarray | None = None

        LOGGER.info(
            "Initialized DriftDetector (warning=%f, drift=%f)",
            warning_threshold,
            drift_threshold,
        )

    def set_baseline(
        self,
        data: np.ndarray | None = None,
        predictions: np.ndarray | None = None,
    ) -> None:
        """Set baseline distributions.

        Args:
            data: Baseline input data
            predictions: Baseline predictions
        """
        if data is not None:
            self.baseline_data = np.array(data)
            LOGGER.info("Set baseline data (n=%d)", len(data))

        if predictions is not None:
            self.baseline_predictions = np.array(predictions)
            LOGGER.info("Set baseline predictions (n=%d)", len(predictions))

    def _kolmogorov_smirnov_test(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
    ) -> tuple[float, float]:
        """Perform Kolmogorov-Smirnov test.

        Args:
            baseline: Baseline distribution
            current: Current distribution

        Returns:
            Tuple of (statistic, p_value)
        """
        # For multivariate data, flatten or use first dimension
        if baseline.ndim > 1:
            baseline = baseline.flatten()
        if current.ndim > 1:
            current = current.flatten()

        statistic, p_value = stats.ks_2samp(baseline, current)
        return float(statistic), float(p_value)

    def _population_stability_index(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        bins: int = 10,
    ) -> float:
        """Calculate Population Stability Index (PSI).

        Args:
            baseline: Baseline distribution
            current: Current distribution
            bins: Number of bins for histogram

        Returns:
            PSI score
        """
        # Flatten if multivariate
        if baseline.ndim > 1:
            baseline = baseline.flatten()
        if current.ndim > 1:
            current = current.flatten()

        # Create bins based on baseline
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)

        # Calculate distributions
        baseline_dist, _ = np.histogram(baseline, bins=bin_edges)
        current_dist, _ = np.histogram(current, bins=bin_edges)

        # Normalize to probabilities
        baseline_dist = baseline_dist / baseline_dist.sum()
        current_dist = current_dist / current_dist.sum()

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        baseline_dist = baseline_dist + epsilon
        current_dist = current_dist + epsilon

        # Calculate PSI
        psi = np.sum((current_dist - baseline_dist) * np.log(current_dist / baseline_dist))

        return float(psi)

    def detect_data_drift(
        self,
        current_data: np.ndarray,
        method: str = "ks",
    ) -> DriftResult:
        """Detect drift in input data.

        Args:
            current_data: Current data to compare against baseline
            method: Detection method ('ks' or 'psi')

        Returns:
            Drift detection result
        """
        if self.baseline_data is None:
            raise ValueError("Baseline data not set. Call set_baseline() first.")

        current_data = np.array(current_data)

        if method == "ks":
            # Kolmogorov-Smirnov test
            statistic, p_value = self._kolmogorov_smirnov_test(
                self.baseline_data,
                current_data,
            )
            drift_score = statistic

        elif method == "psi":
            # Population Stability Index
            psi = self._population_stability_index(
                self.baseline_data,
                current_data,
            )
            drift_score = psi
            # Convert PSI to pseudo p-value for consistency
            # PSI > 0.2 is significant drift
            p_value = max(0.0, 1.0 - (psi / 0.2))

        else:
            raise ValueError(f"Unknown method: {method}")

        # Determine status
        if p_value >= self.warning_threshold:
            status = DriftStatus.NO_DRIFT
            message = "No significant drift detected"
        elif p_value >= self.drift_threshold:
            status = DriftStatus.WARNING
            message = "Warning: Potential drift detected"
        else:
            status = DriftStatus.DRIFT_DETECTED
            message = "Alert: Significant drift detected!"

        return DriftResult(
            status=status,
            drift_score=drift_score,
            p_value=p_value,
            threshold=self.drift_threshold,
            timestamp=datetime.now(),
            message=message,
        )

    def detect_prediction_drift(
        self,
        current_predictions: np.ndarray,
        method: str = "ks",
    ) -> DriftResult:
        """Detect drift in model predictions.

        Args:
            current_predictions: Current predictions to compare
            method: Detection method ('ks' or 'psi')

        Returns:
            Drift detection result
        """
        if self.baseline_predictions is None:
            raise ValueError(
                "Baseline predictions not set. Call set_baseline() first."
            )

        current_predictions = np.array(current_predictions)

        if method == "ks":
            # Kolmogorov-Smirnov test
            statistic, p_value = self._kolmogorov_smirnov_test(
                self.baseline_predictions,
                current_predictions,
            )
            drift_score = statistic

        elif method == "psi":
            # Population Stability Index
            psi = self._population_stability_index(
                self.baseline_predictions,
                current_predictions,
            )
            drift_score = psi
            p_value = max(0.0, 1.0 - (psi / 0.2))

        else:
            raise ValueError(f"Unknown method: {method}")

        # Determine status
        if p_value >= self.warning_threshold:
            status = DriftStatus.NO_DRIFT
            message = "No significant prediction drift"
        elif p_value >= self.drift_threshold:
            status = DriftStatus.WARNING
            message = "Warning: Potential prediction drift"
        else:
            status = DriftStatus.DRIFT_DETECTED
            message = "Alert: Significant prediction drift!"

        return DriftResult(
            status=status,
            drift_score=drift_score,
            p_value=p_value,
            threshold=self.drift_threshold,
            timestamp=datetime.now(),
            message=message,
        )

    def get_summary(
        self,
        current_data: np.ndarray | None = None,
        current_predictions: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Get drift detection summary.

        Args:
            current_data: Current data
            current_predictions: Current predictions

        Returns:
            Summary dict
        """
        summary: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "warning_threshold": self.warning_threshold,
            "drift_threshold": self.drift_threshold,
        }

        # Data drift
        if current_data is not None and self.baseline_data is not None:
            data_drift = self.detect_data_drift(current_data)
            summary["data_drift"] = {
                "status": data_drift.status.value,
                "drift_score": data_drift.drift_score,
                "p_value": data_drift.p_value,
                "message": data_drift.message,
            }

        # Prediction drift
        if current_predictions is not None and self.baseline_predictions is not None:
            pred_drift = self.detect_prediction_drift(current_predictions)
            summary["prediction_drift"] = {
                "status": pred_drift.status.value,
                "drift_score": pred_drift.drift_score,
                "p_value": pred_drift.p_value,
                "message": pred_drift.message,
            }

        return summary


def detect_drift(
    baseline: np.ndarray,
    current: np.ndarray,
    method: str = "ks",
    drift_threshold: float = 0.01,
) -> DriftResult:
    """Detect drift (convenience function).

    Args:
        baseline: Baseline distribution
        current: Current distribution
        method: Detection method
        drift_threshold: P-value threshold

    Returns:
        Drift detection result
    """
    detector = DriftDetector(drift_threshold=drift_threshold)
    detector.set_baseline(data=baseline)
    return detector.detect_data_drift(current, method=method)
