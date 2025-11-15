#!/usr/bin/env python
"""Statistical drift detection (Phase 23).

This module provides tools for detecting distribution drift in data using
statistical tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


class DriftTest(str, Enum):
    """Statistical tests for drift detection."""

    KS_TEST = "ks_test"  # Kolmogorov-Smirnov test
    PSI = "psi"  # Population Stability Index
    CHI_SQUARE = "chi_square"  # Chi-square test
    JENSEN_SHANNON = "jensen_shannon"  # Jensen-Shannon divergence


@dataclass
class DriftResult:
    """Result of drift detection."""

    feature_name: str
    test_name: str
    statistic: float
    p_value: float | None
    is_drift: bool
    threshold: float
    reference_stats: dict[str, float]
    current_stats: dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class DriftDetector:
    """Detector for distribution drift."""

    def __init__(self, significance_level: float = 0.05):
        """Initialize drift detector.

        Args:
            significance_level: Significance level for statistical tests
        """
        self.significance_level = significance_level
        LOGGER.info(f"Initialized DriftDetector (alpha={significance_level})")

    def ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "feature",
    ) -> DriftResult:
        """Kolmogorov-Smirnov test for drift.

        Args:
            reference: Reference distribution
            current: Current distribution
            feature_name: Name of the feature

        Returns:
            Drift detection result
        """
        # Sort both distributions
        ref_sorted = np.sort(reference)
        cur_sorted = np.sort(current)

        # Create combined CDF
        all_values = np.concatenate([ref_sorted, cur_sorted])
        all_values = np.sort(np.unique(all_values))

        # Compute CDFs
        ref_cdf = np.searchsorted(ref_sorted, all_values, side="right") / len(
            ref_sorted
        )
        cur_cdf = np.searchsorted(cur_sorted, all_values, side="right") / len(
            cur_sorted
        )

        # KS statistic (max absolute difference)
        ks_stat = np.max(np.abs(ref_cdf - cur_cdf))

        # Approximate p-value using asymptotic formula
        n1, n2 = len(reference), len(current)
        en = np.sqrt(n1 * n2 / (n1 + n2))
        p_value = 2 * np.exp(-2 * (en * ks_stat) ** 2)

        is_drift = p_value < self.significance_level

        return DriftResult(
            feature_name=feature_name,
            test_name=DriftTest.KS_TEST.value,
            statistic=ks_stat,
            p_value=p_value,
            is_drift=is_drift,
            threshold=self.significance_level,
            reference_stats={
                "mean": float(np.mean(reference)),
                "std": float(np.std(reference)),
                "min": float(np.min(reference)),
                "max": float(np.max(reference)),
            },
            current_stats={
                "mean": float(np.mean(current)),
                "std": float(np.std(current)),
                "min": float(np.min(current)),
                "max": float(np.max(current)),
            },
        )

    def psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "feature",
        n_bins: int = 10,
    ) -> DriftResult:
        """Population Stability Index for drift.

        Args:
            reference: Reference distribution
            current: Current distribution
            feature_name: Name of the feature
            n_bins: Number of bins for discretization

        Returns:
            Drift detection result
        """
        # Create bins based on reference distribution
        _, bin_edges = np.histogram(reference, bins=n_bins)

        # Get counts in each bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions (add small epsilon to avoid division by zero)
        epsilon = 1e-10
        ref_props = (ref_counts + epsilon) / np.sum(ref_counts + epsilon)
        cur_props = (cur_counts + epsilon) / np.sum(cur_counts + epsilon)

        # Calculate PSI
        psi_value = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

        # PSI thresholds: <0.1 no drift, 0.1-0.2 moderate, >0.2 significant
        threshold = 0.1
        is_drift = psi_value > threshold

        return DriftResult(
            feature_name=feature_name,
            test_name=DriftTest.PSI.value,
            statistic=float(psi_value),
            p_value=None,  # PSI doesn't have p-value
            is_drift=is_drift,
            threshold=threshold,
            reference_stats={
                "mean": float(np.mean(reference)),
                "std": float(np.std(reference)),
            },
            current_stats={
                "mean": float(np.mean(current)),
                "std": float(np.std(current)),
            },
        )

    def jensen_shannon_divergence(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "feature",
        n_bins: int = 10,
    ) -> DriftResult:
        """Jensen-Shannon divergence for drift.

        Args:
            reference: Reference distribution
            current: Current distribution
            feature_name: Name of the feature
            n_bins: Number of bins

        Returns:
            Drift detection result
        """
        # Create bins
        _, bin_edges = np.histogram(np.concatenate([reference, current]), bins=n_bins)

        # Get counts
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to probabilities
        epsilon = 1e-10
        p = (ref_counts + epsilon) / np.sum(ref_counts + epsilon)
        q = (cur_counts + epsilon) / np.sum(cur_counts + epsilon)

        # Calculate JS divergence
        m = (p + q) / 2
        js_div = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))

        # Threshold for JS divergence (0-1 scale, sqrt to get JS distance)
        js_distance = np.sqrt(js_div)
        threshold = 0.1
        is_drift = js_distance > threshold

        return DriftResult(
            feature_name=feature_name,
            test_name=DriftTest.JENSEN_SHANNON.value,
            statistic=float(js_distance),
            p_value=None,
            is_drift=is_drift,
            threshold=threshold,
            reference_stats={
                "mean": float(np.mean(reference)),
                "std": float(np.std(reference)),
            },
            current_stats={
                "mean": float(np.mean(current)),
                "std": float(np.std(current)),
            },
        )

    def detect_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "feature",
        test: DriftTest = DriftTest.KS_TEST,
    ) -> DriftResult:
        """Detect drift using specified test.

        Args:
            reference: Reference distribution
            current: Current distribution
            feature_name: Feature name
            test: Statistical test to use

        Returns:
            Drift detection result
        """
        if test == DriftTest.KS_TEST:
            return self.ks_test(reference, current, feature_name)
        if test == DriftTest.PSI:
            return self.psi(reference, current, feature_name)
        if test == DriftTest.JENSEN_SHANNON:
            return self.jensen_shannon_divergence(reference, current, feature_name)
        msg = f"Unsupported test: {test}"
        raise ValueError(msg)


def detect_drift(
    reference: np.ndarray,
    current: np.ndarray,
    feature_name: str = "feature",
    test: DriftTest = DriftTest.KS_TEST,
    significance_level: float = 0.05,
) -> DriftResult:
    """Detect drift (convenience function).

    Args:
        reference: Reference distribution
        current: Current distribution
        feature_name: Feature name
        test: Statistical test
        significance_level: Significance level

    Returns:
        Drift detection result
    """
    detector = DriftDetector(significance_level)
    return detector.detect_drift(reference, current, feature_name, test)
