#!/usr/bin/env python
"""Data Quality & Drift Detection (Phase 23).

This module provides tools for monitoring data quality and detecting drift:
- Statistical drift detection (KS test, PSI, chi-square)
- Data validation (schema, types, ranges)
- Quality metrics (completeness, validity, consistency)
- Anomaly detection in inputs
- Alert generation for quality issues
"""

from __future__ import annotations

from psy_agents_noaug.data_quality.anomaly_detector import (
    AnomalyDetector,
    AnomalyResult,
    IsolationForestDetector,
    detect_anomalies,
)
from psy_agents_noaug.data_quality.drift_detector import (
    DriftDetector,
    DriftResult,
    DriftTest,
    detect_drift,
)
from psy_agents_noaug.data_quality.quality_metrics import (
    QualityAnalyzer,
    QualityMetrics,
    QualityReport,
    calculate_quality_metrics,
)
from psy_agents_noaug.data_quality.validator import (
    DataValidator,
    ValidationResult,
    ValidationRule,
    validate_data,
)

__all__ = [
    # Drift detection
    "DriftDetector",
    "DriftResult",
    "DriftTest",
    "detect_drift",
    # Data validation
    "DataValidator",
    "ValidationResult",
    "ValidationRule",
    "validate_data",
    # Quality metrics
    "QualityAnalyzer",
    "QualityMetrics",
    "QualityReport",
    "calculate_quality_metrics",
    # Anomaly detection
    "AnomalyDetector",
    "AnomalyResult",
    "IsolationForestDetector",
    "detect_anomalies",
]
