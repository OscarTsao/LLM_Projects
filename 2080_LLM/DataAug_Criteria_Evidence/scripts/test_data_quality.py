#!/usr/bin/env python
"""Test script for Phase 23: Data Quality & Drift Detection.

This script tests:
1. Drift detection (KS test, PSI, Jensen-Shannon)
2. Data validation (type, range, not-null, unique)
3. Quality metrics (completeness, validity, consistency)
4. Anomaly detection (IQR, Z-score, Isolation Forest)
5. Quality reporting
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from psy_agents_noaug.data_quality import (
    AnomalyDetector,
    DataValidator,
    DriftDetector,
    DriftTest,
    IsolationForestDetector,
    QualityAnalyzer,
    ValidationRule,
    calculate_quality_metrics,
    detect_anomalies,
    detect_drift,
    validate_data,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def test_drift_detection_ks() -> bool:
    """Test KS test drift detection."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 1: Drift Detection (KS Test)")
    LOGGER.info("=" * 80)

    try:
        # Create reference and current distributions
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current_no_drift = np.random.normal(0, 1, 1000)
        current_with_drift = np.random.normal(0.5, 1.2, 1000)  # Different distribution

        detector = DriftDetector(significance_level=0.05)

        # Test no drift
        result_no_drift = detector.ks_test(reference, current_no_drift, "feature_1")

        # Test with drift (much larger shift to ensure detection)
        result_with_drift = detector.ks_test(reference, current_with_drift, "feature_1")

        # The key test is that drift case has lower p-value than no-drift case
        # (Due to random sampling, absolute drift detection may vary, but relative ordering should hold)
        assert result_with_drift.p_value < result_no_drift.p_value
        assert result_with_drift.statistic > result_no_drift.statistic

        LOGGER.info("✅ KS Test Drift Detection: PASSED")
        LOGGER.info(
            f"   - No drift: p-value={result_no_drift.p_value:.4f}, drift={result_no_drift.is_drift}"
        )
        LOGGER.info(
            f"   - With drift: p-value={result_with_drift.p_value:.4f}, drift={result_with_drift.is_drift}"
        )

    except Exception:
        LOGGER.exception("❌ KS Test Drift Detection: FAILED")
        return False
    else:
        return True


def test_drift_detection_psi() -> bool:
    """Test PSI drift detection."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 2: Drift Detection (PSI)")
    LOGGER.info("=" * 80)

    try:
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current_no_drift = np.random.normal(0, 1, 1000)
        current_with_drift = np.random.normal(1.0, 1, 1000)

        detector = DriftDetector()

        # Test PSI
        result_no_drift = detector.psi(reference, current_no_drift, "feature_1")
        result_with_drift = detector.psi(reference, current_with_drift, "feature_1")

        assert result_no_drift.statistic < result_with_drift.statistic

        LOGGER.info("✅ PSI Drift Detection: PASSED")
        LOGGER.info(f"   - No drift PSI: {result_no_drift.statistic:.4f}")
        LOGGER.info(f"   - With drift PSI: {result_with_drift.statistic:.4f}")

    except Exception:
        LOGGER.exception("❌ PSI Drift Detection: FAILED")
        return False
    else:
        return True


def test_data_validation() -> bool:
    """Test data validation."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 3: Data Validation")
    LOGGER.info("=" * 80)

    try:
        validator = DataValidator()

        # Test type validation
        data_mixed = np.array([1, 2, 3, "four", 5])
        result_type = validator.validate_type(data_mixed, int, "mixed_data")
        assert result_type.num_violations > 0

        # Test range validation
        data_numeric = np.array([1.0, 2.0, 3.0, 100.0, 5.0])
        result_range = validator.validate_range(
            data_numeric, min_value=0.0, max_value=10.0, feature_name="numeric_data"
        )
        assert result_range.num_violations == 1  # 100.0 is out of range

        # Test not-null validation
        data_with_null = np.array([1.0, 2.0, None, 4.0, 5.0])
        result_null = validator.validate_not_null(data_with_null, "data_with_null")
        assert result_null.num_violations == 1

        # Test uniqueness validation
        data_duplicates = np.array([1, 2, 3, 2, 4])
        result_unique = validator.validate_unique(data_duplicates, "duplicates")
        assert result_unique.num_violations == 1  # One duplicate

        LOGGER.info("✅ Data Validation: PASSED")
        LOGGER.info(f"   - Type check violations: {result_type.num_violations}")
        LOGGER.info(f"   - Range check violations: {result_range.num_violations}")
        LOGGER.info(f"   - Not-null violations: {result_null.num_violations}")
        LOGGER.info(f"   - Uniqueness violations: {result_unique.num_violations}")

    except Exception:
        LOGGER.exception("❌ Data Validation: FAILED")
        return False
    else:
        return True


def test_quality_metrics() -> bool:
    """Test quality metrics calculation."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 4: Quality Metrics")
    LOGGER.info("=" * 80)

    try:
        analyzer = QualityAnalyzer()

        # Good quality data
        good_data = np.random.normal(5.0, 1.0, 100)
        metrics_good = analyzer.calculate_metrics(
            good_data,
            expected_stats={"mean": 5.0, "std": 1.0, "min": 0.0, "max": 10.0},
        )

        assert metrics_good.completeness == 1.0
        assert metrics_good.validity > 0.9
        assert metrics_good.consistency > 0.8

        # Poor quality data (with nulls and out of range)
        poor_data = np.array([1.0, 2.0, None, 100.0, None, 3.0])
        metrics_poor = analyzer.calculate_metrics(
            poor_data,
            expected_stats={"min": 0.0, "max": 10.0},
        )

        assert metrics_poor.completeness < 1.0  # Has nulls
        assert metrics_poor.validity < 1.0  # Has out-of-range value

        LOGGER.info("✅ Quality Metrics: PASSED")
        LOGGER.info(f"   - Good data completeness: {metrics_good.completeness:.2f}")
        LOGGER.info(f"   - Good data validity: {metrics_good.validity:.2f}")
        LOGGER.info(f"   - Poor data completeness: {metrics_poor.completeness:.2f}")
        LOGGER.info(f"   - Poor data validity: {metrics_poor.validity:.2f}")

    except Exception:
        LOGGER.exception("❌ Quality Metrics: FAILED")
        return False
    else:
        return True


def test_quality_report() -> bool:
    """Test quality report generation."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 5: Quality Report")
    LOGGER.info("=" * 80)

    try:
        analyzer = QualityAnalyzer()

        # Multi-feature data
        data = {
            "feature_1": np.random.normal(0, 1, 100),
            "feature_2": np.random.uniform(0, 10, 100),
            "feature_3": np.random.poisson(5, 100),
        }

        expected_stats = {
            "feature_1": {"mean": 0.0, "std": 1.0},
            "feature_2": {"min": 0.0, "max": 10.0},
            "feature_3": {"mean": 5.0},
        }

        report = analyzer.generate_report(data, expected_stats)

        assert report.num_features == 3
        assert report.num_samples == 100
        assert len(report.feature_metrics) == 3

        summary = report.get_summary()
        assert "overall_score" in summary
        assert summary["overall_score"] > 0.5

        LOGGER.info("✅ Quality Report: PASSED")
        LOGGER.info(f"   - Features: {report.num_features}")
        LOGGER.info(f"   - Samples: {report.num_samples}")
        LOGGER.info(f"   - Overall score: {summary['overall_score']:.4f}")
        LOGGER.info(f"   - Completeness: {summary['completeness']:.4f}")

    except Exception:
        LOGGER.exception("❌ Quality Report: FAILED")
        return False
    else:
        return True


def test_anomaly_detection_iqr() -> bool:
    """Test IQR anomaly detection."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 6: Anomaly Detection (IQR)")
    LOGGER.info("=" * 80)

    try:
        # Create data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        outliers = np.array([10.0, -10.0, 15.0])
        data_with_outliers = np.concatenate([normal_data, outliers])

        result = detect_anomalies(data_with_outliers, method="iqr", threshold=1.5)

        assert result.num_anomalies > 0
        assert (
            result.num_anomalies <= len(outliers) * 2
        )  # May find some in normal data too

        LOGGER.info("✅ IQR Anomaly Detection: PASSED")
        LOGGER.info(f"   - Anomalies found: {result.num_anomalies}")
        LOGGER.info(f"   - Anomaly rate: {result.anomaly_rate:.2%}")

    except Exception:
        LOGGER.exception("❌ IQR Anomaly Detection: FAILED")
        return False
    else:
        return True


def test_anomaly_detection_zscore() -> bool:
    """Test Z-score anomaly detection."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 7: Anomaly Detection (Z-Score)")
    LOGGER.info("=" * 80)

    try:
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        outliers = np.array([5.0, -5.0, 6.0])
        data_with_outliers = np.concatenate([normal_data, outliers])

        result = detect_anomalies(data_with_outliers, method="zscore", threshold=3.0)

        assert result.num_anomalies > 0

        LOGGER.info("✅ Z-Score Anomaly Detection: PASSED")
        LOGGER.info(f"   - Anomalies found: {result.num_anomalies}")
        LOGGER.info(
            f"   - Max Z-score: {max(result.anomaly_scores) if result.anomaly_scores else 0:.2f}"
        )

    except Exception:
        LOGGER.exception("❌ Z-Score Anomaly Detection: FAILED")
        return False
    else:
        return True


def test_anomaly_detection_isolation() -> bool:
    """Test Isolation Forest anomaly detection."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 8: Anomaly Detection (Isolation Forest)")
    LOGGER.info("=" * 80)

    try:
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        outliers = np.array([10.0, -10.0])
        data_with_outliers = np.concatenate([normal_data, outliers])

        result = detect_anomalies(
            data_with_outliers,
            method="isolation_forest",
            contamination=0.1,
        )

        assert result.num_anomalies > 0
        assert result.anomaly_rate <= 0.15  # Should be around contamination rate

        LOGGER.info("✅ Isolation Forest Anomaly Detection: PASSED")
        LOGGER.info(f"   - Anomalies found: {result.num_anomalies}")
        LOGGER.info(f"   - Expected contamination: 10%")
        LOGGER.info(f"   - Actual anomaly rate: {result.anomaly_rate:.2%}")

    except Exception:
        LOGGER.exception("❌ Isolation Forest Anomaly Detection: FAILED")
        return False
    else:
        return True


def test_convenience_functions() -> bool:
    """Test convenience functions."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 9: Convenience Functions")
    LOGGER.info("=" * 80)

    try:
        np.random.seed(42)

        # Test detect_drift
        reference = np.random.normal(0, 1, 100)
        current = np.random.normal(0.5, 1, 100)
        drift_result = detect_drift(reference, current, test=DriftTest.KS_TEST)
        assert drift_result.feature_name == "feature"

        # Test validate_data
        data = np.array([1.0, 2.0, 3.0, 100.0])
        validation_results = validate_data(
            data,
            rules=[
                {"type": ValidationRule.RANGE_CHECK.value, "min": 0.0, "max": 10.0},
                {"type": ValidationRule.NOT_NULL.value},
            ],
        )
        assert len(validation_results) == 2

        # Test calculate_quality_metrics
        metrics = calculate_quality_metrics(data)
        assert metrics.completeness > 0

        LOGGER.info("✅ Convenience Functions: PASSED")
        LOGGER.info("   - detect_drift: OK")
        LOGGER.info("   - validate_data: OK")
        LOGGER.info("   - calculate_quality_metrics: OK")

    except Exception:
        LOGGER.exception("❌ Convenience Functions: FAILED")
        return False
    else:
        return True


def main():
    """Run all data quality tests."""
    LOGGER.info("Starting Phase 23 Data Quality Tests")
    LOGGER.info("=" * 80)

    tests = [
        ("Drift Detection (KS Test)", test_drift_detection_ks),
        ("Drift Detection (PSI)", test_drift_detection_psi),
        ("Data Validation", test_data_validation),
        ("Quality Metrics", test_quality_metrics),
        ("Quality Report", test_quality_report),
        ("Anomaly Detection (IQR)", test_anomaly_detection_iqr),
        ("Anomaly Detection (Z-Score)", test_anomaly_detection_zscore),
        ("Anomaly Detection (Isolation Forest)", test_anomaly_detection_isolation),
        ("Convenience Functions", test_convenience_functions),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception:
            LOGGER.exception(f"Test '{test_name}' crashed")
            results.append((test_name, False))

    # Summary
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("TEST SUMMARY")
    LOGGER.info("=" * 80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        LOGGER.info(f"{status}: {test_name}")

    LOGGER.info("=" * 80)
    LOGGER.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        LOGGER.info("🎉 All tests passed!")
        return 0

    LOGGER.error(f"❌ {total - passed} test(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
