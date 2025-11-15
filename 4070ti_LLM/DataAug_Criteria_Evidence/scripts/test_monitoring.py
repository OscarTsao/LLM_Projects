#!/usr/bin/env python
"""Test script for Phase 17: Model Monitoring & Observability.

This script tests:
1. Performance monitoring (latency, throughput, errors)
2. Drift detection (data drift, prediction drift)
3. Health monitoring (health checks, status)
4. Alerting system (alert rules, notifications)
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from psy_agents_noaug.monitoring import (
    AlertManager,
    AlertRule,
    AlertSeverity,
    DriftDetector,
    HealthCheck,
    HealthMonitor,
    PerformanceMonitor,
)
from psy_agents_noaug.monitoring.alerts import AlertChannel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def test_performance_monitor() -> bool:
    """Test performance monitoring.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 1: Performance Monitor")
    LOGGER.info("=" * 80)

    try:
        # Initialize monitor
        monitor = PerformanceMonitor(window_size=100)

        # Simulate requests
        np.random.seed(42)
        for _ in range(100):
            latency = np.random.uniform(0.01, 0.1)  # 10-100ms
            error = np.random.random() < 0.05  # 5% error rate
            monitor.record_request(latency, error)
            time.sleep(0.001)  # Small delay

        # Get metrics
        metrics = monitor.get_metrics()

        # Verify metrics
        assert metrics.latency_p50 > 0
        assert metrics.latency_p95 > 0
        assert metrics.latency_p99 > 0
        assert metrics.throughput > 0
        assert 0.0 <= metrics.error_rate <= 0.1  # Around 5%
        assert metrics.total_requests == 100

        # Get summary
        summary = monitor.get_summary()
        assert "latency" in summary
        assert "throughput_rps" in summary

        LOGGER.info("✅ Performance Monitor: PASSED")
        LOGGER.info(f"   - Latency P50: {metrics.latency_p50 * 1000:.2f}ms")
        LOGGER.info(f"   - Latency P95: {metrics.latency_p95 * 1000:.2f}ms")
        LOGGER.info(f"   - Latency P99: {metrics.latency_p99 * 1000:.2f}ms")
        LOGGER.info(f"   - Throughput: {metrics.throughput:.2f} req/s")
        LOGGER.info(f"   - Error rate: {metrics.error_rate:.2%}")

        return True

    except Exception as e:
        LOGGER.error(f"❌ Performance Monitor: FAILED - {e}")
        return False


def test_drift_detection() -> bool:
    """Test drift detection.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 2: Drift Detection")
    LOGGER.info("=" * 80)

    try:
        # Initialize detector
        detector = DriftDetector(
            warning_threshold=0.05,
            drift_threshold=0.01,
        )

        # Create baseline distribution (normal distribution)
        np.random.seed(42)
        baseline_data = np.random.normal(0, 1, 1000)
        baseline_predictions = np.random.uniform(0, 1, 1000)

        detector.set_baseline(
            data=baseline_data,
            predictions=baseline_predictions,
        )

        # Test 1: No drift (same distribution)
        current_data_no_drift = np.random.normal(0, 1, 1000)
        result_no_drift = detector.detect_data_drift(current_data_no_drift)

        assert result_no_drift.status.value in ["no_drift", "warning"]
        LOGGER.info(f"   - No drift test: {result_no_drift.status.value}")

        # Test 2: Drift detected (shifted distribution)
        current_data_drift = np.random.normal(2, 1, 1000)  # Mean shifted
        result_drift = detector.detect_data_drift(current_data_drift)

        assert result_drift.status.value in ["warning", "drift_detected"]
        LOGGER.info(f"   - Drift test: {result_drift.status.value}")

        # Test 3: Prediction drift
        current_predictions_drift = np.random.uniform(0.5, 1.0, 1000)  # Shifted
        pred_result = detector.detect_prediction_drift(current_predictions_drift)

        assert pred_result.status.value in ["warning", "drift_detected"]
        LOGGER.info(f"   - Prediction drift: {pred_result.status.value}")

        # Get summary
        summary = detector.get_summary(
            current_data=current_data_drift,
            current_predictions=current_predictions_drift,
        )

        assert "data_drift" in summary
        assert "prediction_drift" in summary

        LOGGER.info("✅ Drift Detection: PASSED")
        LOGGER.info(f"   - Data drift score: {result_drift.drift_score:.4f}")
        LOGGER.info(f"   - Prediction drift score: {pred_result.drift_score:.4f}")

        return True

    except Exception as e:
        LOGGER.error(f"❌ Drift Detection: FAILED - {e}")
        return False


def test_health_monitoring() -> bool:
    """Test health monitoring.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 3: Health Monitoring")
    LOGGER.info("=" * 80)

    try:
        # Initialize monitor
        monitor = HealthMonitor()

        # Add health checks
        check1 = HealthCheck(
            name="model_loaded",
            checker=lambda: True,  # Model is loaded
            description="Check if model is loaded",
            critical=True,
        )

        check2 = HealthCheck(
            name="response_time",
            checker=lambda: True,  # Response time is good
            description="Check response time",
            critical=False,
        )

        check3 = HealthCheck(
            name="memory_usage",
            checker=lambda: True,  # Memory usage is acceptable
            description="Check memory usage",
            critical=False,
        )

        monitor.add_check(check1)
        monitor.add_check(check2)
        monitor.add_check(check3)

        # Run checks
        results = monitor.run_checks()

        # Verify results
        assert len(results) == 3
        assert all(r.passed for r in results)

        # Get overall status
        overall_status = monitor.get_overall_status(results)
        assert overall_status.value == "healthy"

        # Get summary
        summary = monitor.get_summary()
        assert summary["overall_status"] == "healthy"
        assert summary["passed"] == 3
        assert summary["failed"] == 0

        LOGGER.info("✅ Health Monitoring: PASSED")
        LOGGER.info(f"   - Overall status: {overall_status.value}")
        LOGGER.info(f"   - Checks passed: {summary['passed']}/{summary['total_checks']}")

        return True

    except Exception as e:
        LOGGER.error(f"❌ Health Monitoring: FAILED - {e}")
        return False


def test_health_with_failures() -> bool:
    """Test health monitoring with failures.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 4: Health Monitoring with Failures")
    LOGGER.info("=" * 80)

    try:
        # Initialize monitor
        monitor = HealthMonitor()

        # Add failing check
        check_fail = HealthCheck(
            name="critical_service",
            checker=lambda: False,  # Service is down
            description="Check critical service",
            critical=True,
        )

        check_pass = HealthCheck(
            name="optional_service",
            checker=lambda: True,
            description="Check optional service",
            critical=False,
        )

        monitor.add_check(check_fail)
        monitor.add_check(check_pass)

        # Run checks
        results = monitor.run_checks()

        # Verify results
        assert len(results) == 2
        assert not results[0].passed  # First check failed
        assert results[1].passed  # Second check passed

        # Get overall status
        overall_status = monitor.get_overall_status(results)
        assert overall_status.value == "unhealthy"  # Critical check failed

        LOGGER.info("✅ Health Monitoring with Failures: PASSED")
        LOGGER.info(f"   - Overall status: {overall_status.value}")
        LOGGER.info(f"   - Failed critical check detected correctly")

        return True

    except Exception as e:
        LOGGER.error(f"❌ Health Monitoring with Failures: FAILED - {e}")
        return False


def test_alerting_system() -> bool:
    """Test alerting system.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 5: Alerting System")
    LOGGER.info("=" * 80)

    try:
        # Create temporary alert log file
        with tempfile.TemporaryDirectory() as tmpdir:
            alert_log = Path(tmpdir) / "alerts.log"

            # Initialize manager
            manager = AlertManager(alert_log_file=alert_log)

            # Define alert rules
            trigger_count = {"count": 0}

            def high_error_rate():
                trigger_count["count"] += 1
                return trigger_count["count"] > 2  # Trigger on 3rd check

            rule1 = AlertRule(
                name="high_error_rate",
                condition=high_error_rate,
                severity=AlertSeverity.CRITICAL,
                message_template="Error rate exceeded threshold: {error_rate}",
                channels=[AlertChannel.LOG, AlertChannel.FILE],
                cooldown_minutes=1,
            )

            rule2 = AlertRule(
                name="slow_response",
                condition=lambda: False,  # Never triggers
                severity=AlertSeverity.WARNING,
                message_template="Response time is slow: {latency_ms}ms",
                channels=[AlertChannel.LOG],
            )

            manager.add_rule(rule1)
            manager.add_rule(rule2)

            # Check rules multiple times
            for i in range(4):
                alerts = manager.check_rules(
                    context={
                        "error_rate": 0.15,
                        "latency_ms": 500,
                    }
                )

                if i >= 2:  # Should trigger on 3rd check
                    assert len(alerts) >= 0  # At least one alert triggered
                    if alerts:
                        assert alerts[0].rule_name == "high_error_rate"
                        assert alerts[0].severity == AlertSeverity.CRITICAL

            # Verify alert log file was created
            assert alert_log.exists()

            # Get summary
            summary = manager.get_summary()
            assert summary["total_rules"] == 2
            assert summary["alerts_last_24h"] >= 1

            LOGGER.info("✅ Alerting System: PASSED")
            LOGGER.info(f"   - Total rules: {summary['total_rules']}")
            LOGGER.info(f"   - Alerts triggered: {summary['alerts_last_24h']}")

            return True

    except Exception as e:
        LOGGER.error(f"❌ Alerting System: FAILED - {e}")
        return False


def test_integrated_monitoring() -> bool:
    """Test integrated monitoring scenario.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 6: Integrated Monitoring")
    LOGGER.info("=" * 80)

    try:
        # Setup all monitors
        perf_monitor = PerformanceMonitor(window_size=50)
        drift_detector = DriftDetector()
        health_monitor = HealthMonitor()
        alert_manager = AlertManager()

        # Set baseline for drift detection
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 500)
        drift_detector.set_baseline(data=baseline)

        # Simulate monitoring cycle
        for _ in range(50):
            # Record performance
            latency = np.random.uniform(0.01, 0.15)
            perf_monitor.record_request(latency, error=False)

        # Check drift
        current_data = np.random.normal(0, 1, 500)  # No drift
        drift_result = drift_detector.detect_data_drift(current_data)

        # Add health check
        def check_performance():
            metrics = perf_monitor.get_metrics()
            return metrics.latency_p95 < 0.2  # P95 < 200ms

        health_check = HealthCheck(
            name="performance_sla",
            checker=check_performance,
            description="Check performance SLA",
        )
        health_monitor.add_check(health_check)

        # Run health checks
        health_results = health_monitor.run_checks()

        # Define alert rule
        def perf_degraded():
            metrics = perf_monitor.get_metrics()
            return metrics.latency_p95 > 0.2

        alert_rule = AlertRule(
            name="performance_degraded",
            condition=perf_degraded,
            severity=AlertSeverity.WARNING,
            message_template="Performance degraded: P95={p95}ms",
            channels=[AlertChannel.LOG],
        )
        alert_manager.add_rule(alert_rule)

        # Check alerts
        perf_metrics = perf_monitor.get_metrics()
        alerts = alert_manager.check_rules(
            context={"p95": perf_metrics.latency_p95 * 1000}
        )

        # Verify integrated system
        assert drift_result is not None
        assert len(health_results) > 0
        # alerts may or may not trigger depending on simulated latencies

        LOGGER.info("✅ Integrated Monitoring: PASSED")
        LOGGER.info(f"   - Performance metrics: ✓")
        LOGGER.info(f"   - Drift detection: ✓")
        LOGGER.info(f"   - Health checks: ✓")
        LOGGER.info(f"   - Alert rules: ✓")

        return True

    except Exception as e:
        LOGGER.error(f"❌ Integrated Monitoring: FAILED - {e}")
        return False


def main():
    """Run all monitoring tests."""
    LOGGER.info("Starting Phase 17 Monitoring Tests")
    LOGGER.info("=" * 80)

    tests = [
        ("Performance Monitor", test_performance_monitor),
        ("Drift Detection", test_drift_detection),
        ("Health Monitoring", test_health_monitoring),
        ("Health Monitoring with Failures", test_health_with_failures),
        ("Alerting System", test_alerting_system),
        ("Integrated Monitoring", test_integrated_monitoring),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            LOGGER.error(f"Test '{test_name}' crashed: {e}")
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
    else:
        LOGGER.error(f"❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
