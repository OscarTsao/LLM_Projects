#!/usr/bin/env python
"""Model monitoring and observability (Phase 26 - Enhanced).

This module provides comprehensive model monitoring including:
- Performance monitoring (latency, throughput, resource usage)
- Prediction drift detection with statistical tests
- Data quality and distribution monitoring
- Model health checks and validation
- Alert system with thresholds and rules

Phase 26 Enhancements:
- Enhanced performance monitoring with percentile metrics
- Advanced drift detection (KS test, JS divergence)
- Comprehensive health checks with validators
- Improved alert system with cooldown and consecutive violations
"""

from __future__ import annotations

# Phase 17 - Original monitoring (backward compatibility)
from psy_agents_noaug.monitoring.drift import DriftDetector, detect_drift
from psy_agents_noaug.monitoring.health import (
    HealthCheck,
    HealthMonitor,
    check_health,
)
from psy_agents_noaug.monitoring.performance import (
    PerformanceMonitor as PerformanceMonitorV1,
)
from psy_agents_noaug.monitoring.performance import monitor_performance

# Phase 26 - Enhanced monitoring
from psy_agents_noaug.monitoring.alerts import (
    Alert,
    AlertChannel,
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    create_drift_alert_rule,
    create_threshold_rule,
)
from psy_agents_noaug.monitoring.health_checks import (
    HealthCheckResult,
    HealthChecker,
    HealthStatus,
    ModelValidator,
    create_error_rate_check,
    create_latency_check,
)
from psy_agents_noaug.monitoring.performance_monitor import (
    LatencyTracker,
    PerformanceMetrics,
    PerformanceMonitor,
    RequestRecord,
    track_latency,
)
from psy_agents_noaug.monitoring.prediction_monitor import (
    DriftMetrics,
    PredictionMonitor,
    PredictionStats,
    calculate_prediction_entropy,
)

__all__ = [
    # Phase 17 - Original (backward compatibility)
    "PerformanceMonitorV1",
    "monitor_performance",
    "DriftDetector",
    "detect_drift",
    "HealthMonitor",
    "HealthCheck",
    "check_health",
    # Phase 26 - Enhanced performance monitoring
    "PerformanceMonitor",
    "PerformanceMetrics",
    "RequestRecord",
    "LatencyTracker",
    "track_latency",
    # Phase 26 - Enhanced prediction monitoring
    "PredictionMonitor",
    "PredictionStats",
    "DriftMetrics",
    "calculate_prediction_entropy",
    # Phase 26 - Enhanced health checks
    "HealthChecker",
    "HealthCheckResult",
    "HealthStatus",
    "ModelValidator",
    "create_latency_check",
    "create_error_rate_check",
    # Phase 26 - Enhanced alerts
    "AlertManager",
    "Alert",
    "AlertRule",
    "AlertSeverity",
    "AlertStatus",
    "AlertChannel",
    "create_threshold_rule",
    "create_drift_alert_rule",
]
