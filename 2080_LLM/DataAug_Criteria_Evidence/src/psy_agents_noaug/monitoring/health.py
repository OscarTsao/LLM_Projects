#!/usr/bin/env python
"""Health monitoring for model services (Phase 17).

This module provides health monitoring including:
- Model availability checks
- Response time validation
- Error rate monitoring
- Resource health checks
- Dependency health
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

LOGGER = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """A single health check definition."""

    name: str
    checker: Callable[[], bool]
    description: str = ""
    critical: bool = True  # If False, failures are warnings


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    check_name: str
    status: HealthStatus
    passed: bool
    message: str
    timestamp: datetime
    duration_ms: float


class HealthMonitor:
    """Monitor model and service health."""

    def __init__(self):
        """Initialize health monitor."""
        self.checks: list[HealthCheck] = []
        self.check_history: list[HealthCheckResult] = []

        LOGGER.info("Initialized HealthMonitor")

    def add_check(self, check: HealthCheck) -> None:
        """Add a health check.

        Args:
            check: Health check to add
        """
        self.checks.append(check)
        LOGGER.debug("Added health check: %s", check.name)

    def _run_check(self, check: HealthCheck) -> HealthCheckResult:
        """Run a single health check.

        Args:
            check: Health check to run

        Returns:
            Check result
        """
        import time

        start_time = time.time()
        timestamp = datetime.now()

        try:
            passed = check.checker()
            duration_ms = (time.time() - start_time) * 1000

            if passed:
                status = HealthStatus.HEALTHY
                message = f"✓ {check.name}: Passed"
            else:
                status = (
                    HealthStatus.DEGRADED
                    if not check.critical
                    else HealthStatus.UNHEALTHY
                )
                message = f"✗ {check.name}: Failed"

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            passed = False
            status = HealthStatus.UNHEALTHY
            message = f"✗ {check.name}: Error - {e}"
            LOGGER.error("Health check failed: %s - %s", check.name, e)

        return HealthCheckResult(
            check_name=check.name,
            status=status,
            passed=passed,
            message=message,
            timestamp=timestamp,
            duration_ms=duration_ms,
        )

    def run_checks(self) -> list[HealthCheckResult]:
        """Run all health checks.

        Returns:
            List of check results
        """
        LOGGER.info("Running %d health checks", len(self.checks))

        results = []
        for check in self.checks:
            result = self._run_check(check)
            results.append(result)
            self.check_history.append(result)

        return results

    def get_overall_status(
        self,
        results: list[HealthCheckResult] | None = None,
    ) -> HealthStatus:
        """Get overall health status.

        Args:
            results: Check results (if None, runs checks)

        Returns:
            Overall health status
        """
        if results is None:
            results = self.run_checks()

        # If any critical check is unhealthy, system is unhealthy
        for result in results:
            if result.status == HealthStatus.UNHEALTHY:
                return HealthStatus.UNHEALTHY

        # If any check is degraded, system is degraded
        for result in results:
            if result.status == HealthStatus.DEGRADED:
                return HealthStatus.DEGRADED

        # All checks passed
        return HealthStatus.HEALTHY

    def get_summary(self) -> dict[str, Any]:
        """Get health check summary.

        Returns:
            Summary dict
        """
        results = self.run_checks()
        overall_status = self.get_overall_status(results)

        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status.value,
            "total_checks": len(results),
            "passed": passed_count,
            "failed": failed_count,
            "checks": [
                {
                    "name": r.check_name,
                    "status": r.status.value,
                    "passed": r.passed,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                }
                for r in results
            ],
        }


def check_health(checks: list[HealthCheck]) -> dict[str, Any]:
    """Check health (convenience function).

    Args:
        checks: List of health checks

    Returns:
        Health summary
    """
    monitor = HealthMonitor()

    for check in checks:
        monitor.add_check(check)

    return monitor.get_summary()
