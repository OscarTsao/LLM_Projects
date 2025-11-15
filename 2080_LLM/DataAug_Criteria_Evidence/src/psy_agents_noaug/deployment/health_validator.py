#!/usr/bin/env python
"""Health validation for deployments (Phase 30).

This module provides health checking and validation
for deployed models.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from psy_agents_noaug.deployment.deployment_config import HealthCheckConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    success: bool
    status_code: int | None = None
    response_time_ms: float = 0.0
    error_message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class HealthValidator:
    """Health validator for deployments."""

    def __init__(self, config: HealthCheckConfig):
        """Initialize health validator.

        Args:
            config: Health check configuration
        """
        self.config = config
        self.check_history: list[HealthCheckResult] = []

    def check_health(self, endpoint_url: str) -> HealthCheckResult:
        """Perform single health check.

        Args:
            endpoint_url: Base URL of endpoint

        Returns:
            Health check result
        """
        start_time = time.time()

        try:
            # Simulate health check
            # In production, this would make an actual HTTP request
            full_url = f"{endpoint_url}{self.config.endpoint}"

            LOGGER.debug(f"Health check: {full_url}")

            # Simulate network request
            time.sleep(0.01)

            # Simulate successful health check
            response_time_ms = (time.time() - start_time) * 1000

            result = HealthCheckResult(
                success=True,
                status_code=200,
                response_time_ms=response_time_ms,
                details={"endpoint": full_url, "status": "healthy"},
            )

            self.check_history.append(result)
            return result

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000

            result = HealthCheckResult(
                success=False,
                response_time_ms=response_time_ms,
                error_message=str(e),
                details={"endpoint": endpoint_url, "error": str(e)},
            )

            self.check_history.append(result)
            LOGGER.error(f"Health check failed: {e}")
            return result

    def validate_deployment(self, endpoint_url: str) -> bool:
        """Validate deployment health with retries.

        Args:
            endpoint_url: Base URL of endpoint

        Returns:
            True if deployment is healthy
        """
        LOGGER.info(f"Validating deployment health at {endpoint_url}")

        consecutive_successes = 0
        attempts = 0
        max_attempts = self.config.max_retries * self.config.success_threshold

        while attempts < max_attempts:
            result = self.check_health(endpoint_url)
            attempts += 1

            if result.success:
                consecutive_successes += 1
                LOGGER.info(
                    f"Health check passed ({consecutive_successes}/{self.config.success_threshold})"
                )

                if consecutive_successes >= self.config.success_threshold:
                    LOGGER.info("Deployment health validation successful")
                    return True

            else:
                consecutive_successes = 0
                LOGGER.warning(
                    f"Health check failed (attempt {attempts}/{max_attempts}): {result.error_message}"
                )

            # Wait before next check
            if attempts < max_attempts:
                time.sleep(self.config.interval_seconds)

        LOGGER.error("Deployment health validation failed")
        return False

    def get_health_stats(self) -> dict[str, Any]:
        """Get health check statistics.

        Returns:
            Health check statistics
        """
        if not self.check_history:
            return {
                "total_checks": 0,
                "success_rate": 0.0,
                "avg_response_time_ms": 0.0,
            }

        total = len(self.check_history)
        successes = sum(1 for check in self.check_history if check.success)
        avg_response_time = sum(check.response_time_ms for check in self.check_history) / total

        return {
            "total_checks": total,
            "successful_checks": successes,
            "failed_checks": total - successes,
            "success_rate": successes / total,
            "avg_response_time_ms": avg_response_time,
            "last_check_success": self.check_history[-1].success if self.check_history else None,
        }

    def clear_history(self) -> None:
        """Clear check history."""
        self.check_history.clear()


class MetricsValidator:
    """Validator for deployment metrics."""

    def __init__(self, error_rate_threshold: float = 0.05):
        """Initialize metrics validator.

        Args:
            error_rate_threshold: Error rate threshold (0.05 = 5%)
        """
        self.error_rate_threshold = error_rate_threshold
        self.total_requests = 0
        self.error_count = 0

    def record_request(self, success: bool) -> None:
        """Record a request.

        Args:
            success: Whether request succeeded
        """
        self.total_requests += 1
        if not success:
            self.error_count += 1

    def get_error_rate(self) -> float:
        """Get current error rate.

        Returns:
            Error rate (0.0 to 1.0)
        """
        if self.total_requests == 0:
            return 0.0
        return self.error_count / self.total_requests

    def is_error_rate_acceptable(self) -> bool:
        """Check if error rate is acceptable.

        Returns:
            True if error rate is below threshold
        """
        if self.total_requests < 10:  # Need minimum samples
            return True

        error_rate = self.get_error_rate()
        return error_rate <= self.error_rate_threshold

    def get_stats(self) -> dict[str, Any]:
        """Get validation statistics.

        Returns:
            Validation statistics
        """
        return {
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "success_count": self.total_requests - self.error_count,
            "error_rate": self.get_error_rate(),
            "threshold": self.error_rate_threshold,
            "acceptable": self.is_error_rate_acceptable(),
        }

    def reset(self) -> None:
        """Reset metrics."""
        self.total_requests = 0
        self.error_count = 0


def create_health_validator(config: HealthCheckConfig | None = None) -> HealthValidator:
    """Create health validator (convenience function).

    Args:
        config: Health check configuration

    Returns:
        Health validator instance
    """
    if config is None:
        config = HealthCheckConfig()

    return HealthValidator(config)
