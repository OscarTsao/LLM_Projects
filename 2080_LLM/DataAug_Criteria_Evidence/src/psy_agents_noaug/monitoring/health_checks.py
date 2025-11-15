#!/usr/bin/env python
"""Model health checks and validation (Phase 26).

This module provides tools for checking model health, validating inputs/outputs,
and ensuring model behavior meets expectations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np

LOGGER = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status."""

    HEALTHY = "healthy"
    WARNING = "warning"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    check_name: str
    status: HealthStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def is_healthy(self) -> bool:
        """Check if result is healthy.

        Returns:
            True if healthy
        """
        return self.status == HealthStatus.HEALTHY

    def get_summary(self) -> dict[str, Any]:
        """Get check result summary.

        Returns:
            Summary dictionary
        """
        return {
            "check_name": self.check_name,
            "status": self.status.value,
            "message": self.message,
            "is_healthy": self.is_healthy(),
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


class HealthChecker:
    """Perform model health checks."""

    def __init__(self):
        """Initialize health checker."""
        self.checks: dict[str, Callable[[], HealthCheckResult]] = {}
        self.check_history: list[HealthCheckResult] = []

        LOGGER.info("Initialized HealthChecker")

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], HealthCheckResult],
    ) -> None:
        """Register a health check.

        Args:
            name: Check name
            check_fn: Check function returning HealthCheckResult
        """
        self.checks[name] = check_fn
        LOGGER.info(f"Registered health check: {name}")

    def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check.

        Args:
            name: Check name

        Returns:
            Health check result
        """
        if name not in self.checks:
            return HealthCheckResult(
                check_name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Unknown check: {name}",
            )

        try:
            result = self.checks[name]()
            self.check_history.append(result)

            LOGGER.info(f"Health check '{name}': {result.status.value}")
            return result

        except Exception as e:
            result = HealthCheckResult(
                check_name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed with error: {e!s}",
                details={"error_type": type(e).__name__},
            )
            self.check_history.append(result)

            LOGGER.error(f"Health check '{name}' failed: {e}")
            return result

    def run_all_checks(self) -> dict[str, HealthCheckResult]:
        """Run all registered checks.

        Returns:
            Dictionary of check results
        """
        results = {}

        for name in self.checks:
            results[name] = self.run_check(name)

        return results

    def get_overall_status(self) -> HealthStatus:
        """Get overall health status.

        Returns:
            Overall health status
        """
        results = self.run_all_checks()

        if not results:
            return HealthStatus.UNKNOWN

        # If any unhealthy, overall is unhealthy
        if any(r.status == HealthStatus.UNHEALTHY for r in results.values()):
            return HealthStatus.UNHEALTHY

        # If any warnings, overall is warning
        if any(r.status == HealthStatus.WARNING for r in results.values()):
            return HealthStatus.WARNING

        # Otherwise healthy
        return HealthStatus.HEALTHY

    def get_health_report(self) -> dict[str, Any]:
        """Get comprehensive health report.

        Returns:
            Health report
        """
        results = self.run_all_checks()
        overall = self.get_overall_status()

        return {
            "overall_status": overall.value,
            "num_checks": len(results),
            "num_healthy": sum(
                1 for r in results.values() if r.status == HealthStatus.HEALTHY
            ),
            "num_warnings": sum(
                1 for r in results.values() if r.status == HealthStatus.WARNING
            ),
            "num_unhealthy": sum(
                1 for r in results.values() if r.status == HealthStatus.UNHEALTHY
            ),
            "checks": {name: result.get_summary() for name, result in results.items()},
            "generated_at": datetime.now().isoformat(),
        }


class ModelValidator:
    """Validate model inputs and outputs."""

    def __init__(
        self,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        input_range: tuple[float, float] | None = None,
        output_range: tuple[float, float] | None = None,
    ):
        """Initialize model validator.

        Args:
            input_shape: Expected input shape
            output_shape: Expected output shape
            input_range: Valid input value range (min, max)
            output_range: Valid output value range (min, max)
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_range = input_range
        self.output_range = output_range

        LOGGER.info("Initialized ModelValidator")

    def validate_input(self, data: np.ndarray) -> HealthCheckResult:
        """Validate input data.

        Args:
            data: Input data

        Returns:
            Validation result
        """
        issues = []

        # Check shape
        if self.input_shape is not None:
            if data.shape != self.input_shape:
                issues.append(
                    f"Shape mismatch: expected {self.input_shape}, "
                    f"got {data.shape}"
                )

        # Check for NaN/Inf
        if np.any(np.isnan(data)):
            issues.append("Contains NaN values")

        if np.any(np.isinf(data)):
            issues.append("Contains Inf values")

        # Check range
        if self.input_range is not None:
            min_val, max_val = self.input_range
            if np.any(data < min_val) or np.any(data > max_val):
                issues.append(
                    f"Values outside valid range [{min_val}, {max_val}]"
                )

        # Determine status
        if not issues:
            return HealthCheckResult(
                check_name="input_validation",
                status=HealthStatus.HEALTHY,
                message="Input validation passed",
            )

        return HealthCheckResult(
            check_name="input_validation",
            status=HealthStatus.UNHEALTHY,
            message=f"Input validation failed: {'; '.join(issues)}",
            details={"issues": issues},
        )

    def validate_output(self, data: np.ndarray) -> HealthCheckResult:
        """Validate output data.

        Args:
            data: Output data

        Returns:
            Validation result
        """
        issues = []

        # Check shape
        if self.output_shape is not None:
            if data.shape != self.output_shape:
                issues.append(
                    f"Shape mismatch: expected {self.output_shape}, "
                    f"got {data.shape}"
                )

        # Check for NaN/Inf
        if np.any(np.isnan(data)):
            issues.append("Contains NaN values")

        if np.any(np.isinf(data)):
            issues.append("Contains Inf values")

        # Check range
        if self.output_range is not None:
            min_val, max_val = self.output_range
            if np.any(data < min_val) or np.any(data > max_val):
                issues.append(
                    f"Values outside valid range [{min_val}, {max_val}]"
                )

        # Determine status
        if not issues:
            return HealthCheckResult(
                check_name="output_validation",
                status=HealthStatus.HEALTHY,
                message="Output validation passed",
            )

        return HealthCheckResult(
            check_name="output_validation",
            status=HealthStatus.UNHEALTHY,
            message=f"Output validation failed: {'; '.join(issues)}",
            details={"issues": issues},
        )

    def validate_probabilities(
        self,
        probabilities: np.ndarray,
        tolerance: float = 1e-5,
    ) -> HealthCheckResult:
        """Validate probability distributions.

        Args:
            probabilities: Probability distributions (should sum to 1)
            tolerance: Tolerance for sum check

        Returns:
            Validation result
        """
        issues = []

        # Check range [0, 1]
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            issues.append("Probabilities outside [0, 1] range")

        # Check sum to 1 (for each sample)
        if len(probabilities.shape) == 2:
            sums = np.sum(probabilities, axis=1)
            if not np.allclose(sums, 1.0, atol=tolerance):
                issues.append(
                    f"Probabilities don't sum to 1 (max deviation: "
                    f"{np.max(np.abs(sums - 1.0)):.6f})"
                )

        # Check for NaN/Inf
        if np.any(np.isnan(probabilities)):
            issues.append("Contains NaN values")

        if np.any(np.isinf(probabilities)):
            issues.append("Contains Inf values")

        # Determine status
        if not issues:
            return HealthCheckResult(
                check_name="probability_validation",
                status=HealthStatus.HEALTHY,
                message="Probability validation passed",
            )

        return HealthCheckResult(
            check_name="probability_validation",
            status=HealthStatus.UNHEALTHY,
            message=f"Probability validation failed: {'; '.join(issues)}",
            details={"issues": issues},
        )


def create_latency_check(
    max_latency: float,
    get_current_latency: Callable[[], float],
) -> Callable[[], HealthCheckResult]:
    """Create a latency health check.

    Args:
        max_latency: Maximum acceptable latency (seconds)
        get_current_latency: Function to get current latency

    Returns:
        Health check function
    """

    def check() -> HealthCheckResult:
        current = get_current_latency()

        if current <= max_latency:
            return HealthCheckResult(
                check_name="latency",
                status=HealthStatus.HEALTHY,
                message=f"Latency within threshold ({current:.3f}s <= {max_latency}s)",
                details={"current_latency": current, "max_latency": max_latency},
            )

        # Warning if slightly over, unhealthy if significantly over
        if current <= max_latency * 1.5:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.UNHEALTHY

        return HealthCheckResult(
            check_name="latency",
            status=status,
            message=f"Latency exceeds threshold ({current:.3f}s > {max_latency}s)",
            details={"current_latency": current, "max_latency": max_latency},
        )

    return check


def create_error_rate_check(
    max_error_rate: float,
    get_current_error_rate: Callable[[], float],
) -> Callable[[], HealthCheckResult]:
    """Create an error rate health check.

    Args:
        max_error_rate: Maximum acceptable error rate (0-1)
        get_current_error_rate: Function to get current error rate

    Returns:
        Health check function
    """

    def check() -> HealthCheckResult:
        current = get_current_error_rate()

        if current <= max_error_rate:
            return HealthCheckResult(
                check_name="error_rate",
                status=HealthStatus.HEALTHY,
                message=f"Error rate within threshold ({current:.2%} <= {max_error_rate:.2%})",
                details={"current_error_rate": current, "max_error_rate": max_error_rate},
            )

        # Warning if slightly over, unhealthy if significantly over
        if current <= max_error_rate * 1.5:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.UNHEALTHY

        return HealthCheckResult(
            check_name="error_rate",
            status=status,
            message=f"Error rate exceeds threshold ({current:.2%} > {max_error_rate:.2%})",
            details={"current_error_rate": current, "max_error_rate": max_error_rate},
        )

    return check
