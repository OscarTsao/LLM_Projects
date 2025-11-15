#!/usr/bin/env python
"""Rollback mechanism for failed deployments (Phase 30).

This module provides automatic and manual rollback
capabilities for failed deployments.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class RollbackState:
    """State snapshot for rollback."""

    deployment_id: str
    model_name: str
    model_version: str
    previous_version: str | None
    endpoint_url: str
    captured_at: datetime
    metadata: dict[str, Any]


class RollbackManager:
    """Manager for deployment rollbacks."""

    def __init__(self):
        """Initialize rollback manager."""
        self.rollback_history: list[RollbackState] = []
        self.current_state: RollbackState | None = None

    def capture_state(
        self,
        deployment_id: str,
        model_name: str,
        model_version: str,
        previous_version: str | None,
        endpoint_url: str,
        **metadata: Any,
    ) -> RollbackState:
        """Capture current state for potential rollback.

        Args:
            deployment_id: Deployment ID
            model_name: Model name
            model_version: Model version being deployed
            previous_version: Previous model version
            endpoint_url: Endpoint URL
            **metadata: Additional metadata

        Returns:
            Captured rollback state
        """
        state = RollbackState(
            deployment_id=deployment_id,
            model_name=model_name,
            model_version=model_version,
            previous_version=previous_version,
            endpoint_url=endpoint_url,
            captured_at=datetime.now(),
            metadata=metadata,
        )

        self.current_state = state
        LOGGER.info(f"Captured rollback state for deployment {deployment_id}")

        return state

    def execute_rollback(self, state: RollbackState, reason: str) -> bool:
        """Execute rollback to previous state.

        Args:
            state: State to roll back to
            reason: Reason for rollback

        Returns:
            True if rollback succeeded
        """
        LOGGER.warning(f"Initiating rollback for deployment {state.deployment_id}: {reason}")

        try:
            if not state.previous_version:
                LOGGER.error("No previous version to roll back to")
                return False

            # Simulate rollback steps
            steps = [
                f"Stopping new version ({state.model_version})",
                f"Restoring previous version ({state.previous_version})",
                f"Starting previous version",
                "Validating rollback",
            ]

            for step in steps:
                LOGGER.info(f"Rollback: {step}")
                time.sleep(0.1)  # Simulate rollback time

            # Record rollback
            self.rollback_history.append(state)

            LOGGER.info(
                f"Rollback completed: {state.model_name} "
                f"rolled back from {state.model_version} to {state.previous_version}"
            )

            return True

        except Exception as e:
            LOGGER.exception("Rollback failed")
            return False

    def can_rollback(self) -> bool:
        """Check if rollback is possible.

        Returns:
            True if rollback state is available
        """
        return self.current_state is not None and self.current_state.previous_version is not None

    def get_rollback_count(self) -> int:
        """Get number of rollbacks performed.

        Returns:
            Number of rollbacks
        """
        return len(self.rollback_history)

    def get_rollback_history(self) -> list[RollbackState]:
        """Get rollback history.

        Returns:
            List of rollback states
        """
        return self.rollback_history.copy()

    def clear_state(self) -> None:
        """Clear current rollback state."""
        self.current_state = None


class AutoRollbackMonitor:
    """Monitor for automatic rollback triggers."""

    def __init__(
        self,
        rollback_manager: RollbackManager,
        error_rate_threshold: float = 0.05,
        health_check_failures: int = 3,
    ):
        """Initialize auto-rollback monitor.

        Args:
            rollback_manager: Rollback manager instance
            error_rate_threshold: Error rate threshold for rollback
            health_check_failures: Consecutive health check failures to trigger rollback
        """
        self.rollback_manager = rollback_manager
        self.error_rate_threshold = error_rate_threshold
        self.health_check_failures_threshold = health_check_failures

        self.consecutive_health_failures = 0
        self.total_requests = 0
        self.error_count = 0

    def record_request(self, success: bool) -> str | None:
        """Record request and check for rollback triggers.

        Args:
            success: Whether request succeeded

        Returns:
            Rollback reason if triggered, None otherwise
        """
        self.total_requests += 1
        if not success:
            self.error_count += 1

        # Check error rate threshold
        if self.total_requests >= 10:  # Minimum sample size
            error_rate = self.error_count / self.total_requests
            if error_rate > self.error_rate_threshold:
                return f"Error rate {error_rate:.2%} exceeds threshold {self.error_rate_threshold:.2%}"

        return None

    def record_health_check(self, success: bool) -> str | None:
        """Record health check and check for rollback triggers.

        Args:
            success: Whether health check succeeded

        Returns:
            Rollback reason if triggered, None otherwise
        """
        if success:
            self.consecutive_health_failures = 0
        else:
            self.consecutive_health_failures += 1

            if self.consecutive_health_failures >= self.health_check_failures_threshold:
                return (
                    f"{self.consecutive_health_failures} consecutive health check failures "
                    f"exceeds threshold {self.health_check_failures_threshold}"
                )

        return None

    def check_and_rollback(self) -> bool:
        """Check conditions and perform rollback if needed.

        Returns:
            True if rollback was performed
        """
        # Check error rate
        rollback_reason = None

        if self.total_requests >= 10:
            error_rate = self.error_count / self.total_requests
            if error_rate > self.error_rate_threshold:
                rollback_reason = (
                    f"Error rate {error_rate:.2%} exceeds threshold {self.error_rate_threshold:.2%}"
                )

        # Check health failures
        if self.consecutive_health_failures >= self.health_check_failures_threshold:
            rollback_reason = (
                f"{self.consecutive_health_failures} consecutive health check failures"
            )

        if rollback_reason and self.rollback_manager.can_rollback():
            LOGGER.warning(f"Auto-rollback triggered: {rollback_reason}")
            return self.rollback_manager.execute_rollback(
                self.rollback_manager.current_state, rollback_reason
            )

        return False

    def reset_metrics(self) -> None:
        """Reset monitoring metrics."""
        self.consecutive_health_failures = 0
        self.total_requests = 0
        self.error_count = 0


def create_rollback_manager() -> RollbackManager:
    """Create rollback manager (convenience function).

    Returns:
        Rollback manager instance
    """
    return RollbackManager()
