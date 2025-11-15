#!/usr/bin/env python
"""Deployment orchestrator for coordinating deployments (Phase 30).

This module provides the main orchestrator that coordinates
deployment strategies, health checks, and rollbacks.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from psy_agents_noaug.deployment.deployment_config import (
    DeploymentConfig,
    DeploymentRecord,
    DeploymentStatus,
)
from psy_agents_noaug.deployment.health_validator import (
    HealthValidator,
    MetricsValidator,
)
from psy_agents_noaug.deployment.rollback import RollbackManager
from psy_agents_noaug.deployment.strategies import get_strategy

LOGGER = logging.getLogger(__name__)


class DeploymentOrchestrator:
    """Orchestrator for model deployments."""

    def __init__(self):
        """Initialize deployment orchestrator."""
        self.deployments: dict[str, DeploymentRecord] = {}
        self.rollback_manager = RollbackManager()
        self.active_deployment: DeploymentRecord | None = None

    def deploy(
        self, config: DeploymentConfig, previous_version: str | None = None
    ) -> DeploymentRecord:
        """Execute deployment.

        Args:
            config: Deployment configuration
            previous_version: Previous model version for rollback

        Returns:
            Deployment record
        """
        LOGGER.info(f"Starting deployment: {config.deployment_id}")

        # Create deployment record
        record = DeploymentRecord(
            deployment_id=config.deployment_id,
            config=config,
            status=DeploymentStatus.PENDING,
            started_at=datetime.now(),
        )

        self.deployments[config.deployment_id] = record
        self.active_deployment = record

        try:
            # Capture rollback state
            self.rollback_manager.capture_state(
                deployment_id=config.deployment_id,
                model_name=config.model_name,
                model_version=config.model_version,
                previous_version=previous_version,
                endpoint_url=config.target.get_endpoint_url(),
                strategy=config.strategy.value,
            )

            # Get and execute deployment strategy
            strategy = get_strategy(config)
            record.add_log(f"Using {config.strategy.value} deployment strategy")

            success = strategy.execute(record)

            if not success:
                record.mark_failed("Deployment strategy execution failed")
                return record

            # Validate deployment health
            record.set_status(DeploymentStatus.VALIDATING)
            health_validator = HealthValidator(config.target.health_check)

            if not health_validator.validate_deployment(
                config.target.get_endpoint_url()
            ):
                record.add_log("Health validation failed")

                if config.auto_rollback and config.rollback_on_health_failure:
                    record.add_log("Triggering automatic rollback")
                    self._execute_rollback(record, "Health validation failed")
                else:
                    record.mark_failed("Health validation failed")

                return record

            # Validate metrics if configured
            if config.rollback_on_error_rate:
                metrics_validator = MetricsValidator(config.error_rate_threshold)
                record.add_log(
                    f"Monitoring error rate (threshold: {config.error_rate_threshold:.2%})"
                )

                # In production, this would monitor actual traffic
                # For now, we simulate successful validation
                if metrics_validator.is_error_rate_acceptable():
                    record.add_log("Error rate is acceptable")
                else:
                    record.add_log("Error rate exceeds threshold")

                    if config.auto_rollback:
                        record.add_log("Triggering automatic rollback")
                        self._execute_rollback(record, "Error rate threshold exceeded")
                        return record

            # Deployment successful
            record.mark_completed()
            record.add_log(
                f"Deployment completed: {config.model_name} v{config.model_version}"
            )

            return record

        except Exception as e:
            record.mark_failed(str(e))
            LOGGER.exception("Deployment failed")

            if config.auto_rollback:
                record.add_log("Triggering automatic rollback due to exception")
                self._execute_rollback(record, f"Exception: {e}")

            return record

    def _execute_rollback(self, record: DeploymentRecord, reason: str) -> bool:
        """Execute rollback for a deployment.

        Args:
            record: Deployment record
            reason: Reason for rollback

        Returns:
            True if rollback succeeded
        """
        if not self.rollback_manager.can_rollback():
            record.add_log("Rollback not possible: no previous state")
            return False

        success = self.rollback_manager.execute_rollback(
            self.rollback_manager.current_state,
            reason,
        )

        if success:
            record.mark_rolled_back(reason)
        else:
            record.add_log("Rollback failed")

        return success

    def rollback_deployment(
        self, deployment_id: str, reason: str = "Manual rollback"
    ) -> bool:
        """Manually trigger rollback for a deployment.

        Args:
            deployment_id: Deployment ID
            reason: Reason for rollback

        Returns:
            True if rollback succeeded
        """
        record = self.deployments.get(deployment_id)
        if not record:
            LOGGER.error(f"Deployment {deployment_id} not found")
            return False

        LOGGER.info(f"Manual rollback triggered for {deployment_id}: {reason}")
        return self._execute_rollback(record, reason)

    def get_deployment(self, deployment_id: str) -> DeploymentRecord | None:
        """Get deployment record.

        Args:
            deployment_id: Deployment ID

        Returns:
            Deployment record or None
        """
        return self.deployments.get(deployment_id)

    def get_active_deployment(self) -> DeploymentRecord | None:
        """Get currently active deployment.

        Returns:
            Active deployment record or None
        """
        return self.active_deployment

    def list_deployments(
        self, status: DeploymentStatus | None = None
    ) -> list[DeploymentRecord]:
        """List deployments with optional status filter.

        Args:
            status: Filter by status

        Returns:
            List of deployment records
        """
        deployments = list(self.deployments.values())

        if status:
            deployments = [d for d in deployments if d.status == status]

        return sorted(deployments, key=lambda d: d.started_at, reverse=True)

    def get_deployment_stats(self) -> dict[str, Any]:
        """Get deployment statistics.

        Returns:
            Deployment statistics
        """
        total = len(self.deployments)

        if total == 0:
            return {
                "total_deployments": 0,
                "completed": 0,
                "failed": 0,
                "rolled_back": 0,
                "success_rate": 0.0,
            }

        completed = len(
            [
                d
                for d in self.deployments.values()
                if d.status == DeploymentStatus.COMPLETED
            ]
        )
        failed = len(
            [
                d
                for d in self.deployments.values()
                if d.status == DeploymentStatus.FAILED
            ]
        )
        rolled_back = len(
            [
                d
                for d in self.deployments.values()
                if d.status == DeploymentStatus.ROLLED_BACK
            ]
        )

        # Calculate average deployment time
        completed_deployments = [
            d
            for d in self.deployments.values()
            if d.status == DeploymentStatus.COMPLETED and d.get_duration_seconds()
        ]

        avg_duration = (
            sum(d.get_duration_seconds() for d in completed_deployments)
            / len(completed_deployments)
            if completed_deployments
            else 0.0
        )

        return {
            "total_deployments": total,
            "completed": completed,
            "failed": failed,
            "rolled_back": rolled_back,
            "in_progress": total - completed - failed - rolled_back,
            "success_rate": completed / total if total > 0 else 0.0,
            "avg_deployment_time_seconds": avg_duration,
            "rollback_count": self.rollback_manager.get_rollback_count(),
        }


def create_orchestrator() -> DeploymentOrchestrator:
    """Create deployment orchestrator (convenience function).

    Returns:
        Deployment orchestrator instance
    """
    return DeploymentOrchestrator()
