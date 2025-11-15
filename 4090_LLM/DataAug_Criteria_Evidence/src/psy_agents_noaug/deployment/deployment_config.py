#!/usr/bin/env python
"""Deployment configuration for model deployment (Phase 30).

This module provides configuration management for model deployments
across different environments and deployment strategies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

LOGGER = logging.getLogger(__name__)


class DeploymentEnvironment(str, Enum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"


class DeploymentStrategy(str, Enum):
    """Deployment strategy."""

    DIRECT = "direct"  # Direct replacement
    BLUE_GREEN = "blue_green"  # Blue-green deployment
    CANARY = "canary"  # Canary deployment
    ROLLING = "rolling"  # Rolling deployment


class DeploymentStatus(str, Enum):
    """Deployment status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ResourceRequirements:
    """Resource requirements for deployment."""

    cpu_cores: int = 2
    memory_gb: int = 4
    gpu_count: int = 0
    gpu_memory_gb: int = 0
    disk_gb: int = 10


@dataclass
class HealthCheckConfig:
    """Health check configuration."""

    endpoint: str = "/health"
    timeout_seconds: int = 10
    interval_seconds: int = 30
    max_retries: int = 3
    success_threshold: int = 2  # Consecutive successes needed


@dataclass
class DeploymentTarget:
    """Deployment target configuration."""

    name: str
    environment: DeploymentEnvironment
    host: str
    port: int
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_endpoint_url(self) -> str:
        """Get full endpoint URL.

        Returns:
            Endpoint URL
        """
        return f"http://{self.host}:{self.port}"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""

    deployment_id: str
    model_name: str
    model_version: str
    strategy: DeploymentStrategy
    target: DeploymentTarget

    # Strategy-specific settings
    canary_traffic_percent: float = 10.0  # For canary deployments
    validation_period_seconds: int = 300  # Time to validate before full rollout

    # Rollback settings
    auto_rollback: bool = True
    rollback_on_health_failure: bool = True
    rollback_on_error_rate: bool = True
    error_rate_threshold: float = 0.05  # 5% error rate triggers rollback

    # Model serving settings
    batch_size: int = 32
    timeout_seconds: int = 60
    max_concurrent_requests: int = 100

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "deployment_id": self.deployment_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "strategy": self.strategy.value,
            "target": {
                "name": self.target.name,
                "environment": self.target.environment.value,
                "host": self.target.host,
                "port": self.target.port,
                "resources": {
                    "cpu_cores": self.target.resources.cpu_cores,
                    "memory_gb": self.target.resources.memory_gb,
                    "gpu_count": self.target.resources.gpu_count,
                    "gpu_memory_gb": self.target.resources.gpu_memory_gb,
                    "disk_gb": self.target.resources.disk_gb,
                },
                "health_check": {
                    "endpoint": self.target.health_check.endpoint,
                    "timeout_seconds": self.target.health_check.timeout_seconds,
                    "interval_seconds": self.target.health_check.interval_seconds,
                    "max_retries": self.target.health_check.max_retries,
                    "success_threshold": self.target.health_check.success_threshold,
                },
            },
            "canary_traffic_percent": self.canary_traffic_percent,
            "validation_period_seconds": self.validation_period_seconds,
            "auto_rollback": self.auto_rollback,
            "rollback_on_health_failure": self.rollback_on_health_failure,
            "rollback_on_error_rate": self.rollback_on_error_rate,
            "error_rate_threshold": self.error_rate_threshold,
            "batch_size": self.batch_size,
            "timeout_seconds": self.timeout_seconds,
            "max_concurrent_requests": self.max_concurrent_requests,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "metadata": self.metadata,
        }


@dataclass
class DeploymentRecord:
    """Record of a deployment."""

    deployment_id: str
    config: DeploymentConfig
    status: DeploymentStatus
    started_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)

    def add_log(self, message: str) -> None:
        """Add log entry.

        Args:
            message: Log message
        """
        timestamp = datetime.now().isoformat()
        self.logs.append(f"[{timestamp}] {message}")
        LOGGER.info(f"Deployment {self.deployment_id}: {message}")

    def set_status(self, status: DeploymentStatus, message: str | None = None) -> None:
        """Update deployment status.

        Args:
            status: New status
            message: Optional status message
        """
        self.status = status
        log_msg = f"Status changed to {status.value}"
        if message:
            log_msg += f": {message}"
        self.add_log(log_msg)

        if status in (
            DeploymentStatus.COMPLETED,
            DeploymentStatus.FAILED,
            DeploymentStatus.ROLLED_BACK,
        ):
            self.completed_at = datetime.now()

    def mark_completed(
        self, message: str = "Deployment completed successfully"
    ) -> None:
        """Mark deployment as completed.

        Args:
            message: Completion message
        """
        self.set_status(DeploymentStatus.COMPLETED, message)

    def mark_failed(self, error: str) -> None:
        """Mark deployment as failed.

        Args:
            error: Error message
        """
        self.error_message = error
        self.set_status(DeploymentStatus.FAILED, f"Error: {error}")

    def mark_rolled_back(self, reason: str) -> None:
        """Mark deployment as rolled back.

        Args:
            reason: Rollback reason
        """
        self.set_status(DeploymentStatus.ROLLED_BACK, f"Rolled back: {reason}")

    def get_duration_seconds(self) -> float | None:
        """Get deployment duration in seconds.

        Returns:
            Duration in seconds or None if not completed
        """
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


def create_deployment_config(
    deployment_id: str,
    model_name: str,
    model_version: str,
    environment: DeploymentEnvironment,
    strategy: DeploymentStrategy = DeploymentStrategy.DIRECT,
    **kwargs: Any,
) -> DeploymentConfig:
    """Create deployment configuration (convenience function).

    Args:
        deployment_id: Unique deployment ID
        model_name: Model name
        model_version: Model version
        environment: Target environment
        strategy: Deployment strategy
        **kwargs: Additional configuration options

    Returns:
        Deployment configuration
    """
    # Create default target based on environment
    target = DeploymentTarget(
        name=f"{environment.value}-target",
        environment=environment,
        host=kwargs.pop("host", "localhost"),
        port=kwargs.pop("port", 8000),
    )

    return DeploymentConfig(
        deployment_id=deployment_id,
        model_name=model_name,
        model_version=model_version,
        strategy=strategy,
        target=target,
        **kwargs,
    )
