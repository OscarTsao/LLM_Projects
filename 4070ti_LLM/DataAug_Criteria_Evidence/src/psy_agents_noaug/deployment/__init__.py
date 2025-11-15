#!/usr/bin/env python
"""Deployment automation and CI/CD pipelines (Phase 30).

This module provides comprehensive deployment automation including:
- Multiple deployment strategies (direct, blue-green, canary, rolling)
- Health check validation
- Automatic rollback mechanisms
- Deployment orchestration

Key Components:
- DeploymentOrchestrator: Main orchestrator for deployments
- DeploymentConfig: Configuration for deployments
- DeploymentStrategy: Strategy pattern for different deployment types
- HealthValidator: Health check validation
- RollbackManager: Rollback management

Integration:
- Phase 28: Model registry for version management
- Phase 29: Model serving infrastructure

Example:
    ```python
    from psy_agents_noaug.deployment import (
        DeploymentOrchestrator,
        create_deployment_config,
        DeploymentEnvironment,
        StrategyEnum,
    )

    # Create deployment configuration
    config = create_deployment_config(
        deployment_id="deploy_001",
        model_name="sentiment_model",
        model_version="v2.0.0",
        environment=DeploymentEnvironment.PRODUCTION,
        strategy=StrategyEnum.BLUE_GREEN,
        host="api.example.com",
        port=443,
    )

    # Execute deployment
    orchestrator = DeploymentOrchestrator()
    record = orchestrator.deploy(config, previous_version="v1.0.0")

    # Check deployment status
    if record.status == DeploymentStatus.COMPLETED:
        print("Deployment successful!")
    elif record.status == DeploymentStatus.ROLLED_BACK:
        print(f"Deployment rolled back: {record.error_message}")
    ```
"""

from __future__ import annotations

# Configuration
from psy_agents_noaug.deployment.deployment_config import (
    DeploymentConfig,
    DeploymentEnvironment,
    DeploymentRecord,
    DeploymentStatus,
    DeploymentStrategy as StrategyEnum,
    DeploymentTarget,
    HealthCheckConfig,
    ResourceRequirements,
    create_deployment_config,
)

# Health validation
from psy_agents_noaug.deployment.health_validator import (
    HealthCheckResult,
    HealthValidator,
    MetricsValidator,
    create_health_validator,
)

# Orchestration
from psy_agents_noaug.deployment.orchestrator import (
    DeploymentOrchestrator,
    create_orchestrator,
)

# Rollback
from psy_agents_noaug.deployment.rollback import (
    AutoRollbackMonitor,
    RollbackManager,
    RollbackState,
    create_rollback_manager,
)

# Strategies
from psy_agents_noaug.deployment.strategies import (
    BlueGreenDeploymentStrategy,
    CanaryDeploymentStrategy,
    DeploymentStrategy,
    DirectDeploymentStrategy,
    RollingDeploymentStrategy,
    get_strategy,
)

__all__ = [
    # Configuration
    "DeploymentConfig",
    "DeploymentEnvironment",
    "DeploymentRecord",
    "DeploymentStatus",
    "StrategyEnum",
    "DeploymentTarget",
    "HealthCheckConfig",
    "ResourceRequirements",
    "create_deployment_config",
    # Health validation
    "HealthValidator",
    "HealthCheckResult",
    "MetricsValidator",
    "create_health_validator",
    # Orchestration
    "DeploymentOrchestrator",
    "create_orchestrator",
    # Rollback
    "RollbackManager",
    "RollbackState",
    "AutoRollbackMonitor",
    "create_rollback_manager",
    # Strategies
    "DeploymentStrategy",
    "DirectDeploymentStrategy",
    "BlueGreenDeploymentStrategy",
    "CanaryDeploymentStrategy",
    "RollingDeploymentStrategy",
    "get_strategy",
]
