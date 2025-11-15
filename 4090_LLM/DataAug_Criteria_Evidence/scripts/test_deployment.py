#!/usr/bin/env python
"""Test script for Phase 30: Deployment Automation & CI/CD Pipelines.

This script validates the deployment automation components including:
- Deployment configuration
- Deployment strategies (direct, blue-green, canary, rolling)
- Health validation
- Rollback mechanisms
- Deployment orchestration

Usage:
    python scripts/test_deployment.py
    make test-deployment
"""

from __future__ import annotations

# Phase 30: Deployment Automation
from psy_agents_noaug.deployment import (
    AutoRollbackMonitor,
    BlueGreenDeploymentStrategy,
    CanaryDeploymentStrategy,
    DeploymentEnvironment,
    DeploymentOrchestrator,
    DeploymentStatus,
    DirectDeploymentStrategy,
    HealthValidator,
    MetricsValidator,
    RollbackManager,
    RollingDeploymentStrategy,
    StrategyEnum,
    create_deployment_config,
    create_health_validator,
    create_orchestrator,
    create_rollback_manager,
)


def test_deployment_config():
    """Test deployment configuration."""
    print("\n=== Testing Deployment Configuration ===")

    # Create deployment config
    config = create_deployment_config(
        deployment_id="test_deploy_001",
        model_name="test_model",
        model_version="v1.0.0",
        environment=DeploymentEnvironment.STAGING,
        strategy=StrategyEnum.BLUE_GREEN,
        host="localhost",
        port=8000,
    )

    print("✓ Created deployment configuration")
    print(f"  - Deployment ID: {config.deployment_id}")
    print(f"  - Model: {config.model_name} v{config.model_version}")
    print(f"  - Environment: {config.target.environment.value}")
    print(f"  - Strategy: {config.strategy.value}")
    print(f"  - Endpoint: {config.target.get_endpoint_url()}")

    # Test configuration dict
    config_dict = config.to_dict()
    assert "deployment_id" in config_dict
    assert config_dict["model_name"] == "test_model"
    assert config_dict["strategy"] == "blue_green"
    print("✓ Configuration serialization works")

    print("\n✅ Deployment configuration tests passed")


def test_deployment_strategies():
    """Test deployment strategies."""
    print("\n=== Testing Deployment Strategies ===")

    from psy_agents_noaug.deployment.deployment_config import DeploymentRecord
    from datetime import datetime

    # Test each strategy
    strategies = [
        (StrategyEnum.DIRECT, "Direct"),
        (StrategyEnum.BLUE_GREEN, "Blue-Green"),
        (StrategyEnum.CANARY, "Canary"),
        (StrategyEnum.ROLLING, "Rolling"),
    ]

    for strategy_enum, strategy_name in strategies:
        print(f"\nTesting {strategy_name} strategy:")

        # Create config
        config = create_deployment_config(
            deployment_id=f"test_{strategy_enum.value}",
            model_name="test_model",
            model_version="v1.0.0",
            environment=DeploymentEnvironment.STAGING,
            strategy=strategy_enum,
        )

        # Create deployment record
        record = DeploymentRecord(
            deployment_id=config.deployment_id,
            config=config,
            status=DeploymentStatus.PENDING,
            started_at=datetime.now(),
        )

        # Execute strategy
        from psy_agents_noaug.deployment.strategies import get_strategy

        strategy = get_strategy(config)
        success = strategy.execute(record)

        print(f"  ✓ {strategy_name} deployment executed")
        print(f"    - Status: {record.status.value}")
        print(f"    - Log entries: {len(record.logs)}")
        print(f"    - Success: {success}")

        assert success, f"{strategy_name} strategy should succeed"
        assert record.status == DeploymentStatus.COMPLETED

    print("\n✅ Deployment strategy tests passed")


def test_health_validation():
    """Test health validation."""
    print("\n=== Testing Health Validation ===")

    # Create health validator
    validator = create_health_validator()
    print("✓ Created health validator")

    # Test single health check
    result = validator.check_health("http://localhost:8000")
    print("✓ Single health check completed")
    print(f"  - Success: {result.success}")
    print(f"  - Response time: {result.response_time_ms:.2f} ms")

    assert result.success, "Health check should succeed"

    # Test deployment validation
    is_healthy = validator.validate_deployment("http://localhost:8000")
    print("✓ Deployment validation completed")
    print(f"  - Deployment healthy: {is_healthy}")

    # Get health stats
    stats = validator.get_health_stats()
    print("✓ Health statistics retrieved")
    print(f"  - Total checks: {stats['total_checks']}")
    print(f"  - Success rate: {stats['success_rate']:.2%}")
    print(f"  - Avg response time: {stats['avg_response_time_ms']:.2f} ms")

    print("\n✅ Health validation tests passed")


def test_metrics_validation():
    """Test metrics validation."""
    print("\n=== Testing Metrics Validation ===")

    validator = MetricsValidator(error_rate_threshold=0.05)
    print("✓ Created metrics validator")

    # Record successful requests
    for _ in range(95):
        validator.record_request(success=True)

    # Record some errors
    for _ in range(3):
        validator.record_request(success=False)

    error_rate = validator.get_error_rate()
    is_acceptable = validator.is_error_rate_acceptable()

    print("✓ Recorded requests")
    print(f"  - Total requests: {validator.total_requests}")
    print(f"  - Error count: {validator.error_count}")
    print(f"  - Error rate: {error_rate:.2%}")
    print(f"  - Acceptable: {is_acceptable}")

    assert error_rate < 0.05, "Error rate should be below threshold"
    assert is_acceptable, "Error rate should be acceptable"

    # Get stats
    stats = validator.get_stats()
    assert stats["acceptable"] is True

    print("\n✅ Metrics validation tests passed")


def test_rollback_mechanism():
    """Test rollback mechanism."""
    print("\n=== Testing Rollback Mechanism ===")

    manager = create_rollback_manager()
    print("✓ Created rollback manager")

    # Capture state
    state = manager.capture_state(
        deployment_id="test_rollback",
        model_name="test_model",
        model_version="v2.0.0",
        previous_version="v1.0.0",
        endpoint_url="http://localhost:8000",
    )

    print("✓ Captured rollback state")
    print(f"  - Current version: {state.model_version}")
    print(f"  - Previous version: {state.previous_version}")

    assert manager.can_rollback(), "Should be able to rollback"

    # Execute rollback
    success = manager.execute_rollback(state, "Test rollback")
    print("✓ Executed rollback")
    print(f"  - Success: {success}")

    assert success, "Rollback should succeed"
    assert manager.get_rollback_count() == 1

    print("\n✅ Rollback mechanism tests passed")


def test_auto_rollback_monitor():
    """Test automatic rollback monitoring."""
    print("\n=== Testing Auto-Rollback Monitor ===")

    rollback_manager = create_rollback_manager()
    monitor = AutoRollbackMonitor(
        rollback_manager=rollback_manager,
        error_rate_threshold=0.1,
        health_check_failures=3,
    )

    print("✓ Created auto-rollback monitor")

    # Simulate successful requests
    for _ in range(8):
        reason = monitor.record_request(success=True)
        assert reason is None, "Should not trigger rollback on success"

    # Simulate errors below threshold
    for _ in range(1):
        reason = monitor.record_request(success=False)
        assert reason is None, "Should not trigger rollback below threshold"

    print("✓ Auto-rollback not triggered (below threshold)")

    # Test health check failures
    for i in range(2):
        reason = monitor.record_health_check(success=False)
        assert reason is None, f"Should not trigger on failure {i + 1}/3"

    print("✓ Monitoring health check failures")

    # Third consecutive failure should trigger
    reason = monitor.record_health_check(success=False)
    assert reason is not None, "Should trigger on 3rd consecutive failure"
    print(f"✓ Auto-rollback triggered: {reason}")

    print("\n✅ Auto-rollback monitor tests passed")


def test_deployment_orchestrator():
    """Test deployment orchestrator."""
    print("\n=== Testing Deployment Orchestrator ===")

    orchestrator = create_orchestrator()
    print("✓ Created deployment orchestrator")

    # Test direct deployment
    config = create_deployment_config(
        deployment_id="orchestrator_test_001",
        model_name="test_model",
        model_version="v1.0.0",
        environment=DeploymentEnvironment.STAGING,
        strategy=StrategyEnum.DIRECT,
    )

    record = orchestrator.deploy(config)
    print("✓ Executed direct deployment")
    print(f"  - Status: {record.status.value}")
    print(f"  - Duration: {record.get_duration_seconds():.2f}s")

    assert record.status == DeploymentStatus.COMPLETED

    # Test blue-green deployment
    config_bg = create_deployment_config(
        deployment_id="orchestrator_test_002",
        model_name="test_model",
        model_version="v2.0.0",
        environment=DeploymentEnvironment.PRODUCTION,
        strategy=StrategyEnum.BLUE_GREEN,
    )

    record_bg = orchestrator.deploy(config_bg, previous_version="v1.0.0")
    print("✓ Executed blue-green deployment")
    print(f"  - Status: {record_bg.status.value}")

    assert record_bg.status == DeploymentStatus.COMPLETED

    # Test deployment listing
    deployments = orchestrator.list_deployments()
    print(f"✓ Listed deployments: {len(deployments)}")

    # Test statistics
    stats = orchestrator.get_deployment_stats()
    print("✓ Deployment statistics:")
    print(f"  - Total: {stats['total_deployments']}")
    print(f"  - Completed: {stats['completed']}")
    print(f"  - Success rate: {stats['success_rate']:.2%}")
    print(f"  - Avg duration: {stats['avg_deployment_time_seconds']:.2f}s")

    assert stats["completed"] == 2
    assert stats["success_rate"] == 1.0

    print("\n✅ Deployment orchestrator tests passed")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Phase 30: Deployment Automation & CI/CD Pipelines - Test Suite")
    print("=" * 70)

    try:
        test_deployment_config()
        test_deployment_strategies()
        test_health_validation()
        test_metrics_validation()
        test_rollback_mechanism()
        test_auto_rollback_monitor()
        test_deployment_orchestrator()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        raise


if __name__ == "__main__":
    main()
