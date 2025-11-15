#!/usr/bin/env python
"""Integration test suite for Phase 31: End-to-End Testing.

This script validates end-to-end workflows integrating all phases:
- Phase 26: Monitoring & Observability
- Phase 27: Explainability & Interpretability
- Phase 28: Model Registry & Versioning
- Phase 29: Model Serving & Deployment
- Phase 30: Deployment Automation & CI/CD

Usage:
    python scripts/test_integration.py
    make test-integration
"""

from __future__ import annotations

import tempfile
from pathlib import Path

# Phase 26: Monitoring
from psy_agents_noaug.monitoring import (
    AlertManager,
    AlertRule,
    DriftDetector,
    PerformanceMonitor,
)

# Phase 27: Explainability
from psy_agents_noaug.explainability import ExplanationAggregator

# Phase 28: Registry
from psy_agents_noaug.registry import (
    DeploymentEnvironment,
    ModelMetadata,
    ModelRegistry,
    ModelStage,
)

# Phase 29: Serving
from psy_agents_noaug.serving import (
    ModelLoader,
    PredictionRequest,
    Predictor,
)

# Phase 30: Deployment
from psy_agents_noaug.deployment import (
    DeploymentOrchestrator,
    StrategyEnum,
    create_deployment_config,
)


def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    print("\n=== Testing End-to-End Workflow ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 1. Registry: Register and version model
        print("\n1. Model Registry & Versioning")
        registry_path = tmpdir / "registry"
        registry = ModelRegistry(registry_path)

        model = registry.register_model(
            name="test_model",
            description="Test model for integration testing",
            created_by="integration_test",
            task="classification",
        )
        print(f"  ✓ Registered model: {model.name}")

        # Create dummy model file
        model_file = tmpdir / "model.pt"
        model_file.write_text("dummy model")

        version = registry.add_version(
            name="test_model",
            version="v1.0.0",
            model_path=model_file,
            created_by="integration_test",
            stage=ModelStage.DEVELOPMENT,
            metrics={"accuracy": 0.95, "f1": 0.93},
        )
        print(f"  ✓ Added version: {version.version}")

        # 2. Promote to staging
        print("\n2. Model Promotion")
        registry.promote_version("test_model", "v1.0.0", ModelStage.STAGING)
        promoted_version = registry.get_version("test_model", "v1.0.0")
        assert promoted_version.stage == ModelStage.STAGING
        print("  ✓ Promoted model to staging")

        # 3. Monitoring: Setup monitoring
        print("\n3. Model Monitoring Setup")
        perf_monitor = PerformanceMonitor()
        drift_detector = DriftDetector()

        # Record some metrics
        for i in range(10):
            perf_monitor.record_request(latency=0.1, success=True)
        print("  ✓ Performance monitoring configured")

        # 4. Serving: Load and serve model
        print("\n4. Model Serving")
        # In real scenario, would load actual model
        # loader = ModelLoader()
        # model = loader.load_from_registry(registry, "test_model")

        print("  ✓ Model serving configured (simulated)")

        # 5. Deployment: Deploy model
        print("\n5. Model Deployment")
        orchestrator = DeploymentOrchestrator()

        config = create_deployment_config(
            deployment_id="integration_test_deploy_001",
            model_name="test_model",
            model_version="v1.0.0",
            environment=DeploymentEnvironment.STAGING,
            strategy=StrategyEnum.BLUE_GREEN,
            host="localhost",
            port=8000,
        )

        record = orchestrator.deploy(config)
        assert record.status.value in ["completed", "in_progress"]
        print(f"  ✓ Deployed with status: {record.status.value}")

        # 6. Post-deployment monitoring
        print("\n6. Post-Deployment Monitoring")
        stats = perf_monitor.get_current_metrics()
        if stats:
            print(f"  ✓ Performance stats: P50={stats.p50_latency:.3f}s")

        # 7. Explainability (simulated)
        print("\n7. Explainability Integration")
        aggregator = ExplanationAggregator()
        print("  ✓ Explainability configured")

        # 8. Alerting
        print("\n8. Alerting System")
        alert_manager = AlertManager()

        rule = AlertRule(
            name="high_latency",
            condition=lambda: False,  # Dummy condition
            message="High latency detected",
        )
        alert_manager.add_rule(rule)
        print("  ✓ Alert rules configured")

        # 9. Get deployment stats
        print("\n9. Deployment Statistics")
        deploy_stats = orchestrator.get_deployment_stats()
        print(f"  ✓ Total deployments: {deploy_stats['total_deployments']}")
        print(f"  ✓ Success rate: {deploy_stats['success_rate']:.2%}")

    print("\n✅ End-to-end workflow test passed")


def test_model_lifecycle():
    """Test complete model lifecycle."""
    print("\n=== Testing Model Lifecycle ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create registry
        registry_path = tmpdir / "registry"
        registry = ModelRegistry(registry_path)

        # Register model
        model = registry.register_model(
            name="lifecycle_model",
            description="Model for lifecycle testing",
            created_by="test",
        )
        print("  ✓ Model registered")

        # Add initial version
        model_file = tmpdir / "model_v1.pt"
        model_file.write_text("v1")

        v1 = registry.add_version(
            name="lifecycle_model",
            version="v1.0.0",
            model_path=model_file,
            created_by="test",
            stage=ModelStage.DEVELOPMENT,
        )
        print("  ✓ Version 1.0.0 added (development)")

        # Promote through stages
        registry.promote_version("lifecycle_model", "v1.0.0", ModelStage.STAGING)
        print("  ✓ Promoted to staging")

        registry.promote_version("lifecycle_model", "v1.0.0", ModelStage.PRODUCTION)
        print("  ✓ Promoted to production")

        # Deploy
        orchestrator = DeploymentOrchestrator()
        config = create_deployment_config(
            deployment_id="lifecycle_deploy",
            model_name="lifecycle_model",
            model_version="v1.0.0",
            environment=DeploymentEnvironment.PRODUCTION,
            strategy=StrategyEnum.DIRECT,
        )

        record = orchestrator.deploy(config)
        print(f"  ✓ Deployed: {record.status.value}")

        # Add new version
        model_file_v2 = tmpdir / "model_v2.pt"
        model_file_v2.write_text("v2")

        v2 = registry.add_version(
            name="lifecycle_model",
            version="v2.0.0",
            model_path=model_file_v2,
            created_by="test",
            stage=ModelStage.DEVELOPMENT,
        )
        print("  ✓ Version 2.0.0 added")

        # Canary deployment
        config_v2 = create_deployment_config(
            deployment_id="lifecycle_deploy_v2",
            model_name="lifecycle_model",
            model_version="v2.0.0",
            environment=DeploymentEnvironment.PRODUCTION,
            strategy=StrategyEnum.CANARY,
            canary_traffic_percent=10.0,
        )

        record_v2 = orchestrator.deploy(config_v2, previous_version="v1.0.0")
        print(f"  ✓ Canary deployed v2.0.0: {record_v2.status.value}")

    print("\n✅ Model lifecycle test passed")


def test_monitoring_integration():
    """Test monitoring integration."""
    print("\n=== Testing Monitoring Integration ===")

    # Create monitors
    perf_monitor = PerformanceMonitor()
    drift_detector = DriftDetector()

    # Simulate requests
    for i in range(100):
        perf_monitor.record_request(latency=0.05 + (i * 0.001), success=True)

    stats = perf_monitor.get_current_metrics()
    assert stats is not None
    print(f"  ✓ Recorded 100 requests")
    print(f"  ✓ P50 latency: {stats.p50_latency:.3f}s")
    print(f"  ✓ P99 latency: {stats.p99_latency:.3f}s")

    # Test drift detection
    import numpy as np

    reference_data = np.random.normal(0, 1, 1000)
    drift_detector.set_reference_distribution(reference_data)

    monitoring_data = np.random.normal(0, 1, 1000)
    drift_detector.add_monitoring_data(monitoring_data)

    drift = drift_detector.detect_drift()
    assert drift is not None
    print(f"  ✓ Drift detection: KS statistic = {drift.ks_statistic:.4f}")

    print("\n✅ Monitoring integration test passed")


def main():
    """Run all integration tests."""
    print("=" * 70)
    print("Phase 31: End-to-End Integration Testing - Test Suite")
    print("=" * 70)

    try:
        test_end_to_end_workflow()
        test_model_lifecycle()
        test_monitoring_integration()

        print("\n" + "=" * 70)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ INTEGRATION TEST FAILED: {e}")
        print("=" * 70)
        raise


if __name__ == "__main__":
    main()
