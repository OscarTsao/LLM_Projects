#!/usr/bin/env python
"""Test script for Phase 20: Model Versioning & Registry.

This script tests:
1. Semantic versioning
2. Model version registration
3. Version tagging and retrieval
4. Model metadata tracking
5. Metadata comparison
6. Promotion workflows
7. Approval workflows
8. Model lineage tracking
9. Lineage graph traversal
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from psy_agents_noaug.registry import (
    LineageTracker,
    MetadataManager,
    ModelLineage,
    ModelMetadata,
    ModelPromoter,
    ModelRegistry,
    PromotionCriteria,
    PromotionWorkflow,
    SemanticVersion,
    create_metadata,
    create_version,
    promote_model,
    track_lineage,
)
from psy_agents_noaug.registry.lineage import DataSource, TrainingRun
from psy_agents_noaug.registry.promotion import Stage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def test_semantic_versioning() -> bool:
    """Test semantic versioning.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 1: Semantic Versioning")
    LOGGER.info("=" * 80)

    try:
        # Create versions
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 0, 1)
        v3 = SemanticVersion(1, 1, 0)
        v4 = SemanticVersion(2, 0, 0)

        assert str(v1) == "1.0.0"
        assert v1 < v2 < v3 < v4

        # Parse from string
        v5 = SemanticVersion.from_string("1.2.3")
        assert v5.major == 1
        assert v5.minor == 2
        assert v5.patch == 3

        # Prerelease
        v6 = SemanticVersion.from_string("1.0.0-alpha")
        assert v6.prerelease == "alpha"
        assert str(v6) == "1.0.0-alpha"

        # Bump versions
        v7 = v1.bump_patch()
        assert str(v7) == "1.0.1"
        v8 = v1.bump_minor()
        assert str(v8) == "1.1.0"
        v9 = v1.bump_major()
        assert str(v9) == "2.0.0"

        LOGGER.info("✅ Semantic Versioning: PASSED")
        LOGGER.info("   - Version creation and comparison")
        LOGGER.info("   - String parsing and formatting")
        LOGGER.info("   - Version bumping")

    except Exception:
        LOGGER.exception("❌ Semantic Versioning: FAILED")
        return False
    else:
        return True


def test_model_registry() -> bool:
    """Test model registry.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 2: Model Registry")
    LOGGER.info("=" * 80)

    try:
        registry = ModelRegistry()

        # Register versions
        v1 = registry.register_version(
            model_name="criteria-classifier",
            version="1.0.0",
            stage="dev",
            metrics={"accuracy": 0.85, "f1": 0.82},
        )

        assert v1.model_name == "criteria-classifier"
        assert str(v1.version) == "1.0.0"

        # Register another version
        registry.register_version(
            model_name="criteria-classifier",
            version="1.1.0",
            stage="staging",
            metrics={"accuracy": 0.88, "f1": 0.86},
        )

        # List versions
        versions = registry.list_versions("criteria-classifier")
        assert len(versions) == 2

        # Get specific version
        version = registry.get_version("criteria-classifier", "1.0.0")
        assert version is not None
        assert version.metrics["accuracy"] == 0.85

        # Get latest
        latest = registry.get_latest_version("criteria-classifier")
        assert latest is not None
        assert str(latest.version) == "1.1.0"

        # Tag version
        success = registry.tag_version("criteria-classifier", "1.1.0", "stable")
        assert success is True

        # Get by tag
        stable = registry.get_version("criteria-classifier", tag="stable")
        assert stable is not None
        assert str(stable.version) == "1.1.0"

        LOGGER.info("✅ Model Registry: PASSED")
        LOGGER.info("   - Version registration")
        LOGGER.info("   - Version listing and retrieval")
        LOGGER.info("   - Tagging and tag-based retrieval")

    except Exception:
        LOGGER.exception("❌ Model Registry: FAILED")
        return False
    else:
        return True


def test_metadata_manager() -> bool:
    """Test metadata manager.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 3: Metadata Manager")
    LOGGER.info("=" * 80)

    try:
        manager = MetadataManager()

        # Create metadata
        metadata1 = ModelMetadata(
            model_name="criteria-classifier",
            model_type="binary-classification",
            version="1.0.0",
            architecture="roberta-base",
            trained_at=datetime.now(),
            training_duration_seconds=3600.0,
            total_parameters=125000000,
            trainable_parameters=125000000,
            metrics={"accuracy": 0.85, "f1": 0.82},
            validation_metrics={"accuracy": 0.83, "f1": 0.80},
            hyperparameters={"learning_rate": 2e-5, "batch_size": 32},
        )

        manager.register_metadata(metadata1)

        # Retrieve metadata
        retrieved = manager.get_metadata("criteria-classifier", "1.0.0")
        assert retrieved is not None
        assert retrieved.architecture == "roberta-base"

        # Create another version
        metadata2 = ModelMetadata(
            model_name="criteria-classifier",
            model_type="binary-classification",
            version="1.1.0",
            architecture="roberta-base",
            trained_at=datetime.now(),
            training_duration_seconds=3200.0,
            total_parameters=125000000,
            trainable_parameters=125000000,
            metrics={"accuracy": 0.88, "f1": 0.86},
            validation_metrics={"accuracy": 0.86, "f1": 0.84},
            hyperparameters={"learning_rate": 1e-5, "batch_size": 32},
        )

        manager.register_metadata(metadata2)

        # Compare metadata
        comparison = manager.compare_metadata(
            "criteria-classifier:1.0.0",
            "criteria-classifier:1.1.0",
        )

        assert "metrics" in comparison
        assert "accuracy" in comparison["metrics"]
        assert (
            comparison["metrics"]["accuracy"]["model2"]
            > comparison["metrics"]["accuracy"]["model1"]
        )

        # Get best model
        best = manager.get_best_model("criteria-classifier", "accuracy")
        assert best is not None
        assert best.version == "1.1.0"

        LOGGER.info("✅ Metadata Manager: PASSED")
        LOGGER.info("   - Metadata registration and retrieval")
        LOGGER.info("   - Metadata comparison")
        LOGGER.info("   - Best model selection")

    except Exception:
        LOGGER.exception("❌ Metadata Manager: FAILED")
        return False
    else:
        return True


def test_promotion_workflow() -> bool:
    """Test promotion workflow.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 4: Promotion Workflow")
    LOGGER.info("=" * 80)

    try:
        workflow = PromotionWorkflow()

        # Test promotion criteria
        metadata = {
            "validation_metrics": {"accuracy": 0.75, "f1": 0.72},
            "test_samples": 150,
            "tags": ["tested", "validated"],
        }

        # Check if can promote dev → staging
        can_promote, errors = workflow.can_promote(
            Stage.DEV,
            Stage.STAGING,
            metadata,
        )
        assert can_promote is True

        # Promote model
        success, record = workflow.promote(
            model_name="criteria-classifier",
            version="1.0.0",
            from_stage=Stage.DEV,
            to_stage=Stage.STAGING,
            metadata=metadata,
        )

        assert success is True
        assert record.from_stage == Stage.DEV
        assert record.to_stage == Stage.STAGING

        # Test promotion to production (requires approval)
        metadata_prod = {
            "validation_metrics": {"accuracy": 0.88, "f1": 0.86},
            "test_samples": 600,
            "tags": ["tested", "validated"],
        }

        success, record = workflow.promote(
            model_name="criteria-classifier",
            version="1.1.0",
            from_stage=Stage.STAGING,
            to_stage=Stage.PRODUCTION,
            metadata=metadata_prod,
        )

        # Should be pending approval
        assert record.approval_status == "pending"

        # Approve promotion
        approved = workflow.approve_promotion(
            "criteria-classifier",
            "1.1.0",
            "admin",
        )
        assert approved is True

        # Get promotion history
        history = workflow.get_promotion_history("criteria-classifier")
        assert len(history) == 2

        LOGGER.info("✅ Promotion Workflow: PASSED")
        LOGGER.info("   - Promotion criteria validation")
        LOGGER.info("   - Model promotion")
        LOGGER.info("   - Approval workflow")

    except Exception:
        LOGGER.exception("❌ Promotion Workflow: FAILED")
        return False
    else:
        return True


def test_promotion_criteria() -> bool:
    """Test promotion criteria validation.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 5: Promotion Criteria")
    LOGGER.info("=" * 80)

    try:
        criteria = PromotionCriteria(
            min_accuracy=0.85,
            min_f1=0.80,
            min_test_samples=500,
            required_tags=["tested"],
        )

        # Test passing criteria
        good_metadata = {
            "validation_metrics": {"accuracy": 0.88, "f1": 0.85},
            "test_samples": 600,
            "tags": ["tested", "validated"],
        }

        is_valid, errors = criteria.validate(good_metadata)
        assert is_valid is True

        # Test failing criteria
        bad_metadata = {
            "validation_metrics": {"accuracy": 0.75, "f1": 0.72},
            "test_samples": 200,
            "tags": ["validated"],
        }

        is_valid, errors = criteria.validate(bad_metadata)
        assert is_valid is False
        assert len(errors) > 0

        LOGGER.info("✅ Promotion Criteria: PASSED")
        LOGGER.info("   - Criteria validation")
        LOGGER.info(f"   - Error detection ({len(errors)} errors for bad metadata)")

    except Exception:
        LOGGER.exception("❌ Promotion Criteria: FAILED")
        return False
    else:
        return True


def test_lineage_tracking() -> bool:
    """Test lineage tracking.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 6: Lineage Tracking")
    LOGGER.info("=" * 80)

    try:
        tracker = LineageTracker()

        # Create lineage
        lineage = ModelLineage(
            model_name="criteria-classifier",
            version="1.0.0",
            created_at=datetime.now(),
        )

        # Add data sources
        lineage.data_sources.append(
            DataSource(
                name="redsm5",
                version="1.0.0",
                path="/data/redsm5",
                checksum="abc123",
                num_samples=1000,
            )
        )

        # Add training run
        lineage.training_run = TrainingRun(
            run_id="run-123",
            experiment_id="exp-456",
            started_at=datetime.now() - timedelta(hours=2),
            ended_at=datetime.now(),
            duration_seconds=7200.0,
            hyperparameters={"learning_rate": 2e-5},
            metrics={"accuracy": 0.85},
        )

        # Set code lineage
        lineage.git_commit = "abc123def456"
        lineage.git_branch = "main"

        tracker.track(lineage)

        # Retrieve lineage
        retrieved = tracker.get_lineage("criteria-classifier", "1.0.0")
        assert retrieved is not None
        assert len(retrieved.data_sources) == 1
        assert retrieved.git_commit == "abc123def456"

        # Create child model
        child_lineage = ModelLineage(
            model_name="criteria-classifier",
            version="1.1.0",
            created_at=datetime.now(),
            parent_models=["criteria-classifier:1.0.0"],
        )

        tracker.track(child_lineage)

        # Update parent with derived model
        lineage.derived_models.append("criteria-classifier:1.1.0")

        # Get ancestors
        ancestors = tracker.get_ancestors("criteria-classifier", "1.1.0")
        assert len(ancestors) == 1

        # Get lineage graph
        graph = tracker.get_lineage_graph("criteria-classifier", "1.0.0")
        assert "current" in graph
        assert "descendants" in graph

        LOGGER.info("✅ Lineage Tracking: PASSED")
        LOGGER.info("   - Lineage creation and tracking")
        LOGGER.info("   - Data source tracking")
        LOGGER.info("   - Parent/child relationships")
        LOGGER.info("   - Lineage graph generation")

    except Exception:
        LOGGER.exception("❌ Lineage Tracking: FAILED")
        return False
    else:
        return True


def test_lineage_queries() -> bool:
    """Test lineage query capabilities.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 7: Lineage Queries")
    LOGGER.info("=" * 80)

    try:
        tracker = LineageTracker()

        # Create lineages with different data sources
        lineage1 = ModelLineage(
            model_name="model-a",
            version="1.0.0",
            created_at=datetime.now(),
            git_commit="commit123",
        )
        lineage1.data_sources.append(
            DataSource(
                name="dataset-v1",
                version="1.0.0",
                path="/data/v1",
                checksum="abc",
                num_samples=1000,
            )
        )

        lineage2 = ModelLineage(
            model_name="model-b",
            version="1.0.0",
            created_at=datetime.now(),
            git_commit="commit123",
        )
        lineage2.data_sources.append(
            DataSource(
                name="dataset-v1",
                version="1.0.0",
                path="/data/v1",
                checksum="abc",
                num_samples=1000,
            )
        )

        tracker.track(lineage1)
        tracker.track(lineage2)

        # Find by data source
        models = tracker.find_by_data_source("dataset-v1")
        assert len(models) == 2

        # Find by git commit
        models = tracker.find_by_git_commit("commit123")
        assert len(models) == 2

        LOGGER.info("✅ Lineage Queries: PASSED")
        LOGGER.info("   - Find by data source")
        LOGGER.info("   - Find by git commit")

    except Exception:
        LOGGER.exception("❌ Lineage Queries: FAILED")
        return False
    else:
        return True


def test_model_promoter() -> bool:
    """Test model promoter.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 8: Model Promoter")
    LOGGER.info("=" * 80)

    try:
        promoter = ModelPromoter()

        # Register hook
        hook_called = {"count": 0}

        def promotion_hook(model_name: str, version: str) -> None:
            hook_called["count"] += 1
            LOGGER.info(f"Hook called: {model_name}:{version}")

        promoter.register_hook(Stage.DEV, Stage.STAGING, promotion_hook)

        # Promote model
        metadata = {
            "validation_metrics": {"accuracy": 0.88, "f1": 0.86},
            "test_samples": 600,
            "tags": ["tested"],
        }

        success = promoter.promote_model(
            model_name="test-model",
            version="1.0.0",
            to_stage=Stage.STAGING,
            metadata=metadata,
            current_stage=Stage.DEV,
        )

        assert success is True
        assert hook_called["count"] == 1

        LOGGER.info("✅ Model Promoter: PASSED")
        LOGGER.info("   - Model promotion with hooks")
        LOGGER.info(f"   - Hook execution ({hook_called['count']} calls)")

    except Exception:
        LOGGER.exception("❌ Model Promoter: FAILED")
        return False
    else:
        return True


def test_convenience_functions() -> bool:
    """Test convenience functions.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 9: Convenience Functions")
    LOGGER.info("=" * 80)

    try:
        # Test create_version
        version = create_version("2.1.3")
        assert version.major == 2
        assert version.minor == 1
        assert version.patch == 3

        # Test create_metadata
        metadata = create_metadata(
            model_name="test",
            model_type="classifier",
            version="1.0.0",
            architecture="transformer",
            training_duration=3600.0,
            total_params=1000000,
            trainable_params=1000000,
        )
        assert metadata.model_name == "test"

        # Test track_lineage
        lineage = track_lineage(
            model_name="test",
            version="1.0.0",
            data_sources=[
                {
                    "name": "data",
                    "path": "/data",
                    "num_samples": 1000,
                }
            ],
            git_commit="abc123",
        )
        assert lineage.git_commit == "abc123"
        assert len(lineage.data_sources) == 1

        # Test promote_model
        success = promote_model(
            model_name="test",
            version="1.0.0",
            to_stage="staging",
            metadata={
                "validation_metrics": {"accuracy": 0.88},
                "test_samples": 600,
            },
        )
        assert success is True

        LOGGER.info("✅ Convenience Functions: PASSED")
        LOGGER.info("   - create_version")
        LOGGER.info("   - create_metadata")
        LOGGER.info("   - track_lineage")
        LOGGER.info("   - promote_model")

    except Exception:
        LOGGER.exception("❌ Convenience Functions: FAILED")
        return False
    else:
        return True


def main():
    """Run all registry tests."""
    LOGGER.info("Starting Phase 20 Model Registry Tests")
    LOGGER.info("=" * 80)

    # Run tests
    tests = [
        ("Semantic Versioning", test_semantic_versioning),
        ("Model Registry", test_model_registry),
        ("Metadata Manager", test_metadata_manager),
        ("Promotion Workflow", test_promotion_workflow),
        ("Promotion Criteria", test_promotion_criteria),
        ("Lineage Tracking", test_lineage_tracking),
        ("Lineage Queries", test_lineage_queries),
        ("Model Promoter", test_model_promoter),
        ("Convenience Functions", test_convenience_functions),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception:
            LOGGER.exception(f"Test '{test_name}' crashed")
            results.append((test_name, False))

    # Summary
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("TEST SUMMARY")
    LOGGER.info("=" * 80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        LOGGER.info(f"{status}: {test_name}")

    LOGGER.info("=" * 80)
    LOGGER.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        LOGGER.info("🎉 All tests passed!")
        return 0

    LOGGER.error(f"❌ {total - passed} test(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
