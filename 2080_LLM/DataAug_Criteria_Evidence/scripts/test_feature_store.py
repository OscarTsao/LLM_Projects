#!/usr/bin/env python
"""Test script for Phase 24: Feature Store & Engineering.

This script tests:
1. Feature registry (registration, groups, search)
2. Feature versioning (lifecycle, lineage)
3. Feature computation (caching, dependencies)
4. Feature serving (online, batch, vectors)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from psy_agents_noaug.feature_store import (
    FeatureComputationEngine,
    FeatureRegistry,
    FeatureServer,
    FeatureSet,
    FeatureType,
    FeatureVersionManager,
    VersionStatus,
    compute_features,
    serve_features,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def test_feature_registry() -> bool:
    """Test feature registry."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 1: Feature Registry")
    LOGGER.info("=" * 80)

    try:
        registry = FeatureRegistry()

        # Register group
        registry.register_group(
            "user_features",
            "User demographic features",
        )

        # Register features
        registry.register_feature(
            name="age",
            feature_type=FeatureType.NUMERICAL,
            description="User age in years",
            group="user_features",
            tags=["demographic"],
        )

        registry.register_feature(
            name="gender",
            feature_type=FeatureType.CATEGORICAL,
            description="User gender",
            group="user_features",
            tags=["demographic"],
        )

        # Test retrieval
        retrieved_age = registry.get_feature("age")
        assert retrieved_age is not None
        assert retrieved_age.name == "age"
        assert retrieved_age.feature_type == FeatureType.NUMERICAL

        # Test listing
        numerical_features = registry.list_features(feature_type=FeatureType.NUMERICAL)
        assert len(numerical_features) == 1
        assert numerical_features[0].name == "age"

        # Test search
        results = registry.search_features("age")
        assert len(results) >= 1

        # Test statistics
        stats = registry.get_statistics()
        assert stats["total_features"] == 2
        assert stats["total_groups"] == 1

        LOGGER.info("✅ Feature Registry: PASSED")
        LOGGER.info(f"   - Features registered: {stats['total_features']}")
        LOGGER.info(f"   - Groups: {stats['total_groups']}")

    except Exception:
        LOGGER.exception("❌ Feature Registry: FAILED")
        return False
    else:
        return True


def test_feature_groups() -> bool:
    """Test feature groups."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 2: Feature Groups")
    LOGGER.info("=" * 80)

    try:
        registry = FeatureRegistry()

        # Create group
        registry.register_group(
            "text_features",
            "Text-based features",
        )

        # Add features
        registry.register_feature(
            name="word_count",
            feature_type=FeatureType.NUMERICAL,
            description="Number of words",
            group="text_features",
        )

        registry.register_feature(
            name="sentiment",
            feature_type=FeatureType.NUMERICAL,
            description="Sentiment score",
            group="text_features",
        )

        # Test group features
        text_features = registry.list_features(group="text_features")
        assert len(text_features) == 2

        feature_names = [f.name for f in text_features]
        assert "word_count" in feature_names
        assert "sentiment" in feature_names

        LOGGER.info("✅ Feature Groups: PASSED")
        LOGGER.info(f"   - Features in group: {len(text_features)}")

    except Exception:
        LOGGER.exception("❌ Feature Groups: FAILED")
        return False
    else:
        return True


def test_feature_versioning() -> bool:
    """Test feature versioning."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 3: Feature Versioning")
    LOGGER.info("=" * 80)

    try:
        manager = FeatureVersionManager()

        # Create versions
        v1 = manager.create_version(
            "embedding",
            "1.0.0",
            "Initial embedding model",
        )

        manager.create_version(
            "embedding",
            "2.0.0",
            "Improved embedding with larger model",
            parent_version="1.0.0",
            breaking_changes=True,
        )

        # Test retrieval
        retrieved_v1 = manager.get_version("embedding", "1.0.0")
        assert retrieved_v1 is not None
        assert retrieved_v1.version == "1.0.0"

        # Test activation
        success = manager.activate_version("embedding", "1.0.0")
        assert success
        assert v1.status == VersionStatus.ACTIVE

        # Test latest version
        latest = manager.get_latest_version("embedding")
        assert latest is not None
        assert latest.version == "1.0.0"  # v2 is still DRAFT

        # Activate v2
        manager.activate_version("embedding", "2.0.0")

        # Test lineage
        lineage = manager.get_version_lineage("embedding", "2.0.0")
        assert lineage["ancestors"] == ["1.0.0"]
        assert lineage["depth"] == 1

        LOGGER.info("✅ Feature Versioning: PASSED")
        LOGGER.info("   - Versions created: 2")
        LOGGER.info(f"   - Lineage depth: {lineage['depth']}")

    except Exception:
        LOGGER.exception("❌ Feature Versioning: FAILED")
        return False
    else:
        return True


def test_version_lifecycle() -> bool:
    """Test version lifecycle transitions."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 4: Version Lifecycle")
    LOGGER.info("=" * 80)

    try:
        manager = FeatureVersionManager()

        # Create version
        version = manager.create_version(
            "model_score",
            "1.0.0",
            "Initial model score",
        )

        # Test transitions
        assert version.status == VersionStatus.DRAFT

        # Activate
        success = manager.activate_version("model_score", "1.0.0")
        assert success
        assert version.status == VersionStatus.ACTIVE

        # Deprecate
        success = manager.deprecate_version("model_score", "1.0.0")
        assert success
        assert version.status == VersionStatus.DEPRECATED

        # Archive
        success = manager.archive_version("model_score", "1.0.0")
        assert success
        assert version.status == VersionStatus.ARCHIVED

        # Cannot transition from archived
        success = manager.activate_version("model_score", "1.0.0")
        assert not success

        LOGGER.info("✅ Version Lifecycle: PASSED")
        LOGGER.info(f"   - Final status: {version.status.value}")

    except Exception:
        LOGGER.exception("❌ Version Lifecycle: FAILED")
        return False
    else:
        return True


def test_feature_computation() -> bool:
    """Test feature computation engine."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 5: Feature Computation")
    LOGGER.info("=" * 80)

    try:
        engine = FeatureComputationEngine(enable_cache=True)

        # Define computation functions
        def compute_length(text: str) -> int:
            return len(text)

        def compute_word_count(text: str) -> int:
            return len(text.split())

        # Compute features
        text = "This is a test sentence for feature computation"

        result_length = engine.compute_single("length", compute_length, text)
        assert result_length.value == len(text)
        assert not result_length.cached

        # Compute again (should be cached)
        result_length_2 = engine.compute_single("length", compute_length, text)
        assert result_length_2.cached

        result_words = engine.compute_single("word_count", compute_word_count, text)
        assert result_words.value == 8

        # Test statistics
        stats = engine.get_statistics()
        assert stats["cached_results"] >= 2

        LOGGER.info("✅ Feature Computation: PASSED")
        LOGGER.info(f"   - Length: {result_length.value}")
        LOGGER.info(f"   - Words: {result_words.value}")
        LOGGER.info(f"   - Cache hit rate: {stats['cache_hit_rate']:.2%}")

    except Exception:
        LOGGER.exception("❌ Feature Computation: FAILED")
        return False
    else:
        return True


def test_computation_dependencies() -> bool:
    """Test feature computation with dependencies."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 6: Computation Dependencies")
    LOGGER.info("=" * 80)

    try:
        engine = FeatureComputationEngine()

        # Register dependencies
        engine.register_feature("text_length", dependencies=[])
        engine.register_feature(
            "avg_word_length", dependencies=["text_length", "word_count"]
        )
        engine.register_feature("word_count", dependencies=[])

        # Resolve dependencies
        order = engine.resolve_dependencies("avg_word_length")

        # Should compute text_length and word_count before avg_word_length
        assert "text_length" in order
        assert "word_count" in order
        assert "avg_word_length" in order
        assert order.index("text_length") < order.index("avg_word_length")
        assert order.index("word_count") < order.index("avg_word_length")

        LOGGER.info("✅ Computation Dependencies: PASSED")
        LOGGER.info(f"   - Computation order: {' -> '.join(order)}")

    except Exception:
        LOGGER.exception("❌ Computation Dependencies: FAILED")
        return False
    else:
        return True


def test_feature_serving() -> bool:
    """Test feature serving."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 7: Feature Serving")
    LOGGER.info("=" * 80)

    try:
        server = FeatureServer(enable_cache=True)

        # Register feature set
        feature_set = FeatureSet(
            name="user_model_features",
            features=["age", "gender_encoded", "activity_score"],
            description="Features for user model",
        )
        server.register_feature_set(feature_set)

        # Serve features
        from psy_agents_noaug.feature_store.serving import ServingRequest

        request = ServingRequest(
            feature_set="user_model_features",
            entity_id="user_123",
        )

        features_data = {
            "age": 25,
            "gender_encoded": 1,
            "activity_score": 0.75,
            "extra_feature": 999,  # Should be filtered out
        }

        response = server.serve_online(request, features_data)

        assert response.feature_set == "user_model_features"
        assert len(response.features) == 3
        assert "age" in response.features
        assert "extra_feature" not in response.features
        assert not response.cached

        # Serve again (should be cached)
        response2 = server.serve_online(request, features_data)
        assert response2.cached

        # Test statistics
        stats = server.get_serving_stats()
        assert stats["total_requests"] == 2

        LOGGER.info("✅ Feature Serving: PASSED")
        LOGGER.info(f"   - Features served: {len(response.features)}")
        LOGGER.info(f"   - Latency: {response.latency_ms:.2f}ms")
        LOGGER.info(f"   - Total requests: {stats['total_requests']}")

    except Exception:
        LOGGER.exception("❌ Feature Serving: FAILED")
        return False
    else:
        return True


def test_batch_serving() -> bool:
    """Test batch feature serving."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 8: Batch Serving")
    LOGGER.info("=" * 80)

    try:
        server = FeatureServer()

        # Register feature set
        feature_set = FeatureSet(
            name="batch_features",
            features=["score_1", "score_2"],
        )
        server.register_feature_set(feature_set)

        # Batch data
        batch_data = [
            {"score_1": 0.8, "score_2": 0.6},
            {"score_1": 0.9, "score_2": 0.7},
            {"score_1": 0.7, "score_2": 0.5},
        ]

        # Serve batch
        responses = server.serve_batch("batch_features", batch_data)

        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert response.features["score_1"] == batch_data[i]["score_1"]
            assert response.features["score_2"] == batch_data[i]["score_2"]

        LOGGER.info("✅ Batch Serving: PASSED")
        LOGGER.info(f"   - Batch size: {len(responses)}")

    except Exception:
        LOGGER.exception("❌ Batch Serving: FAILED")
        return False
    else:
        return True


def test_feature_vectors() -> bool:
    """Test feature vector generation."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 9: Feature Vectors")
    LOGGER.info("=" * 80)

    try:
        server = FeatureServer()

        # Register feature set
        feature_set = FeatureSet(
            name="model_input",
            features=["feature_1", "feature_2", "feature_3"],
        )
        server.register_feature_set(feature_set)

        # Features data
        features_data = {
            "feature_1": 1.5,
            "feature_2": 2.3,
            "feature_3": 0.8,
        }

        # Get feature vector
        vector = server.get_feature_vector("model_input", features_data)

        assert isinstance(vector, np.ndarray)
        assert len(vector) == 3
        assert vector[0] == 1.5
        assert vector[1] == 2.3
        assert vector[2] == 0.8

        LOGGER.info("✅ Feature Vectors: PASSED")
        LOGGER.info(f"   - Vector shape: {vector.shape}")
        LOGGER.info(f"   - Vector: {vector}")

    except Exception:
        LOGGER.exception("❌ Feature Vectors: FAILED")
        return False
    else:
        return True


def test_convenience_functions() -> bool:
    """Test convenience functions."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 10: Convenience Functions")
    LOGGER.info("=" * 80)

    try:
        # Test compute_features
        def add_one(x: int) -> int:
            return x + 1

        def multiply_two(x: int) -> int:
            return x * 2

        compute_fns = {
            "plus_one": add_one,
            "times_two": multiply_two,
        }

        results = compute_features(["plus_one", "times_two"], compute_fns, 5)

        assert results["plus_one"] == 6
        assert results["times_two"] == 10

        # Test serve_features
        feature_sets = {
            "test_set": FeatureSet(
                name="test_set",
                features=["a", "b"],
            )
        }

        features_data = {"a": 1, "b": 2, "c": 3}
        served = serve_features("test_set", features_data, feature_sets)

        assert len(served) == 2
        assert served["a"] == 1
        assert served["b"] == 2
        assert "c" not in served

        LOGGER.info("✅ Convenience Functions: PASSED")
        LOGGER.info("   - compute_features: OK")
        LOGGER.info("   - serve_features: OK")

    except Exception:
        LOGGER.exception("❌ Convenience Functions: FAILED")
        return False
    else:
        return True


def main():
    """Run all feature store tests."""
    LOGGER.info("Starting Phase 24 Feature Store Tests")
    LOGGER.info("=" * 80)

    tests = [
        ("Feature Registry", test_feature_registry),
        ("Feature Groups", test_feature_groups),
        ("Feature Versioning", test_feature_versioning),
        ("Version Lifecycle", test_version_lifecycle),
        ("Feature Computation", test_feature_computation),
        ("Computation Dependencies", test_computation_dependencies),
        ("Feature Serving", test_feature_serving),
        ("Batch Serving", test_batch_serving),
        ("Feature Vectors", test_feature_vectors),
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
