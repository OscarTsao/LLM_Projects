#!/usr/bin/env python
"""Test script for Phase 21: A/B Testing & Experimentation.

This script tests:
1. Traffic splitting strategies
2. Experiment lifecycle management
3. Statistical significance testing
4. Metric tracking and aggregation
5. Conversion rate analysis
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from psy_agents_noaug.ab_testing import (
    BayesianAnalyzer,
    Experiment,
    ExperimentConfig,
    ExperimentManager,
    ExperimentTracker,
    FrequentistAnalyzer,
    MetricAggregator,
    MetricEvent,
    SplitStrategy,
    TrafficAllocation,
    TrafficSplitter,
    Variant,
    calculate_sample_size,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def test_traffic_splitting() -> bool:
    """Test traffic splitting strategies."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 1: Traffic Splitting")
    LOGGER.info("=" * 80)

    try:
        # Test uniform split
        splitter = TrafficSplitter(strategy=SplitStrategy.UNIFORM, seed=42)
        allocations = [
            TrafficAllocation(variant_id="control", weight=0.5),
            TrafficAllocation(variant_id="treatment", weight=0.5),
        ]
        splitter.configure_allocation("exp-1", allocations)

        # Assign 100 users
        assignments = {}
        for i in range(100):
            variant = splitter.assign_variant("exp-1", user_id=f"user-{i}")
            assignments[variant] = assignments.get(variant, 0) + 1

        # Should be roughly equal
        assert abs(assignments.get("control", 0) - 50) < 20
        assert abs(assignments.get("treatment", 0) - 50) < 20

        # Test sticky sessions
        sticky_splitter = TrafficSplitter(strategy=SplitStrategy.STICKY, seed=42)
        sticky_splitter.configure_allocation("exp-2", allocations)

        variant1 = sticky_splitter.assign_variant("exp-2", user_id="user-123")
        variant2 = sticky_splitter.assign_variant("exp-2", user_id="user-123")
        assert variant1 == variant2

        LOGGER.info("✅ Traffic Splitting: PASSED")
        LOGGER.info(f"   - Uniform split: {assignments}")
        LOGGER.info("   - Sticky sessions: consistent assignment")

    except Exception:
        LOGGER.exception("❌ Traffic Splitting: FAILED")
        return False
    else:
        return True


def test_experiment_lifecycle() -> bool:
    """Test experiment lifecycle management."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 2: Experiment Lifecycle")
    LOGGER.info("=" * 80)

    try:
        config = ExperimentConfig(
            name="Model A vs Model B",
            description="Test new model version",
            hypothesis="Model B will improve accuracy",
            primary_metric="accuracy",
            secondary_metrics=["latency", "f1"],
        )

        variants = [
            Variant(
                id="control",
                name="Model A",
                description="Current model",
                allocation=0.5,
            ),
            Variant(
                id="treatment", name="Model B", description="New model", allocation=0.5
            ),
        ]

        experiment = Experiment(id="exp-model-test", config=config, variants=variants)

        # Test lifecycle
        assert experiment.status.value == "draft"

        experiment.start()
        assert experiment.status.value == "running"

        experiment.pause()
        assert experiment.status.value == "paused"

        experiment.resume()
        assert experiment.status.value == "running"

        experiment.complete(winner="treatment")
        assert experiment.status.value == "completed"

        LOGGER.info("✅ Experiment Lifecycle: PASSED")
        LOGGER.info("   - Draft → Running → Paused → Running → Completed")

    except Exception:
        LOGGER.exception("❌ Experiment Lifecycle: FAILED")
        return False
    else:
        return True


def test_statistical_analysis() -> bool:
    """Test statistical significance testing."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 3: Statistical Analysis")
    LOGGER.info("=" * 80)

    try:
        analyzer = FrequentistAnalyzer(significance_level=0.05)

        # Simulated data
        control = [0.82, 0.81, 0.83, 0.82, 0.81] * 20  # Mean ~0.82
        treatment = [0.87, 0.86, 0.88, 0.87, 0.86] * 20  # Mean ~0.87

        result = analyzer.t_test(control, treatment)

        assert result.is_significant is True
        assert result.p_value < 0.05
        assert result.effect_size > 0

        LOGGER.info("✅ Statistical Analysis: PASSED")
        LOGGER.info(f"   - p-value: {result.p_value:.4f}")
        LOGGER.info(f"   - Effect size: {result.effect_size:.4f}")
        LOGGER.info(f"   - Significant: {result.is_significant}")

    except Exception:
        LOGGER.exception("❌ Statistical Analysis: FAILED")
        return False
    else:
        return True


def test_bayesian_analysis() -> bool:
    """Test Bayesian analysis."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 4: Bayesian Analysis")
    LOGGER.info("=" * 80)

    try:
        analyzer = BayesianAnalyzer()

        # Test conversion data
        result = analyzer.beta_binomial_test(
            successes_a=820,
            trials_a=1000,
            successes_b=870,
            trials_b=1000,
        )

        assert "prob_b_better" in result
        assert result["prob_b_better"] > 0.5  # B should be better

        LOGGER.info("✅ Bayesian Analysis: PASSED")
        LOGGER.info(f"   - P(B > A): {result['prob_b_better']:.4f}")
        LOGGER.info(f"   - Expected lift: {result['expected_lift']:.4f}")

    except Exception:
        LOGGER.exception("❌ Bayesian Analysis: FAILED")
        return False
    else:
        return True


def test_experiment_tracking() -> bool:
    """Test experiment tracking."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 5: Experiment Tracking")
    LOGGER.info("=" * 80)

    try:
        tracker = ExperimentTracker()

        # Track events
        for i in range(100):
            variant = "control" if i < 50 else "treatment"
            event = MetricEvent(
                experiment_id="exp-1",
                variant_id=variant,
                metric_name="accuracy",
                value=0.82 if variant == "control" else 0.87,
                user_id=f"user-{i}",
            )
            tracker.track_event(event)

        # Get events
        control_events = tracker.get_events("exp-1", variant_id="control")
        assert len(control_events) == 50

        # Check variant assignment
        assignment = tracker.get_variant_assignment("exp-1", "user-10")
        assert assignment == "control"

        LOGGER.info("✅ Experiment Tracking: PASSED")
        LOGGER.info(f"   - Tracked {len(tracker.events)} events")
        LOGGER.info(f"   - Control events: {len(control_events)}")

    except Exception:
        LOGGER.exception("❌ Experiment Tracking: FAILED")
        return False
    else:
        return True


def test_metric_aggregation() -> bool:
    """Test metric aggregation."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 6: Metric Aggregation")
    LOGGER.info("=" * 80)

    try:
        tracker = ExperimentTracker()

        # Track conversion events
        for i in range(200):
            variant = "control" if i < 100 else "treatment"
            converted = i % 10 < 8 if variant == "control" else i % 10 < 9  # 80% vs 90%

            event = MetricEvent(
                experiment_id="exp-conv",
                variant_id=variant,
                metric_name="conversion",
                value=1.0 if converted else 0.0,
                user_id=f"user-{i}",
            )
            tracker.track_event(event)

        aggregator = MetricAggregator(tracker)

        # Aggregate metrics
        metrics = aggregator.aggregate_metrics("exp-conv", "conversion")
        assert "control" in metrics
        assert "treatment" in metrics

        # Calculate conversion rates
        rates = aggregator.calculate_conversion_rate("exp-conv", "conversion")
        assert abs(rates["control"] - 0.8) < 0.1
        assert abs(rates["treatment"] - 0.9) < 0.1

        LOGGER.info("✅ Metric Aggregation: PASSED")
        LOGGER.info(f"   - Control rate: {rates['control']:.2%}")
        LOGGER.info(f"   - Treatment rate: {rates['treatment']:.2%}")

    except Exception:
        LOGGER.exception("❌ Metric Aggregation: FAILED")
        return False
    else:
        return True


def test_sample_size_calculation() -> bool:
    """Test sample size calculation."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 7: Sample Size Calculation")
    LOGGER.info("=" * 80)

    try:
        sample_size = calculate_sample_size(
            baseline_rate=0.10,
            minimum_detectable_effect=0.20,  # 20% relative improvement
            significance_level=0.05,
            power=0.8,
        )

        assert sample_size > 0
        assert sample_size < 100000  # Sanity check

        LOGGER.info("✅ Sample Size Calculation: PASSED")
        LOGGER.info(f"   - Required per variant: {sample_size}")

    except Exception:
        LOGGER.exception("❌ Sample Size Calculation: FAILED")
        return False
    else:
        return True


def test_experiment_manager() -> bool:
    """Test experiment manager."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 8: Experiment Manager")
    LOGGER.info("=" * 80)

    try:
        manager = ExperimentManager()

        config = ExperimentConfig(
            name="Test Experiment",
            description="Test",
            hypothesis="Test hypothesis",
            primary_metric="accuracy",
        )

        variants = [
            Variant(id="v1", name="V1", description="V1", allocation=0.5),
            Variant(id="v2", name="V2", description="V2", allocation=0.5),
        ]

        # Create experiment
        exp = manager.create_experiment("test-exp", config, variants)
        assert exp.id == "test-exp"

        # Get experiment
        retrieved = manager.get_experiment("test-exp")
        assert retrieved is not None
        assert retrieved.id == exp.id

        # List experiments
        experiments = manager.list_experiments()
        assert len(experiments) == 1

        LOGGER.info("✅ Experiment Manager: PASSED")
        LOGGER.info("   - Created and retrieved experiment")

    except Exception:
        LOGGER.exception("❌ Experiment Manager: FAILED")
        return False
    else:
        return True


def main():
    """Run all A/B testing tests."""
    LOGGER.info("Starting Phase 21 A/B Testing Tests")
    LOGGER.info("=" * 80)

    tests = [
        ("Traffic Splitting", test_traffic_splitting),
        ("Experiment Lifecycle", test_experiment_lifecycle),
        ("Statistical Analysis", test_statistical_analysis),
        ("Bayesian Analysis", test_bayesian_analysis),
        ("Experiment Tracking", test_experiment_tracking),
        ("Metric Aggregation", test_metric_aggregation),
        ("Sample Size Calculation", test_sample_size_calculation),
        ("Experiment Manager", test_experiment_manager),
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
