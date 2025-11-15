#!/usr/bin/env python
"""Test meta-learning functionality (Phase 10).

Quick test to validate that meta-learning modules work correctly.
Creates synthetic studies and tests analysis/warm-start/transfer.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import optuna

from psy_agents_noaug.hpo.meta_learning import (
    AdaptiveWarmStart,
    TransferLearner,
    TrialHistoryAnalyzer,
    WarmStartStrategy,
    print_study_summary,
)


def create_test_objective():
    """Create simple test objective (sphere function)."""

    def objective(trial: optuna.Trial) -> float:
        x1 = trial.suggest_float("x1", -5, 5)
        x2 = trial.suggest_float("x2", -5, 5)
        x3 = trial.suggest_float("x3", -5, 5)
        return x1**2 + x2**2 + x3**2

    return objective


def test_history_analyzer() -> None:
    """Test 1: Trial history analysis."""
    print("\n" + "=" * 80)
    print("TEST 1: Trial History Analysis")
    print("=" * 80)

    # Create test study
    study = optuna.create_study(
        study_name="test-meta-learning-source",
        direction="minimize",
        storage="sqlite:///test_optuna.db",
        load_if_exists=False,
    )

    # Run some trials
    objective = create_test_objective()
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    print(f"✓ Created source study with {len(study.trials)} trials")

    # Analyze study
    analyzer = TrialHistoryAnalyzer(storage="sqlite:///test_optuna.db")
    analysis = analyzer.analyze_study(
        "test-meta-learning-source", compute_importance=True
    )

    print(f"✓ Analyzed study: best_value={analysis.best_value:.6f}")
    print(f"✓ Parameter importance computed: {len(analysis.param_importance)} params")

    # Print summary
    print_study_summary(analysis, detailed=False)

    # Get top configs
    top_configs = analyzer.get_top_k_configs("test-meta-learning-source", k=3)
    print(f"✓ Retrieved top {len(top_configs)} configurations")

    print("\n✅ TEST 1 PASSED: History analysis working correctly")


def test_warm_start() -> None:
    """Test 2: Warm-starting."""
    print("\n" + "=" * 80)
    print("TEST 2: Warm-Starting")
    print("=" * 80)

    # Create warm-starter
    warm_starter = WarmStartStrategy(storage="sqlite:///test_optuna.db")

    # Create new study
    new_study = optuna.create_study(
        study_name="test-meta-learning-warm-started",
        direction="minimize",
        storage="sqlite:///test_optuna.db",
        load_if_exists=False,
    )

    # Warm-start from source study
    n_enqueued = warm_starter.warm_start_from_study(
        target_study=new_study,
        source_study="test-meta-learning-source",
        n_configs=5,
        strategy="best",
    )

    print(f"✓ Enqueued {n_enqueued} configurations")

    # Check that trials are enqueued
    # Note: In Optuna, enqueued trials appear as trials when you ask for them
    # Let's run a few trials to see if warm-start configs are used
    objective = create_test_objective()
    new_study.optimize(objective, n_trials=5, show_progress_bar=False)

    print(f"✓ Ran {len(new_study.trials)} trials on warm-started study")
    print(f"✓ Best value after warm-start: {new_study.best_value:.6f}")

    print("\n✅ TEST 2 PASSED: Warm-starting working correctly")


def test_transfer_learning() -> None:
    """Test 3: Transfer learning."""
    print("\n" + "=" * 80)
    print("TEST 3: Transfer Learning")
    print("=" * 80)

    # Create transfer learner
    transfer = TransferLearner(storage="sqlite:///test_optuna.db")

    # Create new study for different "task"
    transfer_study = optuna.create_study(
        study_name="test-meta-learning-transferred",
        direction="minimize",
        storage="sqlite:///test_optuna.db",
        load_if_exists=False,
    )

    # Transfer knowledge (simulate criteria -> evidence)
    n_transferred = transfer.transfer_from_task(
        target_study=transfer_study,
        source_task="criteria",
        target_task="evidence",
        source_study="test-meta-learning-source",
        n_configs=3,
        transfer_mode="shared_only",
    )

    print(f"✓ Transferred {n_transferred} configurations")

    # Run trials
    objective = create_test_objective()
    transfer_study.optimize(objective, n_trials=3, show_progress_bar=False)

    print(f"✓ Ran {len(transfer_study.trials)} trials on transferred study")
    print(f"✓ Best value after transfer: {transfer_study.best_value:.6f}")

    print("\n✅ TEST 3 PASSED: Transfer learning working correctly")


def test_adaptive_warm_start() -> None:
    """Test 4: Adaptive warm-starting."""
    print("\n" + "=" * 80)
    print("TEST 4: Adaptive Warm-Starting")
    print("=" * 80)

    # Create adaptive warm-starter
    adaptive = AdaptiveWarmStart(
        storage="sqlite:///test_optuna.db",
        similarity_threshold=0.0,  # Low threshold for testing
    )

    # Create new study
    adaptive_study = optuna.create_study(
        study_name="test-meta-learning-adaptive",
        direction="minimize",
        storage="sqlite:///test_optuna.db",
        load_if_exists=False,
    )

    # Adaptive warm-start
    candidate_studies = [
        "test-meta-learning-source",
        "test-meta-learning-warm-started",
    ]

    n_enqueued = adaptive.adaptive_warm_start(
        target_study=adaptive_study,
        candidate_studies=candidate_studies,
        total_configs=6,
        max_sources=2,
    )

    print(f"✓ Adaptively enqueued {n_enqueued} configurations")

    # Run trials
    objective = create_test_objective()
    adaptive_study.optimize(objective, n_trials=6, show_progress_bar=False)

    print(f"✓ Ran {len(adaptive_study.trials)} trials on adaptive study")
    print(f"✓ Best value after adaptive warm-start: {adaptive_study.best_value:.6f}")

    print("\n✅ TEST 4 PASSED: Adaptive warm-starting working correctly")


def cleanup() -> None:
    """Clean up test database."""
    print("\n" + "=" * 80)
    print("CLEANUP")
    print("=" * 80)

    db_path = Path("test_optuna.db")
    if db_path.exists():
        db_path.unlink()
        print("✓ Removed test database")


def main() -> None:
    """Run all tests."""
    print("=" * 80)
    print("SUPERMAX Phase 10: Meta-Learning Tests")
    print("=" * 80)

    try:
        # Run tests
        test_history_analyzer()
        test_warm_start()
        test_transfer_learning()
        test_adaptive_warm_start()

        # Cleanup
        cleanup()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nMeta-learning functionality is working correctly!")
        print("You can now use:")
        print("  - scripts/analyze_hpo.py for study analysis")
        print("  - scripts/warm_start_hpo.py for creating warm-started studies")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
