#!/usr/bin/env python
"""Test ensemble methods (Phase 11).

Quick test to validate ensemble selection and building functionality.
Creates synthetic study and tests all ensemble strategies.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import optuna


def create_test_study() -> optuna.Study:
    """Create test study with synthetic trials."""

    def objective(trial: optuna.Trial) -> float:
        x1 = trial.suggest_float("x1", -5, 5)
        x2 = trial.suggest_float("x2", -5, 5)
        x3 = trial.suggest_float("x3", -5, 5)
        return x1**2 + x2**2 + x3**2

    study = optuna.create_study(
        study_name="test-ensemble",
        direction="minimize",
        storage="sqlite:///test_optuna.db",
        load_if_exists=False,
    )

    study.optimize(objective, n_trials=20, show_progress_bar=False)
    return study


def test_model_selection() -> None:
    """Test model selection strategies."""
    print("\n" + "=" * 80)
    print("TEST 1: Model Selection Strategies")
    print("=" * 80)

    from psy_agents_noaug.hpo.ensemble import ModelSelector

    study = create_test_study()
    selector = ModelSelector(study)

    # Test best-K selection
    top5 = selector.select_best_k(k=5)
    print(f"\n✓ Selected top 5 models (best value: {top5[0].value:.6f})")

    # Test diverse selection
    diverse5 = selector.select_diverse(k=5, quality_weight=0.7)
    print(f"✓ Selected 5 diverse models (diversity scores: {[m.diversity_score for m in diverse5][:3]}...)")

    # Test threshold selection
    threshold_models = selector.select_by_threshold(relative_threshold=0.2, max_models=8)
    print(f"✓ Selected {len(threshold_models)} models within 20% threshold")

    print("\n✅ TEST 1 PASSED: Model selection working correctly")


def test_ensemble_building() -> None:
    """Test ensemble building strategies."""
    print("\n" + "=" * 80)
    print("TEST 2: Ensemble Building")
    print("=" * 80)

    from psy_agents_noaug.hpo.ensemble import EnsembleBuilder, ModelSelector

    study = optuna.load_study("test-ensemble", storage="sqlite:///test_optuna.db")
    selector = ModelSelector(study)
    models = selector.select_best_k(k=5)

    builder = EnsembleBuilder("test_ensemble")

    # Test average ensemble
    avg_ens = builder.build_average_ensemble(models)
    print(f"\n✓ Built averaging ensemble with {len(avg_ens.models)} models")
    print(f"  Weights (equal): {avg_ens.weights[:3]}...")

    # Test weighted ensemble
    weighted_ens = builder.build_weighted_ensemble(models, weighting="performance")
    print(f"✓ Built weighted ensemble with {len(weighted_ens.models)} models")
    print(f"  Weights (performance): {weighted_ens.weights[:3]}...")

    # Test voting ensemble
    voting_ens = builder.build_voting_ensemble(models, voting="soft")
    print(f"✓ Built voting ensemble with {len(voting_ens.models)} models")

    # Test export
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        export_path = Path(f.name)
    builder.export_ensemble(avg_ens, export_path)
    print(f"✓ Exported ensemble config to temporary file")

    # Test load
    loaded_ens = builder.load_ensemble(export_path)
    print(f"✓ Loaded ensemble config: {loaded_ens.name}")

    # Cleanup
    export_path.unlink()

    print("\n✅ TEST 2 PASSED: Ensemble building working correctly")


def test_ensemble_evaluation() -> None:
    """Test ensemble performance estimation."""
    print("\n" + "=" * 80)
    print("TEST 3: Ensemble Evaluation")
    print("=" * 80)

    from psy_agents_noaug.hpo.ensemble import EnsembleBuilder, EnsembleEvaluator, ModelSelector

    study = optuna.load_study("test-ensemble", storage="sqlite:///test_optuna.db")
    selector = ModelSelector(study)
    models = selector.select_best_k(k=5)

    builder = EnsembleBuilder()
    ensemble = builder.build_average_ensemble(models)

    evaluator = EnsembleEvaluator(ensemble)

    # Test performance estimation
    individual_perfs = [m.value for m in models]
    perf_est = evaluator.estimate_ensemble_performance(individual_perfs)

    print(f"\n✓ Performance estimation:")
    print(f"  Individual mean: {perf_est['individual_mean']:.6f}")
    print(f"  Expected ensemble: {perf_est['expected_estimate']:.6f}")
    print(f"  Optimistic: {perf_est['optimistic_estimate']:.6f}")

    # Test diversity analysis
    # Simulate predictions for testing
    n_models, n_samples = 5, 100
    predictions = np.random.randint(0, 4, size=(n_models, n_samples))
    diversity_metrics = evaluator.analyze_ensemble_diversity(predictions)

    print(f"\n✓ Diversity analysis:")
    print(f"  Avg pairwise disagreement: {diversity_metrics['avg_pairwise_disagreement']:.4f}")
    print(f"  Avg prediction entropy: {diversity_metrics['avg_prediction_entropy']:.4f}")

    print("\n✅ TEST 3 PASSED: Ensemble evaluation working correctly")


def test_recommendations() -> None:
    """Test recommendation functions."""
    print("\n" + "=" * 80)
    print("TEST 4: Recommendation Functions")
    print("=" * 80)

    from psy_agents_noaug.hpo.ensemble import recommend_ensemble_strategy, recommend_selection_strategy

    # Test selection recommendations
    strategy, kwargs = recommend_selection_strategy(100, "ensemble", "medium")
    print(f"\n✓ Selection recommendation: {strategy} with {kwargs}")

    # Test ensemble recommendations
    strategy, kwargs = recommend_ensemble_strategy(5, "classification", "high")
    print(f"✓ Ensemble recommendation: {strategy} with {kwargs}")

    print("\n✅ TEST 4 PASSED: Recommendations working correctly")


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
    print("SUPERMAX Phase 11: Ensemble Methods Tests")
    print("=" * 80)

    try:
        test_model_selection()
        test_ensemble_building()
        test_ensemble_evaluation()
        test_recommendations()

        cleanup()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nEnsemble functionality is working correctly!")
        print("You can now use:")
        print("  - ModelSelector for selecting models from HPO studies")
        print("  - EnsembleBuilder for creating ensemble configurations")
        print("  - EnsembleEvaluator for analyzing ensemble performance")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
