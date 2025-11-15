"""Ensemble methods and model selection for HPO (Phase 11).

This module provides tools for building ensembles from HPO trials
and selecting the best models for deployment.

Key Features:
- Model selection strategies (best-K, diverse, threshold, Pareto)
- Ensemble building (averaging, voting, stacking)
- Weighted ensembles with multiple weighting schemes
- Ensemble performance estimation
- Diversity analysis

Example Usage:

    # Select top models from HPO study
    from psy_agents_noaug.hpo.ensemble import ModelSelector
    import optuna

    study = optuna.load_study("my-hpo-study", storage="sqlite:///optuna.db")
    selector = ModelSelector(study)

    # Get top 5 models
    top_models = selector.select_best_k(k=5)

    # Get diverse models
    diverse_models = selector.select_diverse(k=5, quality_weight=0.7)

    # Build ensemble
    from psy_agents_noaug.hpo.ensemble import EnsembleBuilder

    builder = EnsembleBuilder("my_ensemble")

    # Simple averaging
    avg_ensemble = builder.build_average_ensemble(top_models)

    # Weighted by performance
    weighted_ensemble = builder.build_weighted_ensemble(
        top_models,
        weighting="performance",
        temperature=1.0,
    )

    # Export ensemble config
    builder.export_ensemble(weighted_ensemble, "outputs/ensemble_config.json")

    # Evaluate ensemble
    from psy_agents_noaug.hpo.ensemble import EnsembleEvaluator

    evaluator = EnsembleEvaluator(weighted_ensemble)
    perf_est = evaluator.estimate_ensemble_performance([0.85, 0.83, 0.87, 0.84, 0.86])
    print(f"Expected performance: {perf_est['expected_estimate']:.4f}")
"""

# Model selection
from psy_agents_noaug.hpo.ensemble.selection import (
    ModelSelector,
    SelectedModel,
    recommend_selection_strategy,
)

# Ensemble building
from psy_agents_noaug.hpo.ensemble.builder import (
    EnsembleBuilder,
    EnsembleConfig,
    EnsembleEvaluator,
    recommend_ensemble_strategy,
)

__all__ = [
    # Selection
    "ModelSelector",
    "SelectedModel",
    "recommend_selection_strategy",
    # Building
    "EnsembleBuilder",
    "EnsembleConfig",
    "EnsembleEvaluator",
    "recommend_ensemble_strategy",
]
