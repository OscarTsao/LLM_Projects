"""Pruner selection strategies and recommendations.

This module provides intelligent pruner selection based on:
- Search space characteristics (size, complexity)
- Resource constraints (time budget, trial cost)
- Optimization landscape (noisy, smooth)
- Problem requirements (speed vs thoroughness)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import optuna
from optuna.pruners import BasePruner

from psy_agents_noaug.hpo.pruners.factory import (
    create_hyperband_pruner,
    create_median_pruner,
    create_percentile_pruner,
    get_pruner_info,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class PrunerStrategy:
    """Pruner recommendation with rationale."""

    name: str
    pruner: BasePruner
    aggressiveness: Literal["none", "low", "medium", "high", "very_high"]
    best_for: list[str]
    rationale: str
    estimated_speedup: float  # Expected speedup multiplier


def get_recommended_pruner(
    *,
    search_space_size: Literal["small", "medium", "large", "very_large"] = "medium",
    trial_cost: Literal[
        "cheap", "moderate", "expensive", "very_expensive"
    ] = "moderate",
    total_budget_hours: float = 8.0,
    optimization_goal: Literal["speed", "balanced", "thoroughness"] = "balanced",
    noise_level: Literal["low", "medium", "high"] = "medium",
    min_resource: int = 1,
    max_resource: int = 10,
) -> PrunerStrategy:
    """Recommend a pruner based on problem characteristics.

    This function applies expert heuristics to select the best pruner
    for your specific HPO problem.

    Args:
        search_space_size: Number of hyperparameters and their ranges
            - small: 1-3 hyperparameters
            - medium: 4-7 hyperparameters
            - large: 8-15 hyperparameters
            - very_large: 16+ hyperparameters
        trial_cost: Average time per trial
            - cheap: < 30 seconds
            - moderate: 30s - 5 minutes
            - expensive: 5 - 30 minutes
            - very_expensive: > 30 minutes
        total_budget_hours: Total time budget for HPO
        optimization_goal: What matters most
            - speed: Find good solution fast (aggressive pruning)
            - balanced: Balance exploration and efficiency
            - thoroughness: Explore thoroughly (conservative pruning)
        noise_level: Variability in trial metrics
            - low: Metrics are stable and reproducible
            - medium: Some variance between runs
            - high: High variance (use conservative pruning)
        min_resource: Minimum resource allocation (e.g., epochs)
        max_resource: Maximum resource allocation

    Returns:
        Pruner recommendation with rationale

    Example:
        >>> # Large search space, expensive trials, want speed
        >>> strategy = get_recommended_pruner(
        ...     search_space_size="large",
        ...     trial_cost="expensive",
        ...     optimization_goal="speed",
        ... )
        >>> print(strategy.rationale)
        >>> study = optuna.create_study(pruner=strategy.pruner)
    """
    # Decision logic based on characteristics
    score_aggressive = 0.0

    # Search space size factor
    size_scores = {"small": -1, "medium": 0, "large": 1, "very_large": 2}
    score_aggressive += size_scores[search_space_size]

    # Trial cost factor (expensive = more aggressive)
    cost_scores = {"cheap": -1, "moderate": 0, "expensive": 1, "very_expensive": 2}
    score_aggressive += cost_scores[trial_cost]

    # Goal factor
    goal_scores = {"thoroughness": -2, "balanced": 0, "speed": 2}
    score_aggressive += goal_scores[optimization_goal]

    # Noise factor (high noise = less aggressive)
    noise_scores = {"low": 1, "medium": 0, "high": -2}
    score_aggressive += noise_scores[noise_level]

    # Budget factor (tight budget = more aggressive)
    if total_budget_hours < 2.0:
        score_aggressive += 2
    elif total_budget_hours < 8.0:
        score_aggressive += 1
    elif total_budget_hours > 24.0:
        score_aggressive -= 1

    LOGGER.info(
        "Pruner selection score: %.1f (size=%s, cost=%s, goal=%s, noise=%s, budget=%.1fh)",
        score_aggressive,
        search_space_size,
        trial_cost,
        optimization_goal,
        noise_level,
        total_budget_hours,
    )

    # Select pruner based on score
    if score_aggressive >= 4:
        # Very aggressive: Hyperband with aggressive reduction
        pruner = create_hyperband_pruner(
            min_resource=min_resource,
            max_resource=max_resource,
            reduction_factor=4,  # More aggressive
        )
        return PrunerStrategy(
            name="Hyperband (Aggressive)",
            pruner=pruner,
            aggressiveness="very_high",
            best_for=[
                "Very large search spaces",
                "Expensive trials",
                "Tight time budgets",
                "Speed-focused optimization",
            ],
            rationale=(
                f"Selected Hyperband with reduction_factor=4 because: "
                f"search_space={search_space_size}, trial_cost={trial_cost}, "
                f"goal={optimization_goal}. This will prune aggressively to find "
                f"good solutions quickly."
            ),
            estimated_speedup=4.0,
        )

    if score_aggressive >= 2:
        # Aggressive: Standard Hyperband
        pruner = create_hyperband_pruner(
            min_resource=min_resource,
            max_resource=max_resource,
            reduction_factor=3,
        )
        return PrunerStrategy(
            name="Hyperband (Standard)",
            pruner=pruner,
            aggressiveness="high",
            best_for=[
                "Large search spaces",
                "Expensive trials",
                "Moderate time budgets",
            ],
            rationale=(
                f"Selected standard Hyperband because: "
                f"search_space={search_space_size}, trial_cost={trial_cost}. "
                f"This provides good balance between speed and quality."
            ),
            estimated_speedup=3.0,
        )

    if score_aggressive >= 0:
        # Moderate: Percentile pruner
        percentile = 25.0 if score_aggressive >= 1 else 35.0
        n_startup = max(5, int(10 - score_aggressive))

        pruner = create_percentile_pruner(
            percentile=percentile,
            n_startup_trials=n_startup,
            n_warmup_steps=0,
            interval_steps=1,
        )
        return PrunerStrategy(
            name=f"Percentile ({percentile:.0f}%)",
            pruner=pruner,
            aggressiveness="medium",
            best_for=[
                "Balanced exploration/exploitation",
                "Moderate search spaces",
                "Moderate trial costs",
            ],
            rationale=(
                f"Selected Percentile pruner (percentile={percentile:.0f}%) because: "
                f"goal={optimization_goal}, noise={noise_level}. "
                f"This provides moderate pruning with {n_startup} startup trials."
            ),
            estimated_speedup=2.0,
        )

    if score_aggressive >= -2:
        # Conservative: Median pruner
        n_startup = max(10, int(15 + score_aggressive))
        n_warmup = 2 if noise_level == "high" else 0

        pruner = create_median_pruner(
            n_startup_trials=n_startup,
            n_warmup_steps=n_warmup,
            interval_steps=1,
        )
        return PrunerStrategy(
            name="Median (Conservative)",
            pruner=pruner,
            aggressiveness="low",
            best_for=[
                "Thorough exploration",
                "Noisy optimization",
                "Small search spaces",
                "Cheap trials",
            ],
            rationale=(
                f"Selected Median pruner because: "
                f"goal={optimization_goal}, noise={noise_level}. "
                f"This provides conservative pruning with {n_startup} startup trials "
                f"and {n_warmup} warmup steps to handle noise."
            ),
            estimated_speedup=1.5,
        )

    # Very conservative: Median with high startup
    pruner = create_median_pruner(
        n_startup_trials=20,
        n_warmup_steps=3,
        interval_steps=1,
    )
    return PrunerStrategy(
        name="Median (Very Conservative)",
        pruner=pruner,
        aggressiveness="none",
        best_for=[
            "Maximum thoroughness",
            "Very noisy optimization",
            "Small trial budgets (prefer quality)",
        ],
        rationale=(
            f"Selected very conservative Median pruner because: "
            f"noise={noise_level}, goal={optimization_goal}. "
            f"This prioritizes exploration quality over speed."
        ),
        estimated_speedup=1.2,
    )


def compare_pruners(
    pruners: dict[str, BasePruner],
    objective_func,
    n_trials: int = 50,
    sampler_seed: int = 42,
) -> dict[str, dict]:
    """Compare different pruners on the same optimization problem.

    Runs multiple studies with different pruners and compares their:
    - Convergence speed (trials to reach target)
    - Final best value
    - Computational efficiency (pruned trials)
    - Wall-clock time

    Args:
        pruners: Dictionary of {name: pruner} to compare
        objective_func: Objective function for Optuna (trial -> float)
        n_trials: Number of trials per pruner
        sampler_seed: Seed for reproducibility

    Returns:
        Dictionary of results for each pruner

    Example:
        >>> def objective(trial):
        ...     x = trial.suggest_float("x", -10, 10)
        ...     # Report intermediate values for pruning
        ...     for step in range(10):
        ...         intermediate = x**2 * (step + 1) / 10
        ...         trial.report(intermediate, step)
        ...         if trial.should_prune():
        ...             raise optuna.TrialPruned()
        ...     return x**2
        >>>
        >>> pruners = {
        ...     "hyperband": create_hyperband_pruner(),
        ...     "median": create_median_pruner(),
        ...     "percentile": create_percentile_pruner(),
        ... }
        >>> results = compare_pruners(pruners, objective, n_trials=100)
        >>> for name, stats in results.items():
        ...     print(f"{name}: best={stats['best_value']:.4f}, "
        ...           f"pruned={stats['pruned_ratio']:.1%}")
    """
    import time

    results = {}

    for pruner_name, pruner in pruners.items():
        LOGGER.info("Comparing pruner: %s", pruner_name)

        # Create study
        study = optuna.create_study(
            study_name=f"comparison_{pruner_name}",
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=sampler_seed),
            pruner=pruner,
        )

        # Run optimization
        start_time = time.time()
        study.optimize(objective_func, n_trials=n_trials, show_progress_bar=False)
        elapsed_time = time.time() - start_time

        # Collect statistics
        trials = study.get_trials(deepcopy=False)
        completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]

        results[pruner_name] = {
            "best_value": study.best_value if completed else float("inf"),
            "best_trial": study.best_trial.number if completed else None,
            "n_completed": len(completed),
            "n_pruned": len(pruned),
            "pruned_ratio": len(pruned) / len(trials) if trials else 0.0,
            "elapsed_time": elapsed_time,
            "trials_per_second": (
                len(trials) / elapsed_time if elapsed_time > 0 else 0.0
            ),
            "pruner_info": get_pruner_info(pruner),
        }

        LOGGER.info(
            "  Best value: %.4f (trial #%d)",
            results[pruner_name]["best_value"],
            results[pruner_name]["best_trial"] or -1,
        )
        LOGGER.info(
            "  Pruned: %d/%d (%.1f%%)",
            len(pruned),
            len(trials),
            results[pruner_name]["pruned_ratio"] * 100,
        )
        LOGGER.info("  Time: %.2fs", elapsed_time)

    return results


def print_pruner_comparison(results: dict[str, dict]) -> None:
    """Pretty-print pruner comparison results.

    Args:
        results: Results from compare_pruners()
    """
    print("=" * 80)
    print("Pruner Comparison Results")
    print("=" * 80)

    # Sort by best value
    sorted_results = sorted(results.items(), key=lambda x: x[1]["best_value"])

    print(
        f"\n{'Pruner':<25} {'Best Value':>12} {'Completed':>10} {'Pruned':>10} {'Time (s)':>10}"
    )
    print("-" * 80)

    for pruner_name, stats in sorted_results:
        print(
            f"{pruner_name:<25} "
            f"{stats['best_value']:>12.6f} "
            f"{stats['n_completed']:>10d} "
            f"{stats['n_pruned']:>10d} "
            f"{stats['elapsed_time']:>10.2f}"
        )

    print("=" * 80)

    # Winner analysis
    best_quality = sorted_results[0]
    best_speed = max(results.items(), key=lambda x: x[1]["trials_per_second"])
    best_efficiency = max(results.items(), key=lambda x: x[1]["pruned_ratio"])

    print("\nWinner Analysis:")
    print(
        f"  🏆 Best Quality: {best_quality[0]} (value={best_quality[1]['best_value']:.6f})"
    )
    print(
        f"  ⚡ Fastest: {best_speed[0]} ({best_speed[1]['trials_per_second']:.2f} trials/s)"
    )
    print(
        f"  💾 Most Efficient: {best_efficiency[0]} ({best_efficiency[1]['pruned_ratio']:.1%} pruned)"
    )
    print("=" * 80)
