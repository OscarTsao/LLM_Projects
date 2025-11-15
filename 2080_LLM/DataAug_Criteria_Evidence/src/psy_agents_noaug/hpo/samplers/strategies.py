"""Sampler selection strategies and recommendations.

This module provides intelligent sampler selection based on:
- Search space characteristics (continuous vs categorical)
- Optimization objectives (single vs multi-objective)
- Trial budget and parallelization
- Problem complexity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import optuna
from optuna.samplers import BaseSampler

from psy_agents_noaug.hpo.samplers.factory import (
    create_bohb_sampler,
    create_cmaes_sampler,
    create_nsga2_sampler,
    create_random_sampler,
    create_tpe_sampler,
    get_sampler_info,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class SamplerStrategy:
    """Sampler recommendation with rationale."""

    name: str
    sampler: BaseSampler
    characteristics: list[str]
    best_for: list[str]
    rationale: str
    expected_performance: Literal["baseline", "good", "excellent"]


def get_recommended_sampler(
    *,
    search_space_type: Literal["continuous", "mixed", "categorical"] = "mixed",
    n_objectives: int = 1,
    trial_budget: Literal["small", "medium", "large", "very_large"] = "medium",
    parallel_trials: int = 1,
    optimization_goal: Literal["speed", "balanced", "quality"] = "balanced",
    seed: int | None = None,
) -> SamplerStrategy:
    """Recommend a sampler based on problem characteristics.

    This function applies expert heuristics to select the best sampler
    for your specific HPO problem.

    Args:
        search_space_type: Type of hyperparameters
            - continuous: All parameters are continuous (lr, weight_decay, etc.)
            - mixed: Mix of continuous and categorical
            - categorical: Mostly categorical parameters
        n_objectives: Number of optimization objectives
            - 1: Single-objective (most common)
            - 2+: Multi-objective (Pareto optimization)
        trial_budget: Number of trials you can afford
            - small: < 50 trials
            - medium: 50-200 trials
            - large: 200-1000 trials
            - very_large: 1000+ trials
        parallel_trials: Number of parallel workers
        optimization_goal: What matters most
            - speed: Find good solution quickly
            - balanced: Balance speed and quality
            - quality: Best possible solution
        seed: Random seed

    Returns:
        Sampler recommendation with rationale

    Example:
        >>> # Standard ML HPO: mixed params, single objective
        >>> strategy = get_recommended_sampler(
        ...     search_space_type="mixed",
        ...     trial_budget="medium",
        ...     parallel_trials=4,
        ... )
        >>> print(strategy.rationale)
        >>> study = optuna.create_study(sampler=strategy.sampler)
    """
    # Multi-objective: always use NSGA-II
    if n_objectives >= 2:
        sampler = create_nsga2_sampler(
            seed=seed,
            population_size=max(50, parallel_trials * 10),
        )
        return SamplerStrategy(
            name="NSGA-II (Multi-objective)",
            sampler=sampler,
            characteristics=["Multi-objective", "Genetic Algorithm", "Pareto front"],
            best_for=[
                "Finding trade-offs between objectives",
                "Diverse solution sets",
                "2-3 objectives",
            ],
            rationale=(
                f"Selected NSGA-II because you have {n_objectives} objectives. "
                f"This will find the Pareto front of non-dominated solutions."
            ),
            expected_performance="excellent",
        )

    # Continuous-only: CMA-ES is excellent
    if search_space_type == "continuous" and optimization_goal in ["balanced", "quality"]:
        sampler = create_cmaes_sampler(
            seed=seed,
            sigma0=0.2,
            n_startup_trials=5,
        )
        return SamplerStrategy(
            name="CMA-ES (Continuous Optimization)",
            sampler=sampler,
            characteristics=["Continuous-only", "Evolution Strategy", "Covariance adaptation"],
            best_for=[
                "All-continuous search spaces",
                "Non-convex landscapes",
                "High-dimensional continuous",
            ],
            rationale=(
                f"Selected CMA-ES because search_space={search_space_type}. "
                f"CMA-ES excels at continuous optimization with covariance adaptation."
            ),
            expected_performance="excellent",
        )

    # Large budget + aggressive: BOHB-style
    if trial_budget in ["large", "very_large"] and optimization_goal == "speed":
        sampler = create_bohb_sampler(
            seed=seed,
            n_startup_trials=5,
            multivariate=True,
        )
        return SamplerStrategy(
            name="BOHB-style TPE",
            sampler=sampler,
            characteristics=["Bayesian Optimization", "Few startup trials", "Multivariate"],
            best_for=[
                "Large trial budgets (1000+ trials)",
                "When paired with Hyperband pruner",
                "Aggressive optimization",
            ],
            rationale=(
                f"Selected BOHB-style TPE because trial_budget={trial_budget} "
                f"and goal={optimization_goal}. Pair with Hyperband pruner for best results!"
            ),
            expected_performance="excellent",
        )

    # Medium-large budget, quality focus: Standard TPE with multivariate
    if trial_budget in ["medium", "large"] and optimization_goal in ["balanced", "quality"]:
        n_startup = {"medium": 10, "large": 15}.get(trial_budget, 10)
        sampler = create_tpe_sampler(
            seed=seed,
            n_startup_trials=n_startup,
            multivariate=True,
            constant_liar=parallel_trials > 1,
        )
        return SamplerStrategy(
            name="TPE (Multivariate)",
            sampler=sampler,
            characteristics=["Bayesian Optimization", "Multivariate", "General-purpose"],
            best_for=[
                "General ML hyperparameter tuning",
                "Mixed parameter types",
                "Moderate budgets",
            ],
            rationale=(
                f"Selected multivariate TPE because trial_budget={trial_budget}, "
                f"search_space={search_space_type}. This is the most robust general choice."
            ),
            expected_performance="good",
        )

    # Small budget or speed focus: Independent TPE or Random
    if trial_budget == "small" or optimization_goal == "speed":
        if trial_budget == "small" and optimization_goal == "speed":
            # Very small budget + speed: just random
            sampler = create_random_sampler(seed=seed)
            return SamplerStrategy(
                name="Random Search (Baseline)",
                sampler=sampler,
                characteristics=["No modeling overhead", "Fast", "Baseline"],
                best_for=[
                    "Very small budgets (< 50 trials)",
                    "Quick sanity checks",
                    "Baseline comparisons",
                ],
                rationale=(
                    f"Selected Random search because trial_budget={trial_budget} "
                    f"and goal={optimization_goal}. With few trials, modeling overhead isn't worth it."
                ),
                expected_performance="baseline",
            )
        else:
            # Small budget but not speed-critical: independent TPE
            sampler = create_tpe_sampler(
                seed=seed,
                n_startup_trials=5,
                multivariate=False,
                constant_liar=parallel_trials > 1,
            )
            return SamplerStrategy(
                name="TPE (Independent)",
                sampler=sampler,
                characteristics=["Bayesian Optimization", "Independent", "Low overhead"],
                best_for=[
                    "Small budgets (50-100 trials)",
                    "Speed-focused optimization",
                    "Simple search spaces",
                ],
                rationale=(
                    f"Selected independent TPE because trial_budget={trial_budget}. "
                    f"Independent mode has lower overhead than multivariate."
                ),
                expected_performance="good",
            )

    # Default: Standard TPE
    sampler = create_tpe_sampler(
        seed=seed,
        n_startup_trials=10,
        multivariate=False,
        constant_liar=parallel_trials > 1,
    )
    return SamplerStrategy(
        name="TPE (Default)",
        sampler=sampler,
        characteristics=["Bayesian Optimization", "General-purpose"],
        best_for=[
            "Default choice",
            "Unknown problem characteristics",
        ],
        rationale="Selected default TPE as a safe general-purpose choice.",
        expected_performance="good",
    )


def compare_samplers(
    samplers: dict[str, BaseSampler],
    objective_func,
    n_trials: int = 100,
) -> dict[str, dict]:
    """Compare different samplers on the same optimization problem.

    Runs multiple studies with different samplers and compares their:
    - Best value found
    - Convergence speed (trials to reach threshold)
    - Consistency (variance across runs)

    Args:
        samplers: Dictionary of {name: sampler} to compare
        objective_func: Objective function for Optuna (trial -> float)
        n_trials: Number of trials per sampler

    Returns:
        Dictionary of results for each sampler

    Example:
        >>> def objective(trial):
        ...     x = trial.suggest_float("x", -10, 10)
        ...     y = trial.suggest_float("y", -10, 10)
        ...     return x**2 + y**2
        >>>
        >>> samplers = {
        ...     "random": create_random_sampler(seed=42),
        ...     "tpe": create_tpe_sampler(seed=42),
        ...     "bohb": create_bohb_sampler(seed=42),
        ... }
        >>> results = compare_samplers(samplers, objective, n_trials=100)
        >>> for name, stats in results.items():
        ...     print(f"{name}: best={stats['best_value']:.4f}")
    """
    import time

    results = {}

    for sampler_name, sampler in samplers.items():
        LOGGER.info("Comparing sampler: %s", sampler_name)

        # Create study
        study = optuna.create_study(
            study_name=f"comparison_{sampler_name}",
            direction="minimize",
            sampler=sampler,
        )

        # Run optimization
        start_time = time.time()
        study.optimize(objective_func, n_trials=n_trials, show_progress_bar=False)
        elapsed_time = time.time() - start_time

        # Collect statistics
        trials = study.get_trials(deepcopy=False)
        values = [t.value for t in trials if t.value is not None]

        results[sampler_name] = {
            "best_value": study.best_value,
            "best_trial": study.best_trial.number,
            "mean_value": sum(values) / len(values) if values else float("inf"),
            "n_trials": len(trials),
            "elapsed_time": elapsed_time,
            "trials_per_second": len(trials) / elapsed_time if elapsed_time > 0 else 0.0,
            "sampler_info": get_sampler_info(sampler),
        }

        LOGGER.info(
            "  Best value: %.4f (trial #%d)",
            results[sampler_name]["best_value"],
            results[sampler_name]["best_trial"],
        )
        LOGGER.info("  Time: %.2fs", elapsed_time)

    return results


def print_sampler_comparison(results: dict[str, dict]) -> None:
    """Pretty-print sampler comparison results.

    Args:
        results: Results from compare_samplers()
    """
    print("=" * 80)
    print("Sampler Comparison Results")
    print("=" * 80)

    # Sort by best value
    sorted_results = sorted(results.items(), key=lambda x: x[1]["best_value"])

    print(
        f"\n{'Sampler':<25} {'Best Value':>12} {'Mean Value':>12} {'Best Trial':>12} {'Time (s)':>10}"
    )
    print("-" * 80)

    for sampler_name, stats in sorted_results:
        print(
            f"{sampler_name:<25} "
            f"{stats['best_value']:>12.6f} "
            f"{stats['mean_value']:>12.6f} "
            f"{stats['best_trial']:>12d} "
            f"{stats['elapsed_time']:>10.2f}"
        )

    print("=" * 80)

    # Winner analysis
    best_quality = sorted_results[0]
    best_speed = max(results.items(), key=lambda x: x[1]["trials_per_second"])
    fastest_convergence = min(results.items(), key=lambda x: x[1]["best_trial"])

    print("\nWinner Analysis:")
    print(
        f"  🏆 Best Quality: {best_quality[0]} (value={best_quality[1]['best_value']:.6f})"
    )
    print(
        f"  ⚡ Fastest: {best_speed[0]} ({best_speed[1]['trials_per_second']:.2f} trials/s)"
    )
    print(
        f"  🎯 Fastest Convergence: {fastest_convergence[0]} (found best at trial #{fastest_convergence[1]['best_trial']})"
    )
    print("=" * 80)
