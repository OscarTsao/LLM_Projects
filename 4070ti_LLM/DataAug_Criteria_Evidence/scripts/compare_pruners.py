#!/usr/bin/env python
"""SUPERMAX Phase 8: Pruner Comparison Benchmark.

Compare different pruning strategies on synthetic optimization problems
to evaluate their efficiency and quality trade-offs.

This script benchmarks pruners on:
- Synthetic quadratic functions
- Noisy optimization landscapes
- Multi-modal functions
- ML-style convergence curves

Usage:
    # Compare all pruners on default benchmark
    python scripts/compare_pruners.py

    # Compare specific pruners
    python scripts/compare_pruners.py \
        --pruners hyperband median percentile \
        --n-trials 100 \
        --n-steps 20

    # Test on noisy function
    python scripts/compare_pruners.py \
        --benchmark noisy \
        --noise-level 0.3

    # Quick test
    python scripts/compare_pruners.py \
        --n-trials 20 \
        --n-steps 10 \
        --output results/pruner_quick_test.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import optuna

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from psy_agents_noaug.hpo.pruners import (
    compare_pruners,
    create_hyperband_pruner,
    create_median_pruner,
    create_percentile_pruner,
    estimate_trial_budget,
    get_recommended_pruner,
    print_pruner_comparison,
)

LOGGER = logging.getLogger("compare_pruners")


def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SUPERMAX Pruner Comparison - Benchmark pruning strategies"
    )
    parser.add_argument(
        "--pruners",
        nargs="+",
        choices=["hyperband", "median", "percentile", "all"],
        default=["all"],
        help="Pruners to compare (default: all)",
    )
    parser.add_argument(
        "--benchmark",
        choices=["quadratic", "noisy", "multimodal", "ml_convergence"],
        default="ml_convergence",
        help="Benchmark function to use (default: ml_convergence)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials per pruner (default: 50)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=20,
        help="Number of optimization steps (default: 20)",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.1,
        help="Noise level for noisy benchmark (default: 0.1)",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=42,
        help="Random seed for sampler (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pruner_comparison.json"),
        help="Output file for results (default: outputs/pruner_comparison.json)",
    )
    parser.add_argument(
        "--show-budget",
        action="store_true",
        help="Show resource budget estimation",
    )
    parser.add_argument(
        "--show-recommendation",
        action="store_true",
        help="Show pruner recommendation for typical scenarios",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def create_quadratic_objective(n_steps: int, noise_level: float = 0.0):
    """Create quadratic optimization objective.

    Simple bowl-shaped function: f(x, y) = x^2 + y^2 + noise
    Optimal at (0, 0) with value 0.
    """

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)

        # Simulate convergence over steps
        for step in range(n_steps):
            # Progress towards optimum
            progress = (step + 1) / n_steps
            intermediate = (x**2 + y**2) * (1.0 - progress * 0.5)

            # Add noise
            if noise_level > 0:
                intermediate += np.random.randn() * noise_level * intermediate

            trial.report(intermediate, step)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return x**2 + y**2

    return objective


def create_noisy_objective(n_steps: int, noise_level: float = 0.2):
    """Create noisy optimization objective.

    High variance in intermediate values tests pruner robustness.
    """

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -5, 5)

        true_value = x**2

        for step in range(n_steps):
            # Add significant noise to intermediate values
            noise = np.random.randn() * noise_level * (true_value + 1.0)
            intermediate = true_value + noise

            trial.report(intermediate, step)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return true_value

    return objective


def create_multimodal_objective(n_steps: int):
    """Create multi-modal optimization objective.

    Multiple local optima test exploration vs exploitation.
    f(x) = x^2 + 10*sin(x)
    """

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)

        true_value = x**2 + 10 * np.sin(x)

        for step in range(n_steps):
            progress = (step + 1) / n_steps
            intermediate = true_value * (1.0 - progress * 0.3)

            trial.report(intermediate, step)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return true_value

    return objective


def create_ml_convergence_objective(n_steps: int):
    """Create ML-style convergence objective.

    Simulates typical ML training curve with:
    - Fast initial improvement
    - Diminishing returns
    - Variance between configurations
    """

    def objective(trial: optuna.Trial) -> float:
        # Simulate hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        # Compute "true" final performance (lower is better)
        # Good configs have lr ~ 1e-3, wd ~ 1e-4, dropout ~ 0.2
        lr_score = abs(np.log10(lr) - (-3)) ** 2
        wd_score = abs(np.log10(wd) - (-4)) ** 2
        dropout_score = (dropout - 0.2) ** 2

        final_loss = 0.5 + lr_score * 0.1 + wd_score * 0.05 + dropout_score * 0.3

        # Simulate convergence curve
        for step in range(n_steps):
            # Exponential convergence with diminishing returns
            progress = 1.0 - np.exp(-3 * (step + 1) / n_steps)
            current_loss = 2.0 * (1 - progress) + final_loss * progress

            # Add training variance
            noise = np.random.randn() * 0.02

            trial.report(current_loss + noise, step)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return final_loss

    return objective


def show_budget_estimation() -> None:
    """Show resource budget estimation examples."""
    print("=" * 80)
    print("Resource Budget Estimation Examples")
    print("=" * 80)

    scenarios = [
        {
            "name": "Quick exploration (2 hours)",
            "total_time_hours": 2.0,
            "avg_trial_minutes": 10.0,
            "max_resource": 10,
        },
        {
            "name": "Standard HPO (8 hours)",
            "total_time_hours": 8.0,
            "avg_trial_minutes": 15.0,
            "max_resource": 27,
        },
        {
            "name": "Intensive search (24 hours)",
            "total_time_hours": 24.0,
            "avg_trial_minutes": 20.0,
            "max_resource": 50,
        },
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        budget = estimate_trial_budget(**scenario)
        print(f"  Estimated trials: ~{budget.n_trials}")
        print(f"  Resource range: {budget.min_resource} - {budget.max_resource}")
        print(f"  Expected savings: {budget.estimated_savings:.1%}")

    print("=" * 80)


def show_pruner_recommendations() -> None:
    """Show pruner recommendations for typical scenarios."""
    print("=" * 80)
    print("Pruner Recommendations for Common Scenarios")
    print("=" * 80)

    scenarios = [
        {
            "name": "Large search space, expensive trials, need speed",
            "search_space_size": "large",
            "trial_cost": "expensive",
            "optimization_goal": "speed",
            "noise_level": "low",
        },
        {
            "name": "Medium search space, balanced approach",
            "search_space_size": "medium",
            "trial_cost": "moderate",
            "optimization_goal": "balanced",
            "noise_level": "medium",
        },
        {
            "name": "Small search space, thorough exploration",
            "search_space_size": "small",
            "trial_cost": "cheap",
            "optimization_goal": "thoroughness",
            "noise_level": "high",
        },
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        strategy = get_recommended_pruner(**scenario)
        print(f"  Recommended: {strategy.name}")
        print(f"  Aggressiveness: {strategy.aggressiveness}")
        print(f"  Expected speedup: {strategy.estimated_speedup:.1f}x")
        print(f"  Rationale: {strategy.rationale}")

    print("=" * 80)


def main() -> None:
    """Main entry point for pruner comparison."""
    args = parse_args()
    setup_logging(args.log_level)

    # Show informational outputs if requested
    if args.show_budget:
        show_budget_estimation()
        return

    if args.show_recommendation:
        show_pruner_recommendations()
        return

    # Setup pruners to compare
    pruner_choices = args.pruners
    if "all" in pruner_choices:
        pruner_choices = ["hyperband", "median", "percentile"]

    pruners = {}
    if "hyperband" in pruner_choices:
        pruners["Hyperband (3x)"] = create_hyperband_pruner(
            min_resource=1,
            max_resource=args.n_steps,
            reduction_factor=3,
        )
    if "median" in pruner_choices:
        pruners["Median"] = create_median_pruner(
            n_startup_trials=5,
            n_warmup_steps=2,
        )
    if "percentile" in pruner_choices:
        pruners["Percentile (25%)"] = create_percentile_pruner(
            percentile=25.0,
            n_startup_trials=5,
        )

    # Create benchmark objective
    if args.benchmark == "quadratic":
        objective = create_quadratic_objective(args.n_steps)
    elif args.benchmark == "noisy":
        objective = create_noisy_objective(args.n_steps, args.noise_level)
    elif args.benchmark == "multimodal":
        objective = create_multimodal_objective(args.n_steps)
    else:  # ml_convergence
        objective = create_ml_convergence_objective(args.n_steps)

    LOGGER.info("Benchmark: %s", args.benchmark)
    LOGGER.info("Trials per pruner: %d", args.n_trials)
    LOGGER.info("Steps per trial: %d", args.n_steps)
    LOGGER.info("Pruners: %s", list(pruners.keys()))

    # Run comparison
    LOGGER.info("Running comparison...")
    results = compare_pruners(
        pruners=pruners,
        objective_func=objective,
        n_trials=args.n_trials,
        sampler_seed=args.sampler_seed,
    )

    # Print results
    print("\n")
    print_pruner_comparison(results)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "benchmark": args.benchmark,
        "n_trials": args.n_trials,
        "n_steps": args.n_steps,
        "sampler_seed": args.sampler_seed,
        "results": results,
    }
    with args.output.open("w") as f:
        json.dump(output_data, f, indent=2, default=str)

    LOGGER.info("Results saved to: %s", args.output)

    # Summary
    best_quality = min(results.items(), key=lambda x: x[1]["best_value"])
    print("\n✅ Comparison complete!")
    print(
        f"   Best quality: {best_quality[0]} (value={best_quality[1]['best_value']:.6f})"
    )
    print(f"   Results: {args.output}")


if __name__ == "__main__":
    main()
