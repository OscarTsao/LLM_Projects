#!/usr/bin/env python
"""SUPERMAX Phase 9: Sampler Comparison Benchmark.

Compare different sampling strategies (TPE, CMA-ES, BOHB, Random, NSGA-II)
on various optimization problems to evaluate their efficiency and quality.

Usage:
    # Compare all samplers on default benchmark
    python scripts/compare_samplers.py

    # Compare specific samplers
    python scripts/compare_samplers.py \
        --samplers tpe bohb cmaes \
        --n-trials 100

    # Test on specific benchmark
    python scripts/compare_samplers.py \
        --benchmark rastrigin \
        --n-trials 200

    # Show recommendations
    python scripts/compare_samplers.py --show-recommendations
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

from psy_agents_noaug.hpo.samplers import (
    compare_samplers,
    create_bohb_sampler,
    create_cmaes_sampler,
    create_nsga2_sampler,
    create_random_sampler,
    create_tpe_sampler,
    get_recommended_sampler,
    print_sampler_comparison,
)

LOGGER = logging.getLogger("compare_samplers")


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
        description="SUPERMAX Sampler Comparison - Benchmark sampling strategies"
    )
    parser.add_argument(
        "--samplers",
        nargs="+",
        choices=["tpe", "bohb", "cmaes", "random", "all"],
        default=["all"],
        help="Samplers to compare (default: all)",
    )
    parser.add_argument(
        "--benchmark",
        choices=["sphere", "rastrigin", "rosenbrock", "mixed_params"],
        default="sphere",
        help="Benchmark function to use (default: sphere)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials per sampler (default: 100)",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=42,
        help="Random seed for samplers (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sampler_comparison.json"),
        help="Output file for results (default: outputs/sampler_comparison.json)",
    )
    parser.add_argument(
        "--show-recommendations",
        action="store_true",
        help="Show sampler recommendations for typical scenarios",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def create_sphere_objective():
    """Create sphere function optimization objective.

    Simple convex function: f(x) = sum(x_i^2)
    Optimal at origin with value 0.
    """

    def objective(trial: optuna.Trial) -> float:
        x1 = trial.suggest_float("x1", -5, 5)
        x2 = trial.suggest_float("x2", -5, 5)
        x3 = trial.suggest_float("x3", -5, 5)
        return x1**2 + x2**2 + x3**2

    return objective


def create_rastrigin_objective():
    """Create Rastrigin function optimization objective.

    Highly multi-modal function with many local minima.
    Global optimum at origin with value 0.
    """

    def objective(trial: optuna.Trial) -> float:
        x1 = trial.suggest_float("x1", -5.12, 5.12)
        x2 = trial.suggest_float("x2", -5.12, 5.12)
        x3 = trial.suggest_float("x3", -5.12, 5.12)

        A = 10
        n = 3
        return (
            A * n
            + (x1**2 - A * np.cos(2 * np.pi * x1))
            + (x2**2 - A * np.cos(2 * np.pi * x2))
            + (x3**2 - A * np.cos(2 * np.pi * x3))
        )

    return objective


def create_rosenbrock_objective():
    """Create Rosenbrock function optimization objective.

    Banana-shaped valley, difficult for gradient-free methods.
    Global optimum at (1, 1, 1) with value 0.
    """

    def objective(trial: optuna.Trial) -> float:
        x1 = trial.suggest_float("x1", -5, 5)
        x2 = trial.suggest_float("x2", -5, 5)
        x3 = trial.suggest_float("x3", -5, 5)

        return (
            100 * (x2 - x1**2) ** 2
            + (1 - x1) ** 2
            + 100 * (x3 - x2**2) ** 2
            + (1 - x2) ** 2
        )

    return objective


def create_mixed_params_objective():
    """Create mixed parameter optimization objective.

    Includes continuous, integer, and categorical parameters.
    Simulates realistic ML hyperparameter tuning.
    """

    def objective(trial: optuna.Trial) -> float:
        # Continuous parameters
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        # Integer parameter
        n_layers = trial.suggest_int("n_layers", 1, 5)

        # Categorical parameter
        optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])

        # Compute synthetic score
        lr_score = abs(np.log10(lr) - (-3)) ** 2  # Optimal ~1e-3
        wd_score = abs(np.log10(weight_decay) - (-4)) ** 2  # Optimal ~1e-4
        layer_score = (n_layers - 3) ** 2  # Optimal 3 layers

        optimizer_scores = {"adam": 0.0, "sgd": 0.3, "rmsprop": 0.1}
        opt_score = optimizer_scores[optimizer]

        return lr_score + wd_score * 0.5 + layer_score * 0.2 + opt_score

    return objective


def show_sampler_recommendations() -> None:
    """Show sampler recommendations for typical scenarios."""
    print("=" * 80)
    print("Sampler Recommendations for Common Scenarios")
    print("=" * 80)

    scenarios = [
        {
            "name": "Standard ML HPO (mixed params, medium budget)",
            "search_space_type": "mixed",
            "trial_budget": "medium",
            "parallel_trials": 1,
        },
        {
            "name": "All-continuous optimization (hyperparameter fine-tuning)",
            "search_space_type": "continuous",
            "trial_budget": "medium",
            "optimization_goal": "quality",
        },
        {
            "name": "Large-scale HPO (1000+ trials with pruning)",
            "search_space_type": "mixed",
            "trial_budget": "very_large",
            "optimization_goal": "speed",
        },
        {
            "name": "Multi-objective (F1 vs inference time)",
            "search_space_type": "mixed",
            "n_objectives": 2,
            "trial_budget": "medium",
        },
        {
            "name": "Quick exploration (small budget)",
            "search_space_type": "mixed",
            "trial_budget": "small",
            "optimization_goal": "speed",
        },
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        strategy = get_recommended_sampler(**scenario)
        print(f"  Recommended: {strategy.name}")
        print(f"  Performance: {strategy.expected_performance}")
        print(f"  Rationale: {strategy.rationale}")

    print("=" * 80)


def main() -> None:
    """Main entry point for sampler comparison."""
    args = parse_args()
    setup_logging(args.log_level)

    # Show recommendations if requested
    if args.show_recommendations:
        show_sampler_recommendations()
        return

    # Setup samplers to compare
    sampler_choices = args.samplers
    if "all" in sampler_choices:
        sampler_choices = ["tpe", "bohb", "random"]
        if args.benchmark in ["sphere", "rastrigin", "rosenbrock"]:
            # CMA-ES only works with all-continuous
            sampler_choices.append("cmaes")

    samplers = {}
    if "tpe" in sampler_choices:
        samplers["TPE (Multivariate)"] = create_tpe_sampler(
            seed=args.sampler_seed,
            n_startup_trials=10,
            multivariate=True,
        )
    if "bohb" in sampler_choices:
        samplers["BOHB-style TPE"] = create_bohb_sampler(
            seed=args.sampler_seed,
            n_startup_trials=5,
        )
    if "cmaes" in sampler_choices:
        samplers["CMA-ES"] = create_cmaes_sampler(
            seed=args.sampler_seed,
            sigma0=0.2,
        )
    if "random" in sampler_choices:
        samplers["Random"] = create_random_sampler(seed=args.sampler_seed)

    # Create benchmark objective
    if args.benchmark == "sphere":
        objective = create_sphere_objective()
    elif args.benchmark == "rastrigin":
        objective = create_rastrigin_objective()
    elif args.benchmark == "rosenbrock":
        objective = create_rosenbrock_objective()
    else:  # mixed_params
        objective = create_mixed_params_objective()

    LOGGER.info("Benchmark: %s", args.benchmark)
    LOGGER.info("Trials per sampler: %d", args.n_trials)
    LOGGER.info("Samplers: %s", list(samplers.keys()))

    # Run comparison
    LOGGER.info("Running comparison...")
    results = compare_samplers(
        samplers=samplers,
        objective_func=objective,
        n_trials=args.n_trials,
    )

    # Print results
    print("\n")
    print_sampler_comparison(results)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "benchmark": args.benchmark,
        "n_trials": args.n_trials,
        "sampler_seed": args.sampler_seed,
        "results": results,
    }
    with args.output.open("w") as f:
        json.dump(output_data, f, indent=2, default=str)

    LOGGER.info("Results saved to: %s", args.output)

    # Summary
    best_quality = min(results.items(), key=lambda x: x[1]["best_value"])
    print(f"\n✅ Comparison complete!")
    print(
        f"   Best quality: {best_quality[0]} (value={best_quality[1]['best_value']:.6f})"
    )
    print(f"   Results: {args.output}")


if __name__ == "__main__":
    main()
