#!/usr/bin/env python
"""SUPERMAX Phase 6: Ensemble Selection and Stacking.

This script builds powerful ensembles from Pareto front or CV results:
1. Load Stage-B Pareto front or Stage-C CV results
2. Select diverse, high-performing models
3. Build voting ensembles (uniform, performance-based, optimized)
4. Build stacking ensembles (logistic, random forest)
5. Compare strategies and export best ensemble

Usage:
    # From Stage-B Pareto front
    python scripts/build_ensemble.py \\
        --agent criteria \\
        --input-file outputs/supermax/stage_b/criteria/criteria_stage_b_pareto.json \\
        --input-type pareto

    # From Stage-C CV results
    python scripts/build_ensemble.py \\
        --agent criteria \\
        --input-file outputs/supermax/stage_c/criteria/criteria_stage_c_cv5.json \\
        --input-type cv

    # Quick test (simulated predictions)
    python scripts/build_ensemble.py \\
        --agent criteria \\
        --input-file outputs/supermax/stage_b/criteria/criteria_stage_b_pareto.json \\
        --input-type pareto \\
        --n-models 3 \\
        --simulate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from psy_agents_noaug.ensemble import (
    DiversitySelector,
    WeightedVotingEnsemble,
    StackingEnsemble,
)
from psy_agents_noaug.ensemble.stacking import compare_ensemble_strategies

LOGGER = logging.getLogger("build_ensemble")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build ensemble from Pareto front or CV results"
    )
    parser.add_argument(
        "--agent",
        required=True,
        choices=["criteria", "evidence", "share", "joint"],
        help="Agent/architecture",
    )
    parser.add_argument(
        "--input-file",
        required=True,
        type=Path,
        help="Input JSON file (Stage-B pareto or Stage-C CV results)",
    )
    parser.add_argument(
        "--input-type",
        required=True,
        choices=["pareto", "cv"],
        help="Type of input file",
    )
    parser.add_argument(
        "--n-models",
        type=int,
        default=5,
        help="Number of models to select for ensemble (default: 5)",
    )
    parser.add_argument(
        "--selection-strategy",
        default="hybrid",
        choices=["topk", "diversity", "cluster", "hybrid"],
        help="Diversity selection strategy (default: hybrid)",
    )
    parser.add_argument(
        "--diversity-weight",
        type=float,
        default=0.3,
        help="Diversity weight for hybrid strategy (default: 0.3)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use simulated predictions (for testing without trained models)",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/supermax/ensemble",
        help="Output directory for ensemble results",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_input_file(
    path: Path,
    input_type: str,
) -> tuple[list[dict[str, Any]], str]:
    """Load input file (Pareto front or CV results).

    Args:
        path: Path to input JSON file
        input_type: Type of input ("pareto" or "cv")

    Returns:
        Tuple of (configs_list, source_stage)
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r") as f:
        data = json.load(f)

    if input_type == "pareto":
        configs = data.get("pareto_front", [])
        source_stage = "stage_b"
    elif input_type == "cv":
        configs = data.get("cv_results", [])
        source_stage = "stage_c"
    else:
        raise ValueError(f"Unknown input type: {input_type}")

    LOGGER.info("Loaded %d configurations from %s (%s)", len(configs), path, input_type)
    return configs, source_stage


def simulate_model_predictions(
    n_models: int,
    n_samples: int = 200,
    n_classes: int = 4,
    base_accuracy: float = 0.75,
    diversity: float = 0.3,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray]:
    """Simulate model predictions for testing (without trained models).

    Args:
        n_models: Number of models
        n_samples: Number of samples per split
        n_classes: Number of classes
        base_accuracy: Base accuracy for models
        diversity: Diversity between models (0-1)

    Returns:
        Tuple of (train_probs, train_labels, test_probs, test_labels)
    """
    LOGGER.info("Simulating predictions for %d models", n_models)

    # Generate ground truth labels
    train_labels = np.random.randint(0, n_classes, size=n_samples)
    test_labels = np.random.randint(0, n_classes, size=n_samples)

    # Generate model predictions (somewhat correlated but diverse)
    train_probs = []
    test_probs = []

    for model_idx in range(n_models):
        # Each model has slightly different bias
        model_bias = np.random.randn(n_classes) * diversity

        # Training predictions
        train_p = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            true_class = train_labels[i]
            # Higher probability for correct class
            if np.random.rand() < base_accuracy:
                train_p[i, true_class] = np.random.uniform(0.6, 0.95)
            else:
                train_p[i, true_class] = np.random.uniform(0.05, 0.4)

            # Distribute remaining probability
            remaining = 1.0 - train_p[i, true_class]
            other_classes = [c for c in range(n_classes) if c != true_class]
            train_p[i, other_classes] = np.random.dirichlet(
                np.ones(len(other_classes))
            ) * remaining

            # Add model bias
            train_p[i] += model_bias
            train_p[i] = np.maximum(train_p[i], 0.01)
            train_p[i] /= train_p[i].sum()

        train_probs.append(train_p)

        # Test predictions (similar process)
        test_p = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            true_class = test_labels[i]
            if np.random.rand() < base_accuracy:
                test_p[i, true_class] = np.random.uniform(0.6, 0.95)
            else:
                test_p[i, true_class] = np.random.uniform(0.05, 0.4)

            remaining = 1.0 - test_p[i, true_class]
            other_classes = [c for c in range(n_classes) if c != true_class]
            test_p[i, other_classes] = np.random.dirichlet(
                np.ones(len(other_classes))
            ) * remaining

            test_p[i] += model_bias
            test_p[i] = np.maximum(test_p[i], 0.01)
            test_p[i] /= test_p[i].sum()

        test_probs.append(test_p)

    return train_probs, train_labels, test_probs, test_labels


def export_ensemble_results(
    results: dict[str, Any],
    outdir: Path,
    agent: str,
    source_stage: str,
) -> Path:
    """Export ensemble evaluation results to JSON.

    Args:
        results: Dictionary with ensemble results
        outdir: Output directory
        agent: Agent name
        source_stage: Source stage (stage_b or stage_c)

    Returns:
        Path to exported JSON file
    """
    outfile = outdir / f"{agent}_ensemble_{source_stage}.json"
    outfile.parent.mkdir(parents=True, exist_ok=True)

    with outfile.open("w") as f:
        json.dump(results, f, indent=2)

    LOGGER.info("Exported ensemble results to %s", outfile)
    return outfile


def main() -> None:
    """Main entry point for ensemble building."""
    args = parse_args()
    setup_logging(args.log_level)

    # Load input configurations
    configs, source_stage = load_input_file(args.input_file, args.input_type)

    if len(configs) < args.n_models:
        LOGGER.warning(
            "Only %d configs available, using all (requested %d)",
            len(configs),
            args.n_models,
        )
        args.n_models = len(configs)

    LOGGER.info("=" * 80)
    LOGGER.info("SUPERMAX PHASE 6: Ensemble Building")
    LOGGER.info("=" * 80)
    LOGGER.info("Agent: %s", args.agent)
    LOGGER.info("Input: %s (%s)", args.input_file, args.input_type)
    LOGGER.info("Selecting: %d models", args.n_models)
    LOGGER.info("Selection strategy: %s", args.selection_strategy)
    if args.selection_strategy == "hybrid":
        LOGGER.info("  - Performance weight: %.1f%%", (1 - args.diversity_weight) * 100)
        LOGGER.info("  - Diversity weight: %.1f%%", args.diversity_weight * 100)
    LOGGER.info("Output: %s", args.outdir)
    LOGGER.info("=" * 80)

    # Step 1: Select diverse models
    LOGGER.info("\n[Step 1/4] Selecting diverse models from %s", source_stage.upper())
    selector = DiversitySelector(
        strategy=args.selection_strategy,
        n_models=args.n_models,
        diversity_weight=args.diversity_weight,
    )

    # Determine score key based on input type
    score_key = "f1_macro" if args.input_type == "pareto" else "f1_macro_mean"

    selected_configs = selector.select(
        configs,
        config_key="config",
        score_key=score_key,
    )

    LOGGER.info("Selected %d diverse models:", len(selected_configs))
    for i, config in enumerate(selected_configs, 1):
        score = config.get(score_key, 0.0)
        rank = config.get("rank", "?")
        LOGGER.info("  %d. Rank %s: %s=%.4f", i, rank, score_key, score)

    # Step 2: Get model predictions (simulated or real)
    LOGGER.info("\n[Step 2/4] Generating/loading model predictions")

    if args.simulate:
        # Simulate predictions for testing
        model_scores = [c.get(score_key, 0.75) for c in selected_configs]
        train_probs, train_labels, test_probs, test_labels = simulate_model_predictions(
            n_models=len(selected_configs),
            base_accuracy=np.mean(model_scores),
        )
        LOGGER.info("Using simulated predictions (train: %d, test: %d)", len(train_labels), len(test_labels))
    else:
        # In production, this would load/generate real predictions
        LOGGER.error("Real prediction loading not implemented yet. Use --simulate flag.")
        sys.exit(1)

    # Step 3: Build and compare ensemble strategies
    LOGGER.info("\n[Step 3/4] Building and comparing ensemble strategies")

    model_scores = [c.get(score_key, 0.0) for c in selected_configs]
    ensemble_results = compare_ensemble_strategies(
        model_probs_train=train_probs,
        model_probs_test=test_probs,
        y_train=train_labels,
        y_test=test_labels,
        model_scores=model_scores,
    )

    # Step 4: Export results
    LOGGER.info("\n[Step 4/4] Exporting ensemble results")

    # Find best strategy
    best_strategy = max(ensemble_results.items(), key=lambda x: x[1]["f1_macro"])
    best_name, best_metrics = best_strategy

    LOGGER.info("=" * 80)
    LOGGER.info("BEST ENSEMBLE: %s", best_name)
    LOGGER.info("  F1-macro: %.4f", best_metrics["f1_macro"])
    LOGGER.info("  F1-micro: %.4f", best_metrics["f1_micro"])
    LOGGER.info("  Logloss:  %.4f", best_metrics["logloss"])
    LOGGER.info("=" * 80)

    # Prepare export data
    export_data = {
        "agent": args.agent,
        "source_stage": source_stage,
        "source_file": str(args.input_file),
        "n_models_selected": len(selected_configs),
        "selection_strategy": args.selection_strategy,
        "diversity_weight": args.diversity_weight,
        "selected_configs": [
            {
                "rank": i + 1,
                "original_rank": c.get("rank", "?"),
                "score": c.get(score_key, 0.0),
                "config": c.get("config", {}),
            }
            for i, c in enumerate(selected_configs)
        ],
        "ensemble_strategies": ensemble_results,
        "best_strategy": {
            "name": best_name,
            "metrics": best_metrics,
        },
        "simulated": args.simulate,
    }

    outdir = Path(args.outdir)
    export_path = export_ensemble_results(export_data, outdir, args.agent, source_stage)

    # Generate summary report
    report_path = outdir / f"{args.agent}_ensemble_report.txt"
    with report_path.open("w") as f:
        f.write("=" * 80 + "\n")
        f.write("SUPERMAX PHASE 6: ENSEMBLE REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Agent: {args.agent}\n")
        f.write(f"Source: {source_stage.upper()} ({args.input_file})\n")
        f.write(f"Models selected: {len(selected_configs)}\n")
        f.write(f"Selection strategy: {args.selection_strategy}\n")
        f.write("\n")

        f.write("Selected Models:\n")
        for i, config in enumerate(selected_configs, 1):
            score = config.get(score_key, 0.0)
            rank = config.get("rank", "?")
            f.write(f"  {i}. Rank {rank}: {score_key}={score:.4f}\n")
        f.write("\n")

        f.write("Ensemble Strategy Comparison:\n")
        for strategy, metrics in sorted(ensemble_results.items(), key=lambda x: -x[1]["f1_macro"]):
            f.write(
                f"  {strategy:25s}: F1={metrics['f1_macro']:.4f}, Logloss={metrics['logloss']:.4f}\n"
            )
        f.write("\n")

        f.write(f"Best Strategy: {best_name}\n")
        f.write(f"  F1-macro: {best_metrics['f1_macro']:.4f}\n")
        f.write(f"  F1-micro: {best_metrics['f1_micro']:.4f}\n")
        f.write(f"  Logloss:  {best_metrics['logloss']:.4f}\n")
        f.write("\n")

        f.write(f"Ensemble config export: {export_path}\n")
        f.write(f"Recommended: Deploy {best_name} ensemble for production\n")

    LOGGER.info("Summary report: %s", report_path)
    LOGGER.info("=" * 80)
    LOGGER.info("Ensemble building complete!")


if __name__ == "__main__":
    main()
