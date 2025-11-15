#!/usr/bin/env python
"""Visualization utilities for meta-learning analysis (Phase 10).

This module provides utilities for visualizing HPO results,
parameter importance, and meta-learning insights.

Key Features:
- Parameter importance plots
- Convergence analysis
- Transfer learning impact
- Study comparison visualizations
"""

from __future__ import annotations

import logging
from pathlib import Path

from psy_agents_noaug.hpo.meta_learning.history import StudyAnalysis

LOGGER = logging.getLogger(__name__)


def print_study_summary(analysis: StudyAnalysis, detailed: bool = False) -> None:
    """Print a formatted summary of study analysis.

    Args:
        analysis: Study analysis to print
        detailed: Whether to include detailed trial information
    """
    print("=" * 80)
    print(f"Study Analysis: {analysis.study_name}")
    print("=" * 80)
    print(f"Direction: {analysis.direction}")
    print(f"Total Trials: {analysis.n_trials}")
    print(f"  - Completed: {analysis.n_completed_trials}")
    print(f"  - Pruned: {analysis.n_pruned_trials}")
    print(f"  - Failed: {analysis.n_failed_trials}")
    print()

    if analysis.n_completed_trials > 0:
        print(f"Best Value: {analysis.best_value:.6f}")
        print(f"Best Trial: #{analysis.best_trial_number}")
        print("Best Parameters:")
        for param, value in analysis.best_params.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.6f}")
            else:
                print(f"  {param}: {value}")
        print()

        if analysis.param_importance:
            print("Parameter Importance:")
            sorted_importance = sorted(
                analysis.param_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for param, importance in sorted_importance:
                bar = "█" * int(importance * 50)
                print(f"  {param:20s} {bar:50s} {importance:.3f}")
            print()

        if analysis.convergence_iteration is not None:
            print(f"Convergence: Trial #{analysis.convergence_iteration}")
            print(
                f"  (Study converged after {analysis.convergence_iteration} trials, "
                f"{analysis.n_trials - analysis.convergence_iteration} trials did not improve best value)"
            )
            print()

    if detailed and analysis.trials:
        print("Trial History (last 10 trials):")
        print(f"{'Trial':<8} {'Value':<12} {'State':<12} {'Duration (s)':<15}")
        print("-" * 80)
        for trial in analysis.trials[-10:]:
            print(
                f"#{trial.trial_number:<7} {trial.value:<12.6f} {trial.state:<12} {trial.duration:<15.2f}"
            )
        print()

    print("=" * 80)


def print_importance_comparison(
    studies: dict[str, StudyAnalysis],
    top_k: int = 10,
) -> None:
    """Print comparison of parameter importance across studies.

    Args:
        studies: Dict of {study_name: analysis}
        top_k: Number of top parameters to show
    """
    print("=" * 80)
    print("Parameter Importance Comparison")
    print("=" * 80)

    # Collect all parameters
    all_params: set[str] = set()
    for analysis in studies.values():
        all_params.update(analysis.param_importance.keys())

    # Compute average importance per parameter
    param_avg_importance = {}
    for param in all_params:
        importances = [
            analysis.param_importance.get(param, 0.0) for analysis in studies.values()
        ]
        param_avg_importance[param] = sum(importances) / len(importances)

    # Sort by average importance
    sorted_params = sorted(
        param_avg_importance.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    # Print table header
    print(f"{'Parameter':<20s}", end="")
    for study_name in studies:
        print(f"{study_name:<15s}", end="")
    print()
    print("-" * 80)

    # Print importance for each parameter across studies
    for param, avg_importance in sorted_params:
        print(f"{param:<20s}", end="")
        for analysis in studies.values():
            importance = analysis.param_importance.get(param, 0.0)
            print(f"{importance:<15.3f}", end="")
        print(f" (avg: {avg_importance:.3f})")

    print("=" * 80)


def print_transfer_recommendations(
    target_task: str,
    available_studies: dict[str, str],
) -> None:
    """Print transfer learning recommendations.

    Args:
        target_task: Target task name
        available_studies: Available source studies
    """
    from psy_agents_noaug.hpo.meta_learning.transfer import recommend_transfer_sources

    print("=" * 80)
    print(f"Transfer Learning Recommendations for '{target_task}'")
    print("=" * 80)

    sources = recommend_transfer_sources(target_task, available_studies)

    if not sources:
        print(f"No recommended transfer sources for '{target_task}'")
        print("=" * 80)
        return

    print("Recommended sources (in priority order):")
    for i, (source_task, study_name) in enumerate(sources, 1):
        print(f"{i}. {source_task:<15s} → {study_name}")

    print()
    print("Usage:")
    print("  from psy_agents_noaug.hpo.meta_learning import TransferLearner")
    print("  learner = TransferLearner(storage='sqlite:///optuna.db')")
    print("  learner.transfer_from_task(")
    print("      target_study=your_study,")
    print(f"      source_task='{sources[0][0]}',")
    print(f"      target_task='{target_task}',")
    print(f"      source_study='{sources[0][1]}',")
    print("      n_configs=5,")
    print("  )")
    print("=" * 80)


def print_convergence_analysis(
    studies: dict[str, StudyAnalysis],
) -> None:
    """Print convergence analysis for multiple studies.

    Args:
        studies: Dict of {study_name: analysis}
    """
    print("=" * 80)
    print("Convergence Analysis")
    print("=" * 80)
    print(f"{'Study':<30s} {'Trials':<10s} {'Converged':<15s} {'Efficiency':<12s}")
    print("-" * 80)

    for study_name, analysis in studies.items():
        converged_at = analysis.convergence_iteration
        if converged_at is not None:
            efficiency = f"{(converged_at / analysis.n_trials) * 100:.1f}%"
            converged_str = f"Trial #{converged_at}"
        else:
            efficiency = "N/A"
            converged_str = "Not converged"

        print(
            f"{study_name:<30s} {analysis.n_trials:<10d} {converged_str:<15s} {efficiency:<12s}"
        )

    print()
    print("Efficiency = (Convergence Trial / Total Trials) * 100%")
    print("Lower efficiency means faster convergence (better)")
    print("=" * 80)


def export_importance_csv(
    studies: dict[str, StudyAnalysis],
    output_path: Path | str,
) -> None:
    """Export parameter importance to CSV for further analysis.

    Args:
        studies: Dict of {study_name: analysis}
        output_path: Path to output CSV file
    """
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all parameters
    all_params: set[str] = set()
    for analysis in studies.values():
        all_params.update(analysis.param_importance.keys())

    # Write CSV
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["parameter"] + list(studies.keys()) + ["mean", "std"]
        writer.writerow(header)

        # Data rows
        for param in sorted(all_params):
            row = [param]
            importances = []

            for analysis in studies.values():
                importance = analysis.param_importance.get(param, 0.0)
                row.append(f"{importance:.6f}")
                importances.append(importance)

            # Add statistics
            mean_importance = (
                sum(importances) / len(importances) if importances else 0.0
            )
            std_importance = (
                (
                    sum((x - mean_importance) ** 2 for x in importances)
                    / len(importances)
                )
                ** 0.5
                if importances
                else 0.0
            )

            row.append(f"{mean_importance:.6f}")
            row.append(f"{std_importance:.6f}")

            writer.writerow(row)

    LOGGER.info("Exported parameter importance to: %s", output_path)


def compare_study_performance(
    studies: dict[str, StudyAnalysis],
) -> None:
    """Compare performance across multiple studies.

    Args:
        studies: Dict of {study_name: analysis}
    """
    print("=" * 80)
    print("Study Performance Comparison")
    print("=" * 80)
    print(
        f"{'Study':<30s} {'Best Value':<15s} {'Completed':<12s} {'Pruned %':<12s} {'Failed %':<12s}"
    )
    print("-" * 80)

    for study_name, analysis in studies.items():
        if analysis.n_trials > 0:
            pruned_pct = (analysis.n_pruned_trials / analysis.n_trials) * 100
            failed_pct = (analysis.n_failed_trials / analysis.n_trials) * 100
        else:
            pruned_pct = 0.0
            failed_pct = 0.0

        print(
            f"{study_name:<30s} "
            f"{analysis.best_value:<15.6f} "
            f"{analysis.n_completed_trials:<12d} "
            f"{pruned_pct:<12.1f} "
            f"{failed_pct:<12.1f}"
        )

    print("=" * 80)


def analyze_warm_start_impact(
    baseline_study: StudyAnalysis,
    warm_started_study: StudyAnalysis,
    n_warm_start_trials: int,
) -> None:
    """Analyze the impact of warm-starting.

    Args:
        baseline_study: Study without warm-start
        warm_started_study: Study with warm-start
        n_warm_start_trials: Number of trials used for warm-start
    """
    print("=" * 80)
    print("Warm-Start Impact Analysis")
    print("=" * 80)
    print(f"Baseline Study: {baseline_study.study_name}")
    print(f"Warm-Started Study: {warm_started_study.study_name}")
    print(f"Warm-Start Trials: {n_warm_start_trials}")
    print()

    # Compare best values
    baseline_best = baseline_study.best_value
    warm_start_best = warm_started_study.best_value

    improvement = ((baseline_best - warm_start_best) / baseline_best) * 100
    print(f"Baseline Best Value: {baseline_best:.6f}")
    print(f"Warm-Started Best Value: {warm_start_best:.6f}")
    print(f"Improvement: {improvement:+.2f}%")
    print()

    # Compare convergence
    baseline_conv = baseline_study.convergence_iteration
    warm_start_conv = warm_started_study.convergence_iteration

    if baseline_conv is not None and warm_start_conv is not None:
        conv_speedup = baseline_conv - warm_start_conv
        print(f"Baseline Convergence: Trial #{baseline_conv}")
        print(f"Warm-Started Convergence: Trial #{warm_start_conv}")
        print(
            f"Speedup: {conv_speedup} trials ({(conv_speedup/baseline_conv)*100:.1f}%)"
        )
    else:
        print("Convergence: Unable to compare (one or both studies did not converge)")

    print("=" * 80)
