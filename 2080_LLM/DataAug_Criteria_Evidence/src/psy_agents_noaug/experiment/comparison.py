#!/usr/bin/env python
"""Experiment comparison utilities (Phase 15).

This module provides tools for comparing multiple experiments:
- Side-by-side metric comparison
- Parameter difference analysis
- Performance ranking
- Comparison report generation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class ExperimentComparison:
    """Comparison of multiple experiments."""

    experiments: list[dict[str, Any]]
    metrics_comparison: pd.DataFrame
    params_comparison: pd.DataFrame
    best_metrics: dict[str, dict[str, Any]]
    ranking: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)


class ExperimentComparator:
    """Compare multiple experiments."""

    def __init__(
        self,
        tracking_uri: str | None = None,
    ):
        """Initialize experiment comparator.

        Args:
            tracking_uri: MLflow tracking URI
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.client = mlflow.tracking.MlflowClient()

        LOGGER.info("Initialized ExperimentComparator")

    def get_experiment_data(
        self,
        run_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Get data for multiple runs.

        Args:
            run_ids: List of run IDs

        Returns:
            List of experiment data
        """
        experiments = []

        for run_id in run_ids:
            run = mlflow.get_run(run_id)

            experiment_data = {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name or "Unnamed",
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
            }

            experiments.append(experiment_data)

        LOGGER.info("Loaded data for %d experiments", len(experiments))
        return experiments

    def compare_metrics(
        self,
        experiments: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Compare metrics across experiments.

        Args:
            experiments: List of experiment data

        Returns:
            Metrics comparison DataFrame
        """
        metrics_data = []

        for exp in experiments:
            row = {"run_id": exp["run_id"], "run_name": exp["run_name"]}
            row.update(exp["metrics"])
            metrics_data.append(row)

        df = pd.DataFrame(metrics_data)

        # Set run_id as index
        if "run_id" in df.columns:
            df = df.set_index("run_id")

        return df

    def compare_parameters(
        self,
        experiments: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Compare parameters across experiments.

        Args:
            experiments: List of experiment data

        Returns:
            Parameters comparison DataFrame
        """
        params_data = []

        for exp in experiments:
            row = {"run_id": exp["run_id"], "run_name": exp["run_name"]}
            row.update(exp["params"])
            params_data.append(row)

        df = pd.DataFrame(params_data)

        # Set run_id as index
        if "run_id" in df.columns:
            df = df.set_index("run_id")

        return df

    def find_best_metrics(
        self,
        experiments: list[dict[str, Any]],
        metrics: list[str] | None = None,
        higher_is_better: dict[str, bool] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Find best value for each metric.

        Args:
            experiments: List of experiment data
            metrics: Metrics to analyze (None = all)
            higher_is_better: Whether higher is better for each metric

        Returns:
            Best metrics dict
        """
        if higher_is_better is None:
            higher_is_better = {}

        # Get all metrics if not specified
        if metrics is None:
            all_metrics = set()
            for exp in experiments:
                all_metrics.update(exp["metrics"].keys())
            metrics = list(all_metrics)

        best_metrics = {}

        for metric in metrics:
            is_higher_better = higher_is_better.get(
                metric,
                True if "accuracy" in metric.lower() or "f1" in metric.lower() else False,
            )

            best_value = None
            best_exp = None

            for exp in experiments:
                if metric not in exp["metrics"]:
                    continue

                value = exp["metrics"][metric]

                if best_value is None:
                    best_value = value
                    best_exp = exp
                elif is_higher_better and value > best_value:
                    best_value = value
                    best_exp = exp
                elif not is_higher_better and value < best_value:
                    best_value = value
                    best_exp = exp

            if best_exp:
                best_metrics[metric] = {
                    "value": best_value,
                    "run_id": best_exp["run_id"],
                    "run_name": best_exp["run_name"],
                    "higher_is_better": is_higher_better,
                }

        return best_metrics

    def rank_experiments(
        self,
        experiments: list[dict[str, Any]],
        ranking_metric: str = "val_accuracy",
        higher_is_better: bool = True,
    ) -> pd.DataFrame:
        """Rank experiments by a metric.

        Args:
            experiments: List of experiment data
            ranking_metric: Metric to rank by
            higher_is_better: Whether higher is better

        Returns:
            Ranking DataFrame
        """
        ranking_data = []

        for exp in experiments:
            if ranking_metric not in exp["metrics"]:
                continue

            ranking_data.append(
                {
                    "run_id": exp["run_id"],
                    "run_name": exp["run_name"],
                    "score": exp["metrics"][ranking_metric],
                }
            )

        df = pd.DataFrame(ranking_data)

        if not df.empty:
            df = df.sort_values("score", ascending=not higher_is_better)
            df["rank"] = range(1, len(df) + 1)

        return df

    def compare(
        self,
        run_ids: list[str],
        ranking_metric: str = "val_accuracy",
        higher_is_better: dict[str, bool] | None = None,
    ) -> ExperimentComparison:
        """Compare multiple experiments.

        Args:
            run_ids: List of run IDs
            ranking_metric: Metric to rank by
            higher_is_better: Whether higher is better for each metric

        Returns:
            Comparison results
        """
        # Get experiment data
        experiments = self.get_experiment_data(run_ids)

        # Compare metrics
        metrics_df = self.compare_metrics(experiments)

        # Compare parameters
        params_df = self.compare_parameters(experiments)

        # Find best metrics
        best_metrics = self.find_best_metrics(
            experiments,
            higher_is_better=higher_is_better,
        )

        # Rank experiments
        ranking_df = self.rank_experiments(
            experiments,
            ranking_metric=ranking_metric,
            higher_is_better=higher_is_better.get(ranking_metric, True)
            if higher_is_better
            else True,
        )

        # Create summary
        summary = {
            "n_experiments": len(experiments),
            "ranking_metric": ranking_metric,
            "best_run": ranking_df.iloc[0]["run_id"] if not ranking_df.empty else None,
            "best_score": ranking_df.iloc[0]["score"] if not ranking_df.empty else None,
        }

        comparison = ExperimentComparison(
            experiments=experiments,
            metrics_comparison=metrics_df,
            params_comparison=params_df,
            best_metrics=best_metrics,
            ranking=ranking_df,
            summary=summary,
        )

        LOGGER.info(
            "Compared %d experiments (best=%s)",
            len(experiments),
            summary["best_run"],
        )

        return comparison

    def generate_comparison_report(
        self,
        comparison: ExperimentComparison,
        output_path: Path | str,
    ) -> None:
        """Generate comparison report.

        Args:
            comparison: Comparison results
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            f.write("# Experiment Comparison Report\n\n")

            # Summary
            f.write("## Summary\n\n")
            f.write(f"- Total Experiments: {comparison.summary['n_experiments']}\n")
            f.write(f"- Ranking Metric: {comparison.summary['ranking_metric']}\n")
            f.write(f"- Best Run: {comparison.summary['best_run']}\n")
            f.write(f"- Best Score: {comparison.summary['best_score']:.4f}\n")
            f.write("\n")

            # Rankings
            f.write("## Rankings\n\n")
            f.write(comparison.ranking.to_markdown(index=False))
            f.write("\n\n")

            # Best Metrics
            f.write("## Best Metrics\n\n")
            for metric, info in comparison.best_metrics.items():
                f.write(f"### {metric}\n")
                f.write(f"- Value: {info['value']:.4f}\n")
                f.write(f"- Run: {info['run_name']} ({info['run_id'][:8]})\n")
                f.write(f"- Higher is Better: {info['higher_is_better']}\n")
                f.write("\n")

            # Metrics Comparison
            f.write("## Metrics Comparison\n\n")
            f.write(comparison.metrics_comparison.to_markdown())
            f.write("\n\n")

            # Parameter Comparison
            f.write("## Parameter Comparison\n\n")
            f.write(comparison.params_comparison.to_markdown())
            f.write("\n")

        LOGGER.info("Generated comparison report: %s", output_path)


def compare_experiments(
    run_ids: list[str],
    tracking_uri: str | None = None,
    ranking_metric: str = "val_accuracy",
) -> ExperimentComparison:
    """Compare experiments (convenience function).

    Args:
        run_ids: List of run IDs
        tracking_uri: MLflow tracking URI
        ranking_metric: Metric to rank by

    Returns:
        Comparison results
    """
    comparator = ExperimentComparator(tracking_uri=tracking_uri)
    return comparator.compare(
        run_ids=run_ids,
        ranking_metric=ranking_metric,
    )
