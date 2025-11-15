#!/usr/bin/env python
"""Report generation for HPO results (Phase 13).

This module provides tools for generating comprehensive reports
from HPO studies.

Key Features:
- HTML report generation
- Study comparison reports
- Summary statistics export
- Multi-study analysis
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState

from psy_agents_noaug.hpo.visualization.analysis import (
    ConvergenceAnalyzer,
    ParameterAnalyzer,
)
from psy_agents_noaug.hpo.visualization.plots import HPOVisualizer

LOGGER = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive HPO reports."""

    def __init__(self, study: optuna.Study):
        """Initialize report generator.

        Args:
            study: Optuna study
        """
        self.study = study
        self.visualizer = HPOVisualizer(study)

    def generate_report(
        self,
        output_dir: Path | str,
        include_plots: bool = True,
        plot_format: str = "png",
    ) -> Path:
        """Generate comprehensive HTML report.

        Args:
            output_dir: Output directory
            include_plots: Whether to include plots
            plot_format: Plot image format

        Returns:
            Path to generated report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Generating report to: %s", output_dir)

        # Generate plots
        plot_paths = {}
        if include_plots:
            plots_dir = output_dir / "plots"
            plot_paths = self.visualizer.generate_all_plots(
                output_dir=plots_dir,
                format=plot_format,
            )

        # Analyze parameters
        try:
            analyzer = ParameterAnalyzer(self.study)
            param_analysis = analyzer.analyze_all_parameters()
        except Exception as e:
            LOGGER.warning("Failed to analyze parameters: %s", e)
            param_analysis = []

        # Analyze convergence
        try:
            conv_analyzer = ConvergenceAnalyzer(self.study)
            convergence = conv_analyzer.analyze_convergence()
        except Exception as e:
            LOGGER.warning("Failed to analyze convergence: %s", e)
            convergence = None

        # Generate HTML
        html_path = output_dir / "report.html"
        html_content = self._generate_html(
            plot_paths=plot_paths,
            param_analysis=param_analysis,
            convergence=convergence,
        )

        with html_path.open("w") as f:
            f.write(html_content)

        LOGGER.info("Report generated: %s", html_path)
        return html_path

    def _generate_html(
        self,
        plot_paths: dict[str, Path],
        param_analysis: list,
        convergence: Any,
    ) -> str:
        """Generate HTML report content.

        Args:
            plot_paths: Dict of plot paths
            param_analysis: Parameter analysis results
            convergence: Convergence analysis result

        Returns:
            HTML content
        """
        completed = [t for t in self.study.trials if t.state == TrialState.COMPLETE]
        failed = [t for t in self.study.trials if t.state == TrialState.FAIL]
        pruned = [t for t in self.study.trials if t.state == TrialState.PRUNED]

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>HPO Report: {self.study.study_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .plot {{ margin: 20px 0; }}
        .plot img {{ max-width: 100%; height: auto; }}
        .stat {{ background-color: #f2f2f2; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>HPO Report: {self.study.study_name}</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <h2>Study Summary</h2>
    <div class="stat">
        <p><strong>Total Trials:</strong> {len(self.study.trials)}</p>
        <p><strong>Completed:</strong> {len(completed)}</p>
        <p><strong>Failed:</strong> {len(failed)}</p>
        <p><strong>Pruned:</strong> {len(pruned)}</p>
        <p><strong>Direction:</strong> {self.study.direction.name}</p>
        <p><strong>Best Value:</strong> {self.study.best_value:.6f} (Trial {self.study.best_trial.number})</p>
    </div>

    <h2>Best Trial</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
"""

        # Best trial parameters
        for param, value in self.study.best_trial.params.items():
            html += f"        <tr><td>{param}</td><td>{value}</td></tr>\n"

        html += "    </table>\n"

        # Convergence analysis
        if convergence:
            html += f"""
    <h2>Convergence Analysis</h2>
    <div class="stat">
        <p><strong>Converged:</strong> {"Yes" if convergence.is_converged else "No"}</p>
        <p><strong>Convergence Trial:</strong> {convergence.convergence_trial if convergence.convergence_trial else "N/A"}</p>
        <p><strong>Plateau Length:</strong> {convergence.plateau_length} trials</p>
        <p><strong>Improvement Rate:</strong> {convergence.improvement_rate:.6f}</p>
    </div>
"""

        # Parameter analysis
        if param_analysis:
            html += """
    <h2>Parameter Analysis</h2>
    <table>
        <tr>
            <th>Parameter</th>
            <th>Importance</th>
            <th>Correlation</th>
            <th>Best Value</th>
            <th>Range</th>
        </tr>
"""
            for result in param_analysis:
                # Format range based on type
                if result.mean_value is not None:
                    range_str = f"{result.value_range[0]:.4f} - {result.value_range[1]:.4f}"
                else:
                    range_str = f"{result.value_range[0]} - {result.value_range[1]}"

                html += f"""        <tr>
            <td>{result.param_name}</td>
            <td>{result.importance:.4f}</td>
            <td>{result.correlation_with_objective:.4f}</td>
            <td>{result.best_value}</td>
            <td>{range_str}</td>
        </tr>
"""
            html += "    </table>\n"

        # Plots
        if plot_paths:
            html += "    <h2>Visualizations</h2>\n"
            for plot_name, plot_path in plot_paths.items():
                rel_path = plot_path.relative_to(plot_path.parent.parent)
                html += f"""
    <div class="plot">
        <h3>{plot_name.replace("_", " ").title()}</h3>
        <img src="{rel_path}" alt="{plot_name}">
    </div>
"""

        html += """
</body>
</html>
"""
        return html


class StudyComparator:
    """Compare multiple HPO studies."""

    def __init__(self, studies: list[optuna.Study]):
        """Initialize comparator.

        Args:
            studies: List of Optuna studies to compare
        """
        self.studies = studies

        if len(studies) < 2:
            raise ValueError("Need at least 2 studies to compare")

    def compare_studies(self) -> dict[str, Any]:
        """Compare studies and generate summary.

        Returns:
            Comparison summary
        """
        LOGGER.info("Comparing %d studies", len(self.studies))

        comparison = {
            "n_studies": len(self.studies),
            "studies": [],
        }

        for study in self.studies:
            completed = [t for t in study.trials if t.state == TrialState.COMPLETE]

            study_info = {
                "name": study.study_name,
                "n_trials": len(study.trials),
                "n_completed": len(completed),
                "best_value": study.best_value if completed else None,
                "best_trial": study.best_trial.number if completed else None,
                "direction": study.direction.name,
            }

            comparison["studies"].append(study_info)

        return comparison

    def export_comparison(self, output_path: Path | str) -> None:
        """Export comparison to JSON.

        Args:
            output_path: Output file path
        """
        comparison = self.compare_studies()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump(comparison, f, indent=2)

        LOGGER.info("Exported comparison to: %s", output_path)


def export_study_summary(
    study: optuna.Study,
    output_path: Path | str,
) -> None:
    """Export study summary to JSON.

    Args:
        study: Optuna study
        output_path: Output file path
    """
    LOGGER.info("Exporting study summary")

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    failed = [t for t in study.trials if t.state == TrialState.FAIL]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]

    summary = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "n_trials": len(study.trials),
        "n_completed": len(completed),
        "n_failed": len(failed),
        "n_pruned": len(pruned),
        "best_value": study.best_value if completed else None,
        "best_trial": study.best_trial.number if completed else None,
        "best_params": study.best_trial.params if completed else None,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info("Exported summary to: %s", output_path)
