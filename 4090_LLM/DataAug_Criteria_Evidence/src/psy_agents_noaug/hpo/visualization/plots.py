#!/usr/bin/env python
"""Visualization plots for HPO results (Phase 13).

This module provides plotting functions for visualizing hyperparameter
optimization results.

Key Features:
- Optimization history plots
- Parameter importance plots
- Parallel coordinates plots
- Slice plots for hyperparameter analysis
- Contour plots for 2D parameter space
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
from optuna.importance import FanovaImportanceEvaluator
from optuna.trial import TrialState
from optuna.visualization import (
    plot_optimization_history as optuna_plot_history,
)
from optuna.visualization import (
    plot_parallel_coordinate as optuna_plot_parallel,
)
from optuna.visualization import (
    plot_param_importances as optuna_plot_importance,
)

LOGGER = logging.getLogger(__name__)


class HPOVisualizer:
    """Comprehensive visualizer for HPO studies."""

    def __init__(
        self,
        study: optuna.Study,
        figsize: tuple[int, int] = (12, 8),
    ):
        """Initialize visualizer.

        Args:
            study: Optuna study to visualize
            figsize: Figure size for plots
        """
        self.study = study
        self.figsize = figsize

        # Filter completed trials
        self.completed_trials = [
            t for t in study.trials if t.state == TrialState.COMPLETE
        ]

        if not self.completed_trials:
            LOGGER.warning("Study has no completed trials")

    def plot_optimization_history(
        self,
        save_path: Path | str | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """Plot optimization history over trials.

        Args:
            save_path: Path to save figure (None = don't save)
            show: Whether to show plot interactively

        Returns:
            Matplotlib figure
        """
        LOGGER.info("Plotting optimization history")

        fig = optuna_plot_history(self.study)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(save_path))
            LOGGER.info("Saved optimization history to: %s", save_path)

        if show:
            fig.show()

        return fig

    def plot_param_importance(
        self,
        evaluator: str = "fanova",
        save_path: Path | str | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """Plot parameter importance.

        Args:
            evaluator: Importance evaluator ("fanova" or "default")
            save_path: Path to save figure
            show: Whether to show plot

        Returns:
            Matplotlib figure
        """
        LOGGER.info("Plotting parameter importance (evaluator=%s)", evaluator)

        if len(self.completed_trials) < 2:
            LOGGER.warning("Need at least 2 completed trials for importance")
            return None

        # Compute importance
        if evaluator == "fanova":
            evaluator_obj = FanovaImportanceEvaluator()
        else:
            evaluator_obj = None

        fig = optuna_plot_importance(self.study, evaluator=evaluator_obj)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(save_path))
            LOGGER.info("Saved parameter importance to: %s", save_path)

        if show:
            fig.show()

        return fig

    def plot_parallel_coordinates(
        self,
        params: list[str] | None = None,
        save_path: Path | str | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """Plot parallel coordinates.

        Args:
            params: Parameters to include (None = all)
            save_path: Path to save figure
            show: Whether to show plot

        Returns:
            Matplotlib figure
        """
        LOGGER.info("Plotting parallel coordinates")

        if len(self.completed_trials) == 0:
            LOGGER.warning("No completed trials to plot")
            return None

        fig = optuna_plot_parallel(self.study, params=params)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(save_path))
            LOGGER.info("Saved parallel coordinates to: %s", save_path)

        if show:
            fig.show()

        return fig

    def plot_slice(
        self,
        params: list[str] | None = None,
        save_path: Path | str | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """Plot parameter slices.

        Args:
            params: Parameters to plot (None = all)
            save_path: Path to save figure
            show: Whether to show plot

        Returns:
            Matplotlib figure
        """
        from optuna.visualization import plot_slice

        LOGGER.info("Plotting parameter slices")

        if len(self.completed_trials) == 0:
            LOGGER.warning("No completed trials to plot")
            return None

        fig = plot_slice(self.study, params=params)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(save_path))
            LOGGER.info("Saved parameter slices to: %s", save_path)

        if show:
            fig.show()

        return fig

    def plot_contour(
        self,
        params: list[str] | None = None,
        save_path: Path | str | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """Plot contour plot for parameter space.

        Args:
            params: Parameters to plot (must be 2)
            save_path: Path to save figure
            show: Whether to show plot

        Returns:
            Matplotlib figure
        """
        from optuna.visualization import plot_contour

        LOGGER.info("Plotting contour plot")

        if len(self.completed_trials) == 0:
            LOGGER.warning("No completed trials to plot")
            return None

        fig = plot_contour(self.study, params=params)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(save_path))
            LOGGER.info("Saved contour plot to: %s", save_path)

        if show:
            fig.show()

        return fig

    def plot_timeline(
        self,
        save_path: Path | str | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """Plot trial timeline.

        Args:
            save_path: Path to save figure
            show: Whether to show plot

        Returns:
            Matplotlib figure
        """
        from optuna.visualization import plot_timeline

        LOGGER.info("Plotting trial timeline")

        if len(self.study.trials) == 0:
            LOGGER.warning("No trials to plot")
            return None

        fig = plot_timeline(self.study)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(save_path))
            LOGGER.info("Saved timeline to: %s", save_path)

        if show:
            fig.show()

        return fig

    def generate_all_plots(
        self,
        output_dir: Path | str,
        format: str = "png",
    ) -> dict[str, Path]:
        """Generate all available plots.

        Args:
            output_dir: Output directory for plots
            format: Image format (png, svg, pdf)

        Returns:
            Dict mapping plot name to saved path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Generating all plots to: %s", output_dir)

        saved_plots = {}

        # Optimization history
        try:
            path = output_dir / f"optimization_history.{format}"
            self.plot_optimization_history(save_path=path)
            saved_plots["optimization_history"] = path
        except Exception as e:
            LOGGER.warning("Failed to plot optimization history: %s", e)

        # Parameter importance
        try:
            path = output_dir / f"param_importance.{format}"
            self.plot_param_importance(save_path=path)
            saved_plots["param_importance"] = path
        except Exception as e:
            LOGGER.warning("Failed to plot parameter importance: %s", e)

        # Parallel coordinates
        try:
            path = output_dir / f"parallel_coordinates.{format}"
            self.plot_parallel_coordinates(save_path=path)
            saved_plots["parallel_coordinates"] = path
        except Exception as e:
            LOGGER.warning("Failed to plot parallel coordinates: %s", e)

        # Slice plot
        try:
            path = output_dir / f"param_slices.{format}"
            self.plot_slice(save_path=path)
            saved_plots["param_slices"] = path
        except Exception as e:
            LOGGER.warning("Failed to plot parameter slices: %s", e)

        # Timeline
        try:
            path = output_dir / f"timeline.{format}"
            self.plot_timeline(save_path=path)
            saved_plots["timeline"] = path
        except Exception as e:
            LOGGER.warning("Failed to plot timeline: %s", e)

        LOGGER.info("Generated %d plots", len(saved_plots))
        return saved_plots


# Standalone plotting functions
def plot_optimization_history(
    study: optuna.Study,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot optimization history (standalone function).

    Args:
        study: Optuna study
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    visualizer = HPOVisualizer(study)
    return visualizer.plot_optimization_history(save_path=save_path)


def plot_param_importance(
    study: optuna.Study,
    evaluator: str = "fanova",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot parameter importance (standalone function).

    Args:
        study: Optuna study
        evaluator: Importance evaluator
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    visualizer = HPOVisualizer(study)
    return visualizer.plot_param_importance(evaluator=evaluator, save_path=save_path)


def plot_parallel_coordinates(
    study: optuna.Study,
    params: list[str] | None = None,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot parallel coordinates (standalone function).

    Args:
        study: Optuna study
        params: Parameters to include
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    visualizer = HPOVisualizer(study)
    return visualizer.plot_parallel_coordinates(params=params, save_path=save_path)
