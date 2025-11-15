#!/usr/bin/env python
"""Analysis tools for HPO results (Phase 13).

This module provides analytical tools for understanding hyperparameter
optimization results.

Key Features:
- Parameter correlation analysis
- Hyperparameter interaction analysis
- Convergence analysis
- Statistical significance testing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
from optuna.importance import FanovaImportanceEvaluator
from optuna.trial import TrialState
from scipy.stats import pearsonr

LOGGER = logging.getLogger(__name__)


@dataclass
class ParameterAnalysisResult:
    """Results from parameter analysis."""

    param_name: str
    importance: float
    correlation_with_objective: float
    value_range: tuple[float, float]
    best_value: Any
    mean_value: float | None = None
    std_value: float | None = None


@dataclass
class ConvergenceAnalysisResult:
    """Results from convergence analysis."""

    is_converged: bool
    convergence_trial: int | None
    improvement_rate: float
    plateau_length: int
    total_trials: int
    best_value: float
    best_trial: int


class ParameterAnalyzer:
    """Analyze hyperparameter behavior."""

    def __init__(self, study: optuna.Study):
        """Initialize analyzer.

        Args:
            study: Optuna study
        """
        self.study = study
        self.completed_trials = [
            t for t in study.trials if t.state == TrialState.COMPLETE
        ]

        if not self.completed_trials:
            raise ValueError("Study has no completed trials")

    def analyze_parameter(self, param_name: str) -> ParameterAnalysisResult:
        """Analyze a single parameter.

        Args:
            param_name: Parameter name

        Returns:
            Analysis result
        """
        LOGGER.info("Analyzing parameter: %s", param_name)

        # Extract parameter values and objectives
        param_values = []
        objectives = []

        for trial in self.completed_trials:
            if param_name in trial.params:
                param_values.append(trial.params[param_name])
                objectives.append(trial.value)

        if not param_values:
            raise ValueError(f"Parameter {param_name} not found in trials")

        param_values = np.array(param_values)
        objectives = np.array(objectives)

        # Compute correlation
        if len(param_values) > 1:
            try:
                correlation, _ = pearsonr(param_values, objectives)
            except Exception:
                correlation = 0.0
        else:
            correlation = 0.0

        # Compute importance
        try:
            evaluator = FanovaImportanceEvaluator()
            importances = evaluator.evaluate(self.study)
            importance = importances.get(param_name, 0.0)
        except Exception as e:
            LOGGER.warning("Failed to compute importance: %s", e)
            importance = 0.0

        # Best value
        best_trial = self.study.best_trial
        best_value = best_trial.params.get(param_name)

        # Statistics
        if isinstance(param_values[0], (int, float)):
            mean_value = float(np.mean(param_values))
            std_value = float(np.std(param_values))
            value_range = (float(np.min(param_values)), float(np.max(param_values)))
        else:
            mean_value = None
            std_value = None
            unique_values = list(set(param_values))
            value_range = (min(unique_values), max(unique_values))

        return ParameterAnalysisResult(
            param_name=param_name,
            importance=importance,
            correlation_with_objective=correlation,
            value_range=value_range,
            best_value=best_value,
            mean_value=mean_value,
            std_value=std_value,
        )

    def analyze_all_parameters(self) -> list[ParameterAnalysisResult]:
        """Analyze all parameters.

        Returns:
            List of analysis results
        """
        # Get all parameter names
        param_names = set()
        for trial in self.completed_trials:
            param_names.update(trial.params.keys())

        results = []
        for param_name in sorted(param_names):
            try:
                result = self.analyze_parameter(param_name)
                results.append(result)
            except Exception as e:
                LOGGER.warning("Failed to analyze parameter %s: %s", param_name, e)

        # Sort by importance
        results.sort(key=lambda x: x.importance, reverse=True)

        return results

    def compute_parameter_correlation_matrix(self) -> dict[str, dict[str, float]]:
        """Compute correlation matrix between parameters.

        Returns:
            Dict of parameter correlations
        """
        LOGGER.info("Computing parameter correlation matrix")

        # Extract parameter values
        param_names = set()
        for trial in self.completed_trials:
            param_names.update(trial.params.keys())

        param_names = sorted(param_names)

        # Build correlation matrix
        correlations = {}
        for param1 in param_names:
            correlations[param1] = {}

            # Extract values for param1
            values1 = []
            for trial in self.completed_trials:
                if param1 in trial.params:
                    val = trial.params[param1]
                    # Convert to numeric if possible
                    if isinstance(val, (int, float)):
                        values1.append(float(val))
                    else:
                        # Skip non-numeric parameters
                        break
            else:
                # Compute correlations with other parameters
                for param2 in param_names:
                    if param1 == param2:
                        correlations[param1][param2] = 1.0
                        continue

                    values2 = []
                    for trial in self.completed_trials:
                        if param2 in trial.params:
                            val = trial.params[param2]
                            if isinstance(val, (int, float)):
                                values2.append(float(val))
                            else:
                                break
                    else:
                        # Compute correlation
                        if len(values1) == len(values2) and len(values1) > 1:
                            try:
                                corr, _ = pearsonr(values1, values2)
                                correlations[param1][param2] = corr
                            except Exception:
                                correlations[param1][param2] = 0.0
                        else:
                            correlations[param1][param2] = 0.0

        return correlations


class ConvergenceAnalyzer:
    """Analyze HPO convergence."""

    def __init__(
        self,
        study: optuna.Study,
        window_size: int = 10,
        improvement_threshold: float = 0.001,
    ):
        """Initialize convergence analyzer.

        Args:
            study: Optuna study
            window_size: Window size for plateau detection
            improvement_threshold: Threshold for considering improvement
        """
        self.study = study
        self.window_size = window_size
        self.improvement_threshold = improvement_threshold

        self.completed_trials = [
            t for t in study.trials if t.state == TrialState.COMPLETE
        ]

        if not self.completed_trials:
            raise ValueError("Study has no completed trials")

    def analyze_convergence(self) -> ConvergenceAnalysisResult:
        """Analyze study convergence.

        Returns:
            Convergence analysis result
        """
        LOGGER.info("Analyzing convergence")

        # Extract values over time
        values = [t.value for t in self.completed_trials]
        is_minimization = self.study.direction == optuna.study.StudyDirection.MINIMIZE

        # Track best value over time
        best_values = []
        current_best = values[0]

        for value in values:
            if is_minimization:
                current_best = min(current_best, value)
            else:
                current_best = max(current_best, value)
            best_values.append(current_best)

        # Compute improvement rate
        if len(best_values) > 1:
            total_improvement = abs(best_values[-1] - best_values[0])
            improvement_rate = total_improvement / len(best_values)
        else:
            improvement_rate = 0.0

        # Detect plateau
        plateau_length = 0
        for i in range(len(best_values) - 1, 0, -1):
            improvement = abs(best_values[i] - best_values[i - 1])
            if improvement < self.improvement_threshold:
                plateau_length += 1
            else:
                break

        # Determine convergence
        is_converged = plateau_length >= self.window_size

        # Find convergence trial
        if is_converged:
            convergence_trial = len(best_values) - plateau_length
        else:
            convergence_trial = None

        return ConvergenceAnalysisResult(
            is_converged=is_converged,
            convergence_trial=convergence_trial,
            improvement_rate=improvement_rate,
            plateau_length=plateau_length,
            total_trials=len(self.completed_trials),
            best_value=self.study.best_value,
            best_trial=self.study.best_trial.number,
        )

    def should_stop_optimization(self) -> tuple[bool, str]:
        """Determine if optimization should stop.

        Returns:
            (should_stop, reason)
        """
        result = self.analyze_convergence()

        if result.is_converged:
            return (
                True,
                f"Converged after {result.convergence_trial} trials "
                f"(plateau: {result.plateau_length} trials)",
            )

        if result.improvement_rate < self.improvement_threshold:
            return (
                True,
                f"Low improvement rate: {result.improvement_rate:.6f}",
            )

        return (False, "Optimization should continue")


def analyze_hyperparameter_interactions(
    study: optuna.Study,
    param_pairs: list[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], float]:
    """Analyze interactions between hyperparameters.

    Args:
        study: Optuna study
        param_pairs: Pairs of parameters to analyze (None = all pairs)

    Returns:
        Dict mapping parameter pairs to interaction strength
    """
    LOGGER.info("Analyzing hyperparameter interactions")

    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if not completed_trials:
        raise ValueError("Study has no completed trials")

    # Get all parameter names
    param_names = set()
    for trial in completed_trials:
        param_names.update(trial.params.keys())

    param_names = sorted(param_names)

    # Generate pairs if not provided
    if param_pairs is None:
        param_pairs = [
            (p1, p2) for i, p1 in enumerate(param_names) for p2 in param_names[i + 1 :]
        ]

    interactions = {}

    for param1, param2 in param_pairs:
        # Extract parameter values
        values1 = []
        values2 = []
        objectives = []

        for trial in completed_trials:
            if param1 in trial.params and param2 in trial.params:
                val1 = trial.params[param1]
                val2 = trial.params[param2]

                # Only analyze numeric parameters
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    values1.append(float(val1))
                    values2.append(float(val2))
                    objectives.append(trial.value)

        # Compute interaction strength
        if len(values1) > 3:
            try:
                # Simple interaction: correlation between product and objective
                product = np.array(values1) * np.array(values2)
                corr, _ = pearsonr(product, objectives)
                interactions[(param1, param2)] = abs(corr)
            except Exception:
                interactions[(param1, param2)] = 0.0
        else:
            interactions[(param1, param2)] = 0.0

    # Sort by interaction strength
    interactions = dict(sorted(interactions.items(), key=lambda x: x[1], reverse=True))

    LOGGER.info("Analyzed %d parameter interactions", len(interactions))

    return interactions
