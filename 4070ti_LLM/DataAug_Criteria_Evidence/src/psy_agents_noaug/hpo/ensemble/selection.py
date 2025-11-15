#!/usr/bin/env python
"""Model selection strategies for ensemble building (Phase 11).

This module provides strategies for selecting models from HPO trials
to create ensembles or choose the best single model.

Key Features:
- Best-K selection by performance
- Diversity-based selection
- Pareto frontier selection (multi-objective)
- Greedy selection for ensemble building
- Smart thresholding strategies
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import optuna
from optuna.trial import TrialState

LOGGER = logging.getLogger(__name__)


@dataclass
class SelectedModel:
    """Information about a selected model."""

    trial_number: int
    value: float
    params: dict[str, Any]
    diversity_score: float = 0.0
    selection_reason: str = "unknown"


class ModelSelector:
    """Strategies for selecting models from HPO trials."""

    def __init__(self, study: optuna.Study):
        """Initialize model selector.

        Args:
            study: Optuna study with completed trials
        """
        self.study = study
        self.completed_trials = [
            t for t in study.trials if t.state == TrialState.COMPLETE
        ]

        if not self.completed_trials:
            raise ValueError("Study has no completed trials")

    def select_best_k(
        self,
        k: int = 5,
        metric: Literal["value", "validation_score"] = "value",
    ) -> list[SelectedModel]:
        """Select top K models by performance.

        Args:
            k: Number of models to select
            metric: Metric to use for selection

        Returns:
            List of selected models, sorted by performance
        """
        LOGGER.info("Selecting top %d models by %s", k, metric)

        # Sort trials by metric
        is_minimization = self.study.direction == optuna.study.StudyDirection.MINIMIZE
        sorted_trials = sorted(
            self.completed_trials,
            key=lambda t: t.value if t.value is not None else float("inf"),
            reverse=not is_minimization,
        )

        # Select top K
        selected = []
        for i, trial in enumerate(sorted_trials[:k]):
            selected.append(
                SelectedModel(
                    trial_number=trial.number,
                    value=trial.value,
                    params=trial.params,
                    selection_reason=f"top-{i+1} by {metric}",
                )
            )

        LOGGER.info(
            "Selected %d models (best value: %.6f)", len(selected), selected[0].value
        )
        return selected

    def select_diverse(
        self,
        k: int = 5,
        diversity_metric: Literal[
            "param_distance", "prediction_correlation"
        ] = "param_distance",
        quality_weight: float = 0.5,
    ) -> list[SelectedModel]:
        """Select diverse models using greedy diversity maximization.

        Args:
            k: Number of models to select
            diversity_metric: How to measure diversity
            quality_weight: Weight for quality vs diversity (0=only diversity, 1=only quality)

        Returns:
            List of selected models with diversity scores
        """
        LOGGER.info(
            "Selecting %d diverse models (diversity=%s, quality_weight=%.2f)",
            k,
            diversity_metric,
            quality_weight,
        )

        if len(self.completed_trials) <= k:
            # Return all trials if we don't have enough
            return [
                SelectedModel(
                    trial_number=t.number,
                    value=t.value,
                    params=t.params,
                    selection_reason="all_trials",
                )
                for t in self.completed_trials
            ]

        # Start with best model
        is_minimization = self.study.direction == optuna.study.StudyDirection.MINIMIZE
        best_trial = min(
            self.completed_trials,
            key=lambda t: t.value if t.value is not None else float("inf"),
        )

        selected = [
            SelectedModel(
                trial_number=best_trial.number,
                value=best_trial.value,
                params=best_trial.params,
                diversity_score=0.0,
                selection_reason="best_model",
            )
        ]

        # Greedily add most diverse models
        remaining_trials = [
            t for t in self.completed_trials if t.number != best_trial.number
        ]

        for _ in range(k - 1):
            if not remaining_trials:
                break

            # Compute diversity score for each remaining trial
            best_next = None
            best_score = float("-inf")

            for trial in remaining_trials:
                # Compute diversity from selected models
                diversity = self._compute_diversity(
                    trial, [self._get_trial(s.trial_number) for s in selected]
                )

                # Compute quality score (normalized)
                quality = self._normalize_quality(trial.value)

                # Combined score
                combined_score = (
                    quality_weight * quality + (1 - quality_weight) * diversity
                )

                if combined_score > best_score:
                    best_score = combined_score
                    best_next = trial

            if best_next:
                selected.append(
                    SelectedModel(
                        trial_number=best_next.number,
                        value=best_next.value,
                        params=best_next.params,
                        diversity_score=best_score,
                        selection_reason=f"diverse_{len(selected)}",
                    )
                )
                remaining_trials.remove(best_next)

        LOGGER.info("Selected %d diverse models", len(selected))
        return selected

    def select_by_threshold(
        self,
        threshold: float | None = None,
        relative_threshold: float = 0.1,
        max_models: int = 10,
    ) -> list[SelectedModel]:
        """Select all models within a performance threshold.

        Args:
            threshold: Absolute performance threshold (if None, use relative)
            relative_threshold: Relative threshold from best (e.g., 0.1 = within 10% of best)
            max_models: Maximum number of models to select

        Returns:
            List of models within threshold
        """
        is_minimization = self.study.direction == optuna.study.StudyDirection.MINIMIZE
        best_value = self.study.best_value

        if threshold is None:
            # Compute relative threshold
            if is_minimization:
                threshold = best_value * (1 + relative_threshold)
            else:
                threshold = best_value * (1 - relative_threshold)

        LOGGER.info(
            "Selecting models within threshold %.6f (best=%.6f, relative=%.2f)",
            threshold,
            best_value,
            relative_threshold,
        )

        # Select models within threshold
        selected = []
        for trial in self.completed_trials:
            if trial.value is None:
                continue

            # Check if within threshold
            if is_minimization:
                within_threshold = trial.value <= threshold
            else:
                within_threshold = trial.value >= threshold

            if within_threshold:
                selected.append(
                    SelectedModel(
                        trial_number=trial.number,
                        value=trial.value,
                        params=trial.params,
                        selection_reason=f"within_threshold_{threshold:.6f}",
                    )
                )

        # Sort by value and limit to max_models
        selected.sort(key=lambda x: x.value, reverse=not is_minimization)
        selected = selected[:max_models]

        LOGGER.info(
            "Selected %d models within threshold (range: %.6f - %.6f)",
            len(selected),
            selected[0].value if selected else 0.0,
            selected[-1].value if selected else 0.0,
        )
        return selected

    def select_pareto_frontier(
        self,
        objectives: list[str] = ["value", "inference_time"],
    ) -> list[SelectedModel]:
        """Select models on the Pareto frontier (multi-objective).

        Args:
            objectives: List of objectives to consider

        Returns:
            Models on the Pareto frontier
        """
        LOGGER.info("Selecting Pareto frontier for objectives: %s", objectives)

        # Extract objective values for each trial
        trial_objectives = []
        for trial in self.completed_trials:
            obj_values = []
            for obj in objectives:
                if obj == "value":
                    obj_values.append(trial.value)
                else:
                    # Try to get from user_attrs
                    obj_values.append(trial.user_attrs.get(obj, float("inf")))
            trial_objectives.append((trial, obj_values))

        # Find Pareto frontier
        pareto_trials = []
        for trial, obj_values in trial_objectives:
            # Check if this trial is dominated by any other
            is_dominated = False
            for other_trial, other_obj_values in trial_objectives:
                if other_trial.number == trial.number:
                    continue

                # Check if other dominates this trial
                # (better or equal on all objectives, strictly better on at least one)
                all_better_or_equal = all(
                    o1 <= o2
                    for o1, o2 in zip(other_obj_values, obj_values, strict=False)
                )
                at_least_one_better = any(
                    o1 < o2
                    for o1, o2 in zip(other_obj_values, obj_values, strict=False)
                )

                if all_better_or_equal and at_least_one_better:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_trials.append((trial, obj_values))

        # Create selected models
        selected = []
        for trial, obj_values in pareto_trials:
            selected.append(
                SelectedModel(
                    trial_number=trial.number,
                    value=trial.value,
                    params=trial.params,
                    selection_reason=f"pareto_frontier_{objectives}",
                )
            )

        LOGGER.info("Found %d models on Pareto frontier", len(selected))
        return selected

    def _compute_diversity(
        self,
        trial: optuna.trial.FrozenTrial,
        selected_trials: list[optuna.trial.FrozenTrial],
    ) -> float:
        """Compute diversity of trial from selected trials.

        Uses parameter space distance as diversity metric.

        Args:
            trial: Trial to evaluate
            selected_trials: Currently selected trials

        Returns:
            Diversity score (higher = more diverse)
        """
        if not selected_trials:
            return 1.0

        # Compute minimum distance to any selected trial
        min_distance = float("inf")

        for selected in selected_trials:
            distance = self._param_distance(trial.params, selected.params)
            min_distance = min(min_distance, distance)

        # Normalize to [0, 1]
        return min(1.0, min_distance / 10.0)  # Assuming max distance ~10

    def _param_distance(
        self, params1: dict[str, Any], params2: dict[str, Any]
    ) -> float:
        """Compute distance between parameter configurations.

        Args:
            params1: First parameter dict
            params2: Second parameter dict

        Returns:
            Distance metric
        """
        # Get common parameters
        common_params = set(params1.keys()) & set(params2.keys())

        if not common_params:
            return float("inf")

        # Compute normalized distance
        total_distance = 0.0
        for param in common_params:
            v1, v2 = params1[param], params2[param]

            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # Numeric distance (normalized)
                total_distance += abs(v1 - v2)
            elif v1 != v2:
                # Categorical distance
                total_distance += 1.0

        return total_distance / len(common_params)

    def _normalize_quality(self, value: float) -> float:
        """Normalize quality score to [0, 1].

        Args:
            value: Trial value

        Returns:
            Normalized quality score (1 = best)
        """
        if not self.completed_trials:
            return 0.0

        values = [t.value for t in self.completed_trials if t.value is not None]
        if not values:
            return 0.0

        min_val = min(values)
        max_val = max(values)

        if min_val == max_val:
            return 1.0

        # Normalize based on direction
        is_minimization = self.study.direction == optuna.study.StudyDirection.MINIMIZE
        if is_minimization:
            return 1.0 - (value - min_val) / (max_val - min_val)
        return (value - min_val) / (max_val - min_val)

    def _get_trial(self, trial_number: int) -> optuna.trial.FrozenTrial:
        """Get trial by number.

        Args:
            trial_number: Trial number

        Returns:
            Trial object
        """
        for trial in self.study.trials:
            if trial.number == trial_number:
                return trial
        raise ValueError(f"Trial {trial_number} not found")


def recommend_selection_strategy(
    n_trials: int,
    objective: Literal["single_best", "ensemble", "robust"] = "single_best",
    compute_budget: Literal["low", "medium", "high"] = "medium",
) -> tuple[str, dict[str, Any]]:
    """Recommend a model selection strategy based on scenario.

    Args:
        n_trials: Number of completed trials
        objective: Selection objective
        compute_budget: Available compute for inference

    Returns:
        (strategy_name, strategy_kwargs)

    Examples:
        >>> strategy, kwargs = recommend_selection_strategy(100, "ensemble", "high")
        >>> # Returns: ("select_diverse", {"k": 5, "quality_weight": 0.7})
    """
    if objective == "single_best":
        # Just want the best model
        return ("select_best_k", {"k": 1})

    if objective == "ensemble":
        # Want to build an ensemble
        if compute_budget == "low":
            # Small ensemble
            return ("select_best_k", {"k": 3})
        if compute_budget == "medium":
            # Moderate diverse ensemble
            return ("select_diverse", {"k": 5, "quality_weight": 0.7})
        # Large diverse ensemble
        return ("select_diverse", {"k": 10, "quality_weight": 0.6})

    if objective == "robust":
        # Want robust selection (within threshold)
        if n_trials < 20:
            return ("select_best_k", {"k": 3})
        return ("select_by_threshold", {"relative_threshold": 0.05, "max_models": 10})

    return ("select_best_k", {"k": 5})
