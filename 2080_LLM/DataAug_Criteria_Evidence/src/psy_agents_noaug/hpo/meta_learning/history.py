#!/usr/bin/env python
"""Trial history analysis for meta-learning (Phase 10).

This module provides utilities for analyzing completed HPO studies
to extract knowledge for warm-starting and transfer learning.

Key Features:
- Extract best configurations from historical studies
- Analyze parameter importance across trials
- Identify convergence patterns
- Compute similarity between studies
- Export knowledge for reuse
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import optuna
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.study import StudyDirection

LOGGER = logging.getLogger(__name__)


@dataclass
class TrialSummary:
    """Summary of a single trial."""

    trial_number: int
    value: float
    params: dict[str, Any]
    state: str
    duration: float
    datetime_start: str
    datetime_complete: str | None = None
    user_attrs: dict[str, Any] = field(default_factory=dict)
    system_attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class StudyAnalysis:
    """Analysis results for a completed study."""

    study_name: str
    direction: str
    n_trials: int
    n_completed_trials: int
    n_pruned_trials: int
    n_failed_trials: int
    best_value: float
    best_params: dict[str, Any]
    best_trial_number: int
    param_importance: dict[str, float]
    convergence_iteration: int | None
    trials: list[TrialSummary]
    datetime_start: str
    datetime_complete: str | None = None


class TrialHistoryAnalyzer:
    """Analyze trial history from Optuna studies."""

    def __init__(self, storage: str | None = None):
        """Initialize analyzer.

        Args:
            storage: Optuna storage URL (e.g., "sqlite:///optuna.db")
                    If None, uses in-memory storage.
        """
        self.storage = storage

    def load_study(self, study_name: str) -> optuna.Study:
        """Load a study from storage.

        Args:
            study_name: Name of the study to load

        Returns:
            Loaded Optuna study

        Raises:
            KeyError: If study doesn't exist
        """
        if self.storage is None:
            raise ValueError("Cannot load study without storage URL")

        try:
            study = optuna.load_study(study_name=study_name, storage=self.storage)
            LOGGER.info(
                "Loaded study '%s' with %d trials", study_name, len(study.trials)
            )
            return study
        except KeyError:
            LOGGER.error("Study '%s' not found in storage", study_name)
            raise

    def analyze_study(
        self,
        study: optuna.Study | str,
        compute_importance: bool = True,
        importance_evaluator: Literal["fanova", "default"] = "default",
    ) -> StudyAnalysis:
        """Analyze a completed study.

        Args:
            study: Study object or study name (if name, will load from storage)
            compute_importance: Whether to compute parameter importance
            importance_evaluator: Which importance evaluator to use

        Returns:
            StudyAnalysis with extracted knowledge
        """
        # Load study if name provided
        if isinstance(study, str):
            study = self.load_study(study)

        # Extract trial summaries
        trials = []
        for trial in study.trials:
            trials.append(
                TrialSummary(
                    trial_number=trial.number,
                    value=trial.value if trial.value is not None else float("inf"),
                    params=trial.params,
                    state=trial.state.name,
                    duration=trial.duration.total_seconds() if trial.duration else 0.0,
                    datetime_start=(
                        trial.datetime_start.isoformat() if trial.datetime_start else ""
                    ),
                    datetime_complete=(
                        trial.datetime_complete.isoformat()
                        if trial.datetime_complete
                        else None
                    ),
                    user_attrs=dict(trial.user_attrs),
                    system_attrs=dict(trial.system_attrs),
                )
            )

        # Compute statistics
        n_completed = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        n_pruned = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        )
        n_failed = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL
        )

        # Get best trial
        try:
            best_trial = study.best_trial
            best_value = best_trial.value
            best_params = best_trial.params
            best_trial_number = best_trial.number
        except ValueError:
            # No completed trials
            best_value = float("inf")
            best_params = {}
            best_trial_number = -1

        # Compute parameter importance
        param_importance = {}
        if compute_importance and n_completed > 1:
            try:
                if importance_evaluator == "fanova":
                    evaluator = FanovaImportanceEvaluator()
                    param_importance = get_param_importances(study, evaluator=evaluator)
                else:
                    param_importance = get_param_importances(study)

                LOGGER.info(
                    "Parameter importance: %s",
                    {k: f"{v:.3f}" for k, v in param_importance.items()},
                )
            except Exception as e:
                LOGGER.warning("Failed to compute parameter importance: %s", e)

        # Detect convergence (when best value stopped improving significantly)
        convergence_iteration = self._detect_convergence(study)

        # Get study metadata
        datetime_start = (
            min(t.datetime_start for t in trials if t.datetime_start) if trials else ""
        )
        datetime_complete = (
            max(t.datetime_complete for t in trials if t.datetime_complete)
            if trials
            else None
        )

        return StudyAnalysis(
            study_name=study.study_name,
            direction=study.direction.name,
            n_trials=len(study.trials),
            n_completed_trials=n_completed,
            n_pruned_trials=n_pruned,
            n_failed_trials=n_failed,
            best_value=best_value,
            best_params=best_params,
            best_trial_number=best_trial_number,
            param_importance=param_importance,
            convergence_iteration=convergence_iteration,
            trials=trials,
            datetime_start=datetime_start,
            datetime_complete=datetime_complete,
        )

    def _detect_convergence(
        self,
        study: optuna.Study,
        patience: int = 20,
        min_delta: float = 0.001,
    ) -> int | None:
        """Detect when study converged (stopped improving).

        Args:
            study: Study to analyze
            patience: Number of trials without improvement to declare convergence
            min_delta: Minimum improvement to count as significant

        Returns:
            Trial number where convergence occurred, or None if not converged
        """
        if len(study.trials) < patience:
            return None

        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if len(completed_trials) < patience:
            return None

        # Track best value seen so far
        is_minimization = study.direction == StudyDirection.MINIMIZE
        best_value = float("inf") if is_minimization else float("-inf")
        no_improvement_count = 0
        convergence_trial = None

        for trial in completed_trials:
            if trial.value is None:
                continue

            # Check if this is an improvement
            if is_minimization:
                is_improvement = trial.value < (best_value - min_delta)
                best_value = min(best_value, trial.value)
            else:
                is_improvement = trial.value > (best_value + min_delta)
                best_value = max(best_value, trial.value)

            if is_improvement:
                no_improvement_count = 0
                convergence_trial = None
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience and convergence_trial is None:
                    convergence_trial = trial.number

        return convergence_trial

    def get_top_k_configs(
        self,
        study: optuna.Study | str,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Get top K best configurations from a study.

        Args:
            study: Study object or study name
            k: Number of top configurations to return

        Returns:
            List of parameter dictionaries, sorted by performance
        """
        if isinstance(study, str):
            study = self.load_study(study)

        # Get completed trials
        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed_trials:
            return []

        # Sort by value (best first)
        is_minimization = study.direction == StudyDirection.MINIMIZE
        sorted_trials = sorted(
            completed_trials,
            key=lambda t: t.value if t.value is not None else float("inf"),
            reverse=not is_minimization,
        )

        # Return top K configs
        return [t.params for t in sorted_trials[:k]]

    def export_analysis(
        self,
        analysis: StudyAnalysis,
        output_path: Path | str,
    ) -> None:
        """Export study analysis to JSON file.

        Args:
            analysis: Study analysis to export
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict
        data = {
            "study_name": analysis.study_name,
            "direction": analysis.direction,
            "n_trials": analysis.n_trials,
            "n_completed_trials": analysis.n_completed_trials,
            "n_pruned_trials": analysis.n_pruned_trials,
            "n_failed_trials": analysis.n_failed_trials,
            "best_value": analysis.best_value,
            "best_params": analysis.best_params,
            "best_trial_number": analysis.best_trial_number,
            "param_importance": analysis.param_importance,
            "convergence_iteration": analysis.convergence_iteration,
            "datetime_start": analysis.datetime_start,
            "datetime_complete": analysis.datetime_complete,
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "value": t.value,
                    "params": t.params,
                    "state": t.state,
                    "duration": t.duration,
                    "datetime_start": t.datetime_start,
                    "datetime_complete": t.datetime_complete,
                }
                for t in analysis.trials
            ],
        }

        with output_path.open("w") as f:
            json.dump(data, f, indent=2, default=str)

        LOGGER.info("Exported study analysis to: %s", output_path)

    def compute_study_similarity(
        self,
        study1: optuna.Study | str,
        study2: optuna.Study | str,
    ) -> float:
        """Compute similarity between two studies based on parameter spaces.

        Args:
            study1: First study
            study2: Second study

        Returns:
            Similarity score in [0, 1] (1 = identical parameter spaces)
        """
        if isinstance(study1, str):
            study1 = self.load_study(study1)
        if isinstance(study2, str):
            study2 = self.load_study(study2)

        # Get parameter names
        params1 = set()
        params2 = set()

        for trial in study1.trials:
            params1.update(trial.params.keys())
        for trial in study2.trials:
            params2.update(trial.params.keys())

        # Compute Jaccard similarity
        if not params1 and not params2:
            return 1.0
        if not params1 or not params2:
            return 0.0

        intersection = len(params1 & params2)
        union = len(params1 | params2)
        return intersection / union
