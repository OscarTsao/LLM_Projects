#!/usr/bin/env python
"""Warm-starting strategies for HPO (Phase 10).

This module provides utilities for warm-starting new HPO studies
using knowledge from previous completed studies.

Key Features:
- Enqueue promising configurations from historical studies
- Transfer configurations across similar tasks
- Smart initialization based on parameter importance
- Adaptive warm-start strategies
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import optuna

from psy_agents_noaug.hpo.meta_learning.history import TrialHistoryAnalyzer

LOGGER = logging.getLogger(__name__)


class WarmStartStrategy:
    """Strategies for warm-starting HPO studies."""

    def __init__(
        self,
        storage: str | None = None,
        analyzer: TrialHistoryAnalyzer | None = None,
    ):
        """Initialize warm-start strategy.

        Args:
            storage: Optuna storage URL
            analyzer: Pre-configured trial history analyzer
        """
        self.storage = storage
        self.analyzer = analyzer or TrialHistoryAnalyzer(storage=storage)

    def warm_start_from_study(
        self,
        target_study: optuna.Study,
        source_study: optuna.Study | str,
        n_configs: int = 5,
        strategy: Literal["best", "diverse", "importance_weighted"] = "best",
    ) -> int:
        """Warm-start a study using configurations from another study.

        Args:
            target_study: Study to warm-start
            source_study: Source study with historical trials
            n_configs: Number of configurations to enqueue
            strategy: How to select configurations:
                - "best": Top K best configurations
                - "diverse": Diverse configurations across parameter space
                - "importance_weighted": Sample based on parameter importance

        Returns:
            Number of configurations enqueued
        """
        # Load source study if name provided
        if isinstance(source_study, str):
            source_study = self.analyzer.load_study(source_study)

        LOGGER.info(
            "Warm-starting '%s' from '%s' with %d configs (strategy=%s)",
            target_study.study_name,
            source_study.study_name,
            n_configs,
            strategy,
        )

        # Select configurations based on strategy
        if strategy == "best":
            configs = self.analyzer.get_top_k_configs(source_study, k=n_configs)
        elif strategy == "diverse":
            configs = self._select_diverse_configs(source_study, k=n_configs)
        elif strategy == "importance_weighted":
            configs = self._select_importance_weighted_configs(
                source_study, k=n_configs
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Enqueue configurations
        enqueued_count = 0
        for config in configs:
            try:
                target_study.enqueue_trial(config)
                enqueued_count += 1
                LOGGER.debug("Enqueued config: %s", config)
            except Exception as e:
                LOGGER.warning("Failed to enqueue config %s: %s", config, e)

        LOGGER.info(
            "Successfully enqueued %d/%d configurations", enqueued_count, n_configs
        )
        return enqueued_count

    def warm_start_from_analysis(
        self,
        target_study: optuna.Study,
        analysis_file: str,
        n_configs: int = 5,
    ) -> int:
        """Warm-start from exported study analysis JSON.

        Args:
            target_study: Study to warm-start
            analysis_file: Path to exported analysis JSON
            n_configs: Number of best configurations to enqueue

        Returns:
            Number of configurations enqueued
        """
        import json
        from pathlib import Path

        analysis_path = Path(analysis_file)
        if not analysis_path.exists():
            raise FileNotFoundError(f"Analysis file not found: {analysis_file}")

        # Load analysis
        with analysis_path.open() as f:
            data = json.load(f)

        LOGGER.info(
            "Warm-starting '%s' from analysis: %s",
            target_study.study_name,
            analysis_path.name,
        )

        # Extract best trials (sorted by value)
        trials = data.get("trials", [])
        completed_trials = [t for t in trials if t["state"] == "COMPLETE"]

        # Sort by value
        direction = data.get("direction", "MINIMIZE")
        sorted_trials = sorted(
            completed_trials,
            key=lambda t: t["value"],
            reverse=(direction == "MAXIMIZE"),
        )

        # Enqueue top K configs
        enqueued_count = 0
        for trial in sorted_trials[:n_configs]:
            try:
                target_study.enqueue_trial(trial["params"])
                enqueued_count += 1
                LOGGER.debug("Enqueued config: %s", trial["params"])
            except Exception as e:
                LOGGER.warning("Failed to enqueue config: %s", e)

        LOGGER.info(
            "Successfully enqueued %d/%d configurations", enqueued_count, n_configs
        )
        return enqueued_count

    def _select_diverse_configs(
        self,
        study: optuna.Study,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Select diverse configurations using k-means-like clustering.

        Args:
            study: Source study
            k: Number of diverse configurations to select

        Returns:
            List of diverse parameter configurations
        """
        from optuna.trial import TrialState

        # Get completed trials
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if len(completed_trials) <= k:
            return [t.params for t in completed_trials]

        # For simplicity, use quantile-based selection across parameter ranges
        # This ensures we sample from different regions of the search space
        configs = []

        # Get parameter ranges
        all_params: dict[str, list[Any]] = {}
        for trial in completed_trials:
            for param_name, param_value in trial.params.items():
                if param_name not in all_params:
                    all_params[param_name] = []
                all_params[param_name].append(param_value)

        # Select trials at different quantiles
        quantiles = [i / (k - 1) for i in range(k)]
        selected_indices = set()

        for q in quantiles:
            # Find trial closest to this quantile across all parameters
            best_trial_idx = None
            best_score = float("inf")

            for idx, trial in enumerate(completed_trials):
                if idx in selected_indices:
                    continue

                # Compute distance to target quantile
                score = 0.0
                for param_name, param_value in trial.params.items():
                    if isinstance(param_value, (int, float)):
                        values = sorted(all_params[param_name])
                        target_value = values[int(q * (len(values) - 1))]
                        score += abs(param_value - target_value)

                if score < best_score:
                    best_score = score
                    best_trial_idx = idx

            if best_trial_idx is not None:
                selected_indices.add(best_trial_idx)
                configs.append(completed_trials[best_trial_idx].params)

        return configs[:k]

    def _select_importance_weighted_configs(
        self,
        study: optuna.Study,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Select configurations weighted by parameter importance.

        Prioritize trials that are good AND have important parameters
        in promising regions.

        Args:
            study: Source study
            k: Number of configurations to select

        Returns:
            List of importance-weighted configurations
        """
        # Analyze study to get importance
        analysis = self.analyzer.analyze_study(study, compute_importance=True)

        if not analysis.param_importance:
            # Fallback to best configs if importance not available
            LOGGER.warning("Parameter importance not available, using best configs")
            return self.analyzer.get_top_k_configs(study, k=k)

        # Get completed trials
        from optuna.trial import TrialState

        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

        # Score each trial based on performance + important parameter values
        # Strategy: Boost trials that have good important parameters
        trial_scores = []
        for trial in completed_trials:
            if trial.value is None:
                continue

            # Base score is trial value
            score = trial.value

            # Adjust based on important parameters
            # (For simplicity, just use raw score - more sophisticated methods
            # would analyze the distribution of important parameters)
            trial_scores.append((score, trial.params))

        # Sort by score
        is_minimization = study.direction == optuna.study.StudyDirection.MINIMIZE
        sorted_trials = sorted(
            trial_scores, key=lambda x: x[0], reverse=not is_minimization
        )

        return [params for _, params in sorted_trials[:k]]

    def create_warm_started_study(
        self,
        study_name: str,
        source_studies: list[str] | list[optuna.Study],
        direction: Literal["minimize", "maximize"] = "minimize",
        n_configs_per_source: int = 3,
        sampler: optuna.samplers.BaseSampler | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
    ) -> optuna.Study:
        """Create a new study warm-started from multiple source studies.

        Args:
            study_name: Name for new study
            source_studies: List of source study names or objects
            direction: Optimization direction
            n_configs_per_source: Configs to take from each source
            sampler: Sampler for new study
            pruner: Pruner for new study

        Returns:
            New study with enqueued configurations
        """
        # Create new study
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.storage,
            load_if_exists=False,
        )

        LOGGER.info(
            "Created new study '%s' with warm-start from %d sources",
            study_name,
            len(source_studies),
        )

        # Warm-start from each source
        total_enqueued = 0
        for source_study in source_studies:
            n_enqueued = self.warm_start_from_study(
                target_study=study,
                source_study=source_study,
                n_configs=n_configs_per_source,
                strategy="best",
            )
            total_enqueued += n_enqueued

        LOGGER.info("Total configurations enqueued: %d", total_enqueued)
        return study


class AdaptiveWarmStart:
    """Adaptive warm-starting that learns from multiple sources."""

    def __init__(
        self,
        storage: str | None = None,
        similarity_threshold: float = 0.5,
    ):
        """Initialize adaptive warm-starter.

        Args:
            storage: Optuna storage URL
            similarity_threshold: Minimum similarity to use a source study
        """
        self.storage = storage
        self.analyzer = TrialHistoryAnalyzer(storage=storage)
        self.similarity_threshold = similarity_threshold

    def find_similar_studies(
        self,
        target_study: optuna.Study,
        candidate_studies: list[str] | list[optuna.Study],
        max_sources: int = 3,
    ) -> list[str | optuna.Study]:
        """Find studies similar to target for warm-starting.

        Args:
            target_study: Target study to warm-start
            candidate_studies: Candidate source studies
            max_sources: Maximum number of similar studies to return

        Returns:
            List of similar studies, sorted by similarity
        """
        similarities = []
        for candidate in candidate_studies:
            similarity = self.analyzer.compute_study_similarity(target_study, candidate)
            if similarity >= self.similarity_threshold:
                similarities.append((similarity, candidate))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)

        similar_studies = [study for _, study in similarities[:max_sources]]
        LOGGER.info(
            "Found %d similar studies (threshold=%.2f): %s",
            len(similar_studies),
            self.similarity_threshold,
            [s if isinstance(s, str) else s.study_name for s in similar_studies],
        )

        return similar_studies

    def adaptive_warm_start(
        self,
        target_study: optuna.Study,
        candidate_studies: list[str] | list[optuna.Study],
        total_configs: int = 10,
        max_sources: int = 3,
    ) -> int:
        """Adaptively warm-start from most similar studies.

        Args:
            target_study: Study to warm-start
            candidate_studies: Candidate source studies
            total_configs: Total number of configurations to enqueue
            max_sources: Maximum number of sources to use

        Returns:
            Number of configurations enqueued
        """
        # Find similar studies
        similar_studies = self.find_similar_studies(
            target_study, candidate_studies, max_sources=max_sources
        )

        if not similar_studies:
            LOGGER.warning("No similar studies found for warm-starting")
            return 0

        # Distribute configs across sources
        configs_per_source = max(1, total_configs // len(similar_studies))

        # Warm-start from each similar study
        warm_starter = WarmStartStrategy(storage=self.storage, analyzer=self.analyzer)
        total_enqueued = 0

        for source_study in similar_studies:
            n_enqueued = warm_starter.warm_start_from_study(
                target_study=target_study,
                source_study=source_study,
                n_configs=configs_per_source,
                strategy="best",
            )
            total_enqueued += n_enqueued

        return total_enqueued
