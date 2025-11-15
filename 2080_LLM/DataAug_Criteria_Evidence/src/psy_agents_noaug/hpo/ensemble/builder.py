#!/usr/bin/env python
"""Ensemble building strategies (Phase 11).

This module provides utilities for building ensembles from selected models,
including averaging, voting, and stacking strategies.

Key Features:
- Simple averaging
- Weighted averaging (by performance/diversity)
- Majority voting for classification
- Stacking with meta-learner
- Ensemble configuration management
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from psy_agents_noaug.hpo.ensemble.selection import SelectedModel

LOGGER = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for an ensemble."""

    name: str
    strategy: Literal["average", "weighted_average", "voting", "stacking"]
    models: list[dict[str, Any]]  # List of model configs
    weights: list[float] | None = None
    meta_config: dict[str, Any] | None = None


class EnsembleBuilder:
    """Build ensembles from selected models."""

    def __init__(self, ensemble_name: str = "hpo_ensemble"):
        """Initialize ensemble builder.

        Args:
            ensemble_name: Name for the ensemble
        """
        self.ensemble_name = ensemble_name

    def build_average_ensemble(
        self,
        selected_models: list[SelectedModel],
    ) -> EnsembleConfig:
        """Build simple averaging ensemble.

        Args:
            selected_models: Models to include in ensemble

        Returns:
            Ensemble configuration
        """
        LOGGER.info(
            "Building simple averaging ensemble with %d models",
            len(selected_models),
        )

        # Equal weights for all models
        weights = [1.0 / len(selected_models)] * len(selected_models)

        return EnsembleConfig(
            name=f"{self.ensemble_name}_avg_{len(selected_models)}",
            strategy="average",
            models=[
                {
                    "trial_number": m.trial_number,
                    "params": m.params,
                    "value": m.value,
                }
                for m in selected_models
            ],
            weights=weights,
        )

    def build_weighted_ensemble(
        self,
        selected_models: list[SelectedModel],
        weighting: Literal["performance", "inverse_rank", "diversity"] = "performance",
        temperature: float = 1.0,
    ) -> EnsembleConfig:
        """Build weighted averaging ensemble.

        Args:
            selected_models: Models to include
            weighting: Weighting strategy:
                - performance: Weight by model performance
                - inverse_rank: Weight by inverse rank
                - diversity: Weight by diversity score
            temperature: Temperature for softmax weighting (higher = more uniform)

        Returns:
            Ensemble configuration with weights
        """
        LOGGER.info(
            "Building weighted ensemble with %d models (weighting=%s, T=%.2f)",
            len(selected_models),
            weighting,
            temperature,
        )

        # Compute raw weights based on strategy
        if weighting == "performance":
            # Weight by normalized performance
            values = np.array([m.value for m in selected_models])
            # Normalize to [0, 1] and invert if minimization
            min_val, max_val = values.min(), values.max()
            if max_val > min_val:
                normalized = (values - min_val) / (max_val - min_val)
                raw_weights = 1.0 - normalized  # Assuming minimization
            else:
                raw_weights = np.ones(len(selected_models))

        elif weighting == "inverse_rank":
            # Weight by inverse rank (1, 1/2, 1/3, ...)
            raw_weights = np.array([1.0 / (i + 1) for i in range(len(selected_models))])

        elif weighting == "diversity":
            # Weight by diversity score
            raw_weights = np.array(
                [m.diversity_score if m.diversity_score > 0 else 1.0 for m in selected_models]
            )

        else:
            raise ValueError(f"Unknown weighting strategy: {weighting}")

        # Apply softmax with temperature
        weights = self._softmax(raw_weights / temperature)

        LOGGER.info(
            "Computed weights (range: %.4f - %.4f, entropy: %.4f)",
            weights.min(),
            weights.max(),
            self._entropy(weights),
        )

        return EnsembleConfig(
            name=f"{self.ensemble_name}_weighted_{weighting}_{len(selected_models)}",
            strategy="weighted_average",
            models=[
                {
                    "trial_number": m.trial_number,
                    "params": m.params,
                    "value": m.value,
                }
                for m in selected_models
            ],
            weights=weights.tolist(),
        )

    def build_voting_ensemble(
        self,
        selected_models: list[SelectedModel],
        voting: Literal["hard", "soft"] = "hard",
    ) -> EnsembleConfig:
        """Build voting ensemble for classification.

        Args:
            selected_models: Models to include
            voting: Voting strategy:
                - hard: Majority vote on predicted class
                - soft: Average predicted probabilities

        Returns:
            Ensemble configuration
        """
        LOGGER.info(
            "Building %s voting ensemble with %d models",
            voting,
            len(selected_models),
        )

        return EnsembleConfig(
            name=f"{self.ensemble_name}_voting_{voting}_{len(selected_models)}",
            strategy="voting",
            models=[
                {
                    "trial_number": m.trial_number,
                    "params": m.params,
                    "value": m.value,
                }
                for m in selected_models
            ],
            meta_config={"voting_type": voting},
        )

    def build_stacking_ensemble(
        self,
        selected_models: list[SelectedModel],
        meta_learner_config: dict[str, Any] | None = None,
    ) -> EnsembleConfig:
        """Build stacking ensemble with meta-learner.

        Args:
            selected_models: Base models
            meta_learner_config: Configuration for meta-learner

        Returns:
            Ensemble configuration with meta-learner
        """
        LOGGER.info(
            "Building stacking ensemble with %d base models",
            len(selected_models),
        )

        if meta_learner_config is None:
            # Default: simple logistic regression
            meta_learner_config = {
                "type": "logistic_regression",
                "params": {"C": 1.0, "max_iter": 1000},
            }

        return EnsembleConfig(
            name=f"{self.ensemble_name}_stacking_{len(selected_models)}",
            strategy="stacking",
            models=[
                {
                    "trial_number": m.trial_number,
                    "params": m.params,
                    "value": m.value,
                }
                for m in selected_models
            ],
            meta_config=meta_learner_config,
        )

    def export_ensemble(
        self,
        ensemble_config: EnsembleConfig,
        output_path: Path | str,
    ) -> None:
        """Export ensemble configuration to JSON.

        Args:
            ensemble_config: Ensemble configuration
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict
        config_dict = asdict(ensemble_config)

        with output_path.open("w") as f:
            json.dump(config_dict, f, indent=2)

        LOGGER.info("Exported ensemble config to: %s", output_path)

    def load_ensemble(
        self,
        config_path: Path | str,
    ) -> EnsembleConfig:
        """Load ensemble configuration from JSON.

        Args:
            config_path: Path to config file

        Returns:
            Ensemble configuration
        """
        config_path = Path(config_path)

        with config_path.open() as f:
            config_dict = json.load(f)

        ensemble_config = EnsembleConfig(**config_dict)

        LOGGER.info(
            "Loaded ensemble config: %s (%d models)",
            ensemble_config.name,
            len(ensemble_config.models),
        )

        return ensemble_config

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax of array.

        Args:
            x: Input array

        Returns:
            Softmax probabilities
        """
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / exp_x.sum()

    @staticmethod
    def _entropy(p: np.ndarray) -> float:
        """Compute entropy of probability distribution.

        Args:
            p: Probability distribution

        Returns:
            Entropy (in nats)
        """
        # Avoid log(0)
        p = p[p > 0]
        return -np.sum(p * np.log(p))


class EnsembleEvaluator:
    """Evaluate ensemble performance."""

    def __init__(self, ensemble_config: EnsembleConfig):
        """Initialize evaluator.

        Args:
            ensemble_config: Ensemble configuration
        """
        self.config = ensemble_config

    def estimate_ensemble_performance(
        self,
        individual_performances: list[float],
    ) -> dict[str, float]:
        """Estimate ensemble performance from individual model performances.

        Uses theoretical bounds and empirical rules.

        Args:
            individual_performances: Performance of each model

        Returns:
            Dict with estimated metrics
        """
        LOGGER.info("Estimating ensemble performance from %d models", len(individual_performances))

        individual_performances = np.array(individual_performances)

        # Basic statistics
        mean_perf = individual_performances.mean()
        std_perf = individual_performances.std()
        min_perf = individual_performances.min()
        max_perf = individual_performances.max()

        # Optimistic estimate (ensemble usually improves over average)
        # Based on empirical rule: ensemble ~ best + (mean - best) * 0.3
        optimistic_estimate = max_perf + (mean_perf - max_perf) * 0.3

        # Conservative estimate (worse than best, better than mean)
        conservative_estimate = max_perf + (mean_perf - max_perf) * 0.7

        # Expected improvement from diversity
        # More diverse ensemble → better improvement
        diversity_bonus = std_perf * 0.1

        return {
            "individual_mean": float(mean_perf),
            "individual_std": float(std_perf),
            "individual_min": float(min_perf),
            "individual_max": float(max_perf),
            "optimistic_estimate": float(optimistic_estimate + diversity_bonus),
            "conservative_estimate": float(conservative_estimate),
            "expected_estimate": float((optimistic_estimate + conservative_estimate) / 2),
        }

    def analyze_ensemble_diversity(
        self,
        predictions: np.ndarray,
    ) -> dict[str, float]:
        """Analyze diversity of ensemble predictions.

        Args:
            predictions: Array of shape (n_models, n_samples)

        Returns:
            Diversity metrics
        """
        n_models, n_samples = predictions.shape

        # Pairwise disagreement rate
        disagreements = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagreement = (predictions[i] != predictions[j]).mean()
                disagreements.append(disagreement)

        avg_disagreement = np.mean(disagreements) if disagreements else 0.0

        # Prediction entropy (for each sample)
        sample_entropies = []
        for sample_idx in range(n_samples):
            sample_preds = predictions[:, sample_idx]
            unique, counts = np.unique(sample_preds, return_counts=True)
            probs = counts / n_models
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            sample_entropies.append(entropy)

        avg_entropy = np.mean(sample_entropies)

        return {
            "avg_pairwise_disagreement": float(avg_disagreement),
            "avg_prediction_entropy": float(avg_entropy),
            "n_models": n_models,
        }


def recommend_ensemble_strategy(
    n_models: int,
    task_type: Literal["classification", "regression"] = "classification",
    diversity_level: Literal["low", "medium", "high"] = "medium",
) -> tuple[str, dict[str, Any]]:
    """Recommend an ensemble strategy based on scenario.

    Args:
        n_models: Number of models in ensemble
        task_type: Type of task
        diversity_level: Diversity of selected models

    Returns:
        (strategy_name, strategy_kwargs)

    Examples:
        >>> strategy, kwargs = recommend_ensemble_strategy(5, "classification", "high")
        >>> # Returns: ("voting", {"voting": "soft"})
    """
    if task_type == "classification":
        if diversity_level == "high":
            # High diversity: hard voting works well
            return ("voting", {"voting": "hard"})
        else:
            # Low/medium diversity: soft voting or weighted
            if n_models >= 5:
                return ("voting", {"voting": "soft"})
            else:
                return ("weighted_average", {"weighting": "performance"})

    else:  # regression
        if diversity_level == "high":
            # High diversity: simple average
            return ("average", {})
        else:
            # Low diversity: weighted by performance
            return ("weighted_average", {"weighting": "performance"})
