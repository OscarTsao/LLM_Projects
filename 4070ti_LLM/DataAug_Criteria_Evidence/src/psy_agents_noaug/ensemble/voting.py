"""Weighted voting ensemble for classification tasks.

This module implements weighted voting strategies where predictions from
multiple models are combined using learned or optimized weights.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from sklearn.metrics import f1_score, log_loss

LOGGER = logging.getLogger(__name__)


class WeightedVotingEnsemble:
    """Weighted soft voting ensemble for multi-class classification.

    Combines probability predictions from multiple models using weights
    that can be uniform, performance-based, or optimized.
    """

    def __init__(
        self,
        weights: NDArray[np.floating] | None = None,
        weight_strategy: str = "uniform",
    ):
        """Initialize weighted voting ensemble.

        Args:
            weights: Model weights (if None, will be computed based on strategy)
            weight_strategy: How to compute weights ("uniform", "performance", "optimized")
        """
        self.weights = weights
        self.weight_strategy = weight_strategy
        self._is_fitted = weights is not None

    def fit(
        self,
        model_probs: list[NDArray[np.floating]],
        y_true: NDArray[np.int_],
        model_scores: list[float] | None = None,
    ) -> WeightedVotingEnsemble:
        """Fit ensemble weights based on validation data.

        Args:
            model_probs: List of probability arrays from each model (N_models × N_samples × N_classes)
            y_true: Ground truth labels (N_samples,)
            model_scores: Individual model F1 scores (for performance-based weighting)

        Returns:
            Self
        """
        n_models = len(model_probs)

        if self.weight_strategy == "uniform":
            self.weights = np.ones(n_models) / n_models
            LOGGER.info("Using uniform weights: %s", self.weights)

        elif self.weight_strategy == "performance":
            if model_scores is None:
                raise ValueError(
                    "model_scores required for performance-based weighting"
                )
            # Weight by normalized F1 scores
            scores_arr = np.array(model_scores)
            self.weights = scores_arr / scores_arr.sum()
            LOGGER.info("Using performance-based weights: %s", self.weights)

        elif self.weight_strategy == "optimized":
            # Optimize weights to minimize log loss on validation set
            self.weights = optimize_voting_weights(model_probs, y_true)
            LOGGER.info("Using optimized weights: %s", self.weights)

        else:
            raise ValueError(f"Unknown weight strategy: {self.weight_strategy}")

        self._is_fitted = True
        return self

    def predict_proba(
        self,
        model_probs: list[NDArray[np.floating]],
    ) -> NDArray[np.floating]:
        """Predict class probabilities using weighted voting.

        Args:
            model_probs: List of probability arrays from each model

        Returns:
            Ensemble probability predictions (N_samples × N_classes)
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        # Weighted average of probabilities
        ensemble_probs = np.average(
            np.stack(model_probs, axis=0),
            axis=0,
            weights=self.weights,
        )

        return ensemble_probs

    def predict(
        self,
        model_probs: list[NDArray[np.floating]],
    ) -> NDArray[np.int_]:
        """Predict class labels using weighted voting.

        Args:
            model_probs: List of probability arrays from each model

        Returns:
            Ensemble class predictions (N_samples,)
        """
        probs = self.predict_proba(model_probs)
        return np.argmax(probs, axis=1)

    def evaluate(
        self,
        model_probs: list[NDArray[np.floating]],
        y_true: NDArray[np.int_],
    ) -> dict[str, float]:
        """Evaluate ensemble on test data.

        Args:
            model_probs: List of probability arrays from each model
            y_true: Ground truth labels

        Returns:
            Dictionary of metrics
        """
        probs = self.predict_proba(model_probs)
        preds = np.argmax(probs, axis=1)

        metrics = {
            "f1_macro": f1_score(y_true, preds, average="macro"),
            "f1_micro": f1_score(y_true, preds, average="micro"),
            "logloss": log_loss(y_true, probs),
        }

        LOGGER.info(
            "Ensemble evaluation: F1-macro=%.4f, Logloss=%.4f",
            metrics["f1_macro"],
            metrics["logloss"],
        )
        return metrics


def optimize_voting_weights(
    model_probs: list[NDArray[np.floating]],
    y_true: NDArray[np.int_],
    metric: str = "logloss",
) -> NDArray[np.floating]:
    """Optimize ensemble weights to minimize loss on validation set.

    Args:
        model_probs: List of probability arrays from each model
        y_true: Ground truth labels
        metric: Metric to optimize ("logloss" or "f1")

    Returns:
        Optimized weights (sum to 1)
    """
    n_models = len(model_probs)

    # Stack model probabilities
    probs_stack = np.stack(model_probs, axis=0)  # (n_models, n_samples, n_classes)

    # Objective function
    def objective(weights: NDArray[np.floating]) -> float:
        # Normalize weights to sum to 1
        weights_norm = weights / (weights.sum() + 1e-8)

        # Ensemble predictions
        ensemble_probs = np.average(probs_stack, axis=0, weights=weights_norm)

        if metric == "logloss":
            return log_loss(y_true, ensemble_probs)
        if metric == "f1":
            preds = np.argmax(ensemble_probs, axis=1)
            return -f1_score(
                y_true, preds, average="macro"
            )  # Negative for minimization
        raise ValueError(f"Unknown metric: {metric}")

    # Constraints: weights > 0, sum to 1
    constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
    ]
    bounds = [(0.0, 1.0) for _ in range(n_models)]

    # Initial guess: uniform weights
    x0 = np.ones(n_models) / n_models

    # Optimize
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 100},
    )

    if not result.success:
        LOGGER.warning("Weight optimization did not converge: %s", result.message)
        return x0  # Fall back to uniform

    # Normalize to ensure sum to 1 (numerical stability)
    weights_opt = result.x / result.x.sum()

    LOGGER.info(
        "Optimized weights (metric=%s): %s (objective=%.4f)",
        metric,
        weights_opt,
        result.fun,
    )
    return weights_opt


def evaluate_ensemble_combination(
    model_probs: list[NDArray[np.floating]],
    y_true: NDArray[np.int_],
    weights: NDArray[np.floating] | None = None,
) -> dict[str, float]:
    """Evaluate a specific ensemble combination.

    Args:
        model_probs: List of probability arrays
        y_true: Ground truth labels
        weights: Model weights (if None, use uniform)

    Returns:
        Dictionary of metrics
    """
    if weights is None:
        weights = np.ones(len(model_probs)) / len(model_probs)

    # Ensemble predictions
    ensemble_probs = np.average(np.stack(model_probs, axis=0), axis=0, weights=weights)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    metrics = {
        "f1_macro": f1_score(y_true, ensemble_preds, average="macro"),
        "f1_micro": f1_score(y_true, ensemble_preds, average="micro"),
        "logloss": log_loss(y_true, ensemble_probs),
    }

    return metrics
