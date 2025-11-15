#!/usr/bin/env python
"""Feature importance analysis (Phase 27).

This module provides tools for analyzing feature importance in models,
including permutation importance and gradient-based methods.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Feature importance scores."""

    feature_names: list[str]
    importances: np.ndarray
    method: str
    std: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_top_features(self, k: int = 10) -> list[tuple[str, float]]:
        """Get top k most important features.

        Args:
            k: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        indices = np.argsort(self.importances)[::-1][:k]
        return [(self.feature_names[i], float(self.importances[i])) for i in indices]

    def get_summary(self) -> dict[str, Any]:
        """Get importance summary.

        Returns:
            Summary dictionary
        """
        return {
            "method": self.method,
            "num_features": len(self.feature_names),
            "top_features": self.get_top_features(10),
            "mean_importance": float(np.mean(self.importances)),
            "std_importance": float(np.std(self.importances)),
        }


class FeatureImportanceAnalyzer:
    """Analyzer for feature importance."""

    def __init__(self, model: Any, feature_names: list[str]):
        """Initialize feature importance analyzer.

        Args:
            model: Model to analyze
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names

        LOGGER.info(f"Initialized FeatureImportanceAnalyzer with {len(feature_names)} features")

    def permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: Callable[[np.ndarray, np.ndarray], float],
        n_repeats: int = 10,
    ) -> FeatureImportance:
        """Calculate permutation importance.

        Args:
            X: Input features
            y: Target labels
            metric: Scoring function (higher is better)
            n_repeats: Number of permutation repeats

        Returns:
            Feature importance scores
        """
        LOGGER.info("Calculating permutation importance...")

        # Baseline score
        baseline_score = metric(y, self.model.predict(X))

        n_features = X.shape[1]
        importances = np.zeros((n_repeats, n_features))

        for repeat in range(n_repeats):
            for feat_idx in range(n_features):
                # Permute feature
                X_permuted = X.copy()
                X_permuted[:, feat_idx] = np.random.permutation(X_permuted[:, feat_idx])

                # Calculate score drop
                permuted_score = metric(y, self.model.predict(X_permuted))
                importances[repeat, feat_idx] = baseline_score - permuted_score

        # Average over repeats
        mean_importances = np.mean(importances, axis=0)
        std_importances = np.std(importances, axis=0)

        LOGGER.info("Permutation importance calculation complete")

        return FeatureImportance(
            feature_names=self.feature_names,
            importances=mean_importances,
            method="permutation",
            std=std_importances,
        )

    def gradient_importance(
        self,
        X: torch.Tensor,
        target_class: int | None = None,
    ) -> FeatureImportance:
        """Calculate gradient-based feature importance.

        Args:
            X: Input features (torch.Tensor)
            target_class: Target class for importance (None for regression)

        Returns:
            Feature importance scores
        """
        LOGGER.info("Calculating gradient-based importance...")

        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)

        X.requires_grad = True

        # Forward pass
        outputs = self.model(X)

        if target_class is not None:
            # Classification: use specific class
            outputs = outputs[:, target_class]

        # Calculate gradients
        outputs.sum().backward()

        # Importance = mean absolute gradient
        importances = torch.abs(X.grad).mean(dim=0).detach().numpy()

        LOGGER.info("Gradient-based importance calculation complete")

        return FeatureImportance(
            feature_names=self.feature_names,
            importances=importances,
            method="gradient",
        )

    def integrated_gradients(
        self,
        X: torch.Tensor,
        baseline: torch.Tensor | None = None,
        n_steps: int = 50,
        target_class: int | None = None,
    ) -> FeatureImportance:
        """Calculate integrated gradients.

        Args:
            X: Input features
            baseline: Baseline input (default: zeros)
            n_steps: Number of integration steps
            target_class: Target class for importance

        Returns:
            Feature importance scores
        """
        LOGGER.info("Calculating integrated gradients...")

        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)

        if baseline is None:
            baseline = torch.zeros_like(X)

        # Create path from baseline to input
        alphas = torch.linspace(0, 1, n_steps)

        gradients = []
        for alpha in alphas:
            # Interpolate
            X_interp = baseline + alpha * (X - baseline)
            X_interp.requires_grad = True

            # Forward pass
            outputs = self.model(X_interp)

            if target_class is not None:
                outputs = outputs[:, target_class]

            # Calculate gradient
            outputs.sum().backward()
            gradients.append(X_interp.grad.detach())

        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)

        # Multiply by (input - baseline)
        importances = (avg_gradients * (X - baseline)).mean(dim=0).abs().numpy()

        LOGGER.info("Integrated gradients calculation complete")

        return FeatureImportance(
            feature_names=self.feature_names,
            importances=importances,
            method="integrated_gradients",
        )


def calculate_feature_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    method: str = "permutation",
    **kwargs: Any,
) -> FeatureImportance:
    """Calculate feature importance (convenience function).

    Args:
        model: Model to analyze
        X: Input features
        y: Target labels
        feature_names: Feature names
        method: Method to use ("permutation", "gradient", "integrated_gradients")
        **kwargs: Additional arguments for method

    Returns:
        Feature importance scores
    """
    analyzer = FeatureImportanceAnalyzer(model, feature_names)

    if method == "permutation":
        from sklearn.metrics import accuracy_score

        metric = kwargs.get("metric", accuracy_score)
        n_repeats = kwargs.get("n_repeats", 10)
        return analyzer.permutation_importance(X, y, metric, n_repeats)

    elif method == "gradient":
        X_tensor = torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X
        target_class = kwargs.get("target_class")
        return analyzer.gradient_importance(X_tensor, target_class)

    elif method == "integrated_gradients":
        X_tensor = torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X
        baseline = kwargs.get("baseline")
        n_steps = kwargs.get("n_steps", 50)
        target_class = kwargs.get("target_class")
        return analyzer.integrated_gradients(X_tensor, baseline, n_steps, target_class)

    else:
        msg = f"Unknown method: {method}"
        raise ValueError(msg)
