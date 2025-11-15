#!/usr/bin/env python
"""Feature importance tracking (Phase 22).

This module provides tools for calculating and tracking feature importance
across different methods.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


class ImportanceMethod(str, Enum):
    """Feature importance calculation methods."""

    GRADIENT = "gradient"  # Gradient-based importance
    INTEGRATED_GRADIENTS = "integrated_gradients"  # Integrated gradients
    ABLATION = "ablation"  # Feature ablation
    PERMUTATION = "permutation"  # Permutation importance


@dataclass
class FeatureImportance:
    """Feature importance result."""

    feature_indices: np.ndarray  # Shape: (num_features,)
    importance_scores: np.ndarray  # Shape: (num_features,)
    method: ImportanceMethod
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_top_k(self, k: int = 10) -> list[tuple[int, float]]:
        """Get top-k most important features.

        Args:
            k: Number of top features

        Returns:
            List of (feature_index, score) tuples
        """
        top_indices = np.argsort(self.importance_scores)[::-1][:k]
        return [
            (int(self.feature_indices[idx]), float(self.importance_scores[idx]))
            for idx in top_indices
        ]


class FeatureImportanceCalculator:
    """Calculator for feature importance."""

    def __init__(self, model: nn.Module):
        """Initialize feature importance calculator.

        Args:
            model: PyTorch model
        """
        self.model = model
        LOGGER.info("Initialized FeatureImportanceCalculator")

    def gradient_importance(
        self,
        inputs: torch.Tensor,
        target_class: int | None = None,
    ) -> FeatureImportance:
        """Calculate gradient-based feature importance.

        Args:
            inputs: Input tensor
            target_class: Target class for multi-class models

        Returns:
            Feature importance
        """
        self.model.eval()

        with torch.enable_grad():
            inputs = inputs.clone().detach().requires_grad_(True)

            # Forward pass
            outputs = self.model(inputs)

            # Get logits for target class
            if target_class is not None and len(outputs.shape) > 1:
                logits = outputs[:, target_class]
            else:
                logits = outputs.squeeze()

            # Compute gradients
            logits.backward(torch.ones_like(logits))

            # Importance = |gradient * input|
            importance = (inputs.grad.abs() * inputs.abs()).detach().cpu().numpy()

        # Average across batch if multiple samples
        if len(importance.shape) > 1 and importance.shape[0] > 1:
            importance = importance.mean(axis=0)
        else:
            importance = importance.squeeze()

        # Create feature indices
        feature_indices = np.arange(len(importance))

        return FeatureImportance(
            feature_indices=feature_indices,
            importance_scores=importance,
            method=ImportanceMethod.GRADIENT,
        )

    def integrated_gradients(
        self,
        inputs: torch.Tensor,
        baseline: torch.Tensor | None = None,
        n_steps: int = 50,
        target_class: int | None = None,
    ) -> FeatureImportance:
        """Calculate integrated gradients.

        Args:
            inputs: Input tensor
            baseline: Baseline input (defaults to zeros)
            n_steps: Number of integration steps
            target_class: Target class for multi-class models

        Returns:
            Feature importance
        """
        if baseline is None:
            baseline = torch.zeros_like(inputs)

        self.model.eval()

        # Create interpolation path
        alphas = torch.linspace(0, 1, n_steps, device=inputs.device)

        # Accumulate gradients
        accumulated_grads = torch.zeros_like(inputs)

        for alpha in alphas:
            # Interpolate
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated = interpolated.clone().detach().requires_grad_(True)

            # Forward pass
            outputs = self.model(interpolated)

            # Get logits
            if target_class is not None and len(outputs.shape) > 1:
                logits = outputs[:, target_class]
            else:
                logits = outputs.squeeze()

            # Backward
            logits.backward(torch.ones_like(logits))

            # Accumulate
            accumulated_grads += interpolated.grad

        # Average and scale
        avg_grads = accumulated_grads / n_steps
        integrated_grads = (inputs - baseline) * avg_grads

        importance = integrated_grads.abs().detach().cpu().numpy()

        # Average across batch
        if len(importance.shape) > 1 and importance.shape[0] > 1:
            importance = importance.mean(axis=0)
        else:
            importance = importance.squeeze()

        feature_indices = np.arange(len(importance))

        return FeatureImportance(
            feature_indices=feature_indices,
            importance_scores=importance,
            method=ImportanceMethod.INTEGRATED_GRADIENTS,
        )

    def ablation_importance(
        self,
        inputs: torch.Tensor,
        target_class: int | None = None,
        n_samples: int = 10,
    ) -> FeatureImportance:
        """Calculate ablation-based importance.

        Args:
            inputs: Input tensor
            target_class: Target class
            n_samples: Number of ablation samples per feature

        Returns:
            Feature importance
        """
        self.model.eval()

        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(inputs)
            if target_class is not None and len(baseline_output.shape) > 1:
                baseline_score = baseline_output[:, target_class].mean().item()
            else:
                baseline_score = baseline_output.mean().item()

        # Flatten inputs for feature-wise ablation
        inputs_flat = inputs.view(inputs.size(0), -1)
        num_features = inputs_flat.size(1)

        importance_scores = np.zeros(num_features)

        # Ablate each feature
        for feat_idx in range(num_features):
            scores = []

            for _ in range(n_samples):
                # Create ablated input
                ablated = inputs_flat.clone()
                ablated[:, feat_idx] = 0  # Zero out feature

                # Reshape back
                ablated = ablated.view_as(inputs)

                # Predict
                with torch.no_grad():
                    output = self.model(ablated)
                    if target_class is not None and len(output.shape) > 1:
                        score = output[:, target_class].mean().item()
                    else:
                        score = output.mean().item()

                scores.append(baseline_score - score)

            # Average importance
            importance_scores[feat_idx] = np.mean(scores)

        feature_indices = np.arange(num_features)

        return FeatureImportance(
            feature_indices=feature_indices,
            importance_scores=importance_scores,
            method=ImportanceMethod.ABLATION,
        )


class FeatureImportanceTracker:
    """Tracker for feature importance across experiments."""

    def __init__(self):
        """Initialize feature importance tracker."""
        self.importance_history: list[FeatureImportance] = []
        LOGGER.info("Initialized FeatureImportanceTracker")

    def track(self, importance: FeatureImportance) -> None:
        """Track feature importance.

        Args:
            importance: Feature importance to track
        """
        self.importance_history.append(importance)

    def get_aggregate_importance(
        self,
        method: ImportanceMethod | None = None,
        aggregation: str = "mean",
    ) -> FeatureImportance | None:
        """Get aggregated importance across history.

        Args:
            method: Filter by method (optional)
            aggregation: Aggregation method (mean, max)

        Returns:
            Aggregated feature importance
        """
        # Filter by method
        if method:
            history = [fi for fi in self.importance_history if fi.method == method]
        else:
            history = self.importance_history

        if not history:
            return None

        # Ensure all have same features
        num_features = len(history[0].feature_indices)
        scores_list = [fi.importance_scores for fi in history]

        # Aggregate
        if aggregation == "mean":
            agg_scores = np.mean(scores_list, axis=0)
        elif aggregation == "max":
            agg_scores = np.max(scores_list, axis=0)
        else:
            msg = f"Unknown aggregation: {aggregation}"
            raise ValueError(msg)

        return FeatureImportance(
            feature_indices=np.arange(num_features),
            importance_scores=agg_scores,
            method=history[0].method,
            metadata={"aggregation": aggregation, "n_samples": len(history)},
        )


def calculate_feature_importance(
    model: nn.Module,
    inputs: torch.Tensor,
    method: ImportanceMethod = ImportanceMethod.GRADIENT,
    target_class: int | None = None,
) -> FeatureImportance:
    """Calculate feature importance (convenience function).

    Args:
        model: PyTorch model
        inputs: Input tensor
        method: Importance calculation method
        target_class: Target class for multi-class models

    Returns:
        Feature importance
    """
    calculator = FeatureImportanceCalculator(model)

    if method == ImportanceMethod.GRADIENT:
        return calculator.gradient_importance(inputs, target_class)
    if method == ImportanceMethod.INTEGRATED_GRADIENTS:
        return calculator.integrated_gradients(inputs, target_class=target_class)
    if method == ImportanceMethod.ABLATION:
        return calculator.ablation_importance(inputs, target_class)
    msg = f"Unsupported method: {method}"
    raise ValueError(msg)
