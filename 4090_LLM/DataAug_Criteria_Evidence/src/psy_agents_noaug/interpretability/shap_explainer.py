#!/usr/bin/env python
"""SHAP-based model explanations (Phase 22).

This module provides SHAP (SHapley Additive exPlanations) integration for
understanding feature contributions to model predictions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


@dataclass
class SHAPConfig:
    """SHAP explainer configuration."""

    method: str = "gradient"  # gradient, deep, or kernel
    n_samples: int = 100  # Number of samples for background dataset
    batch_size: int = 32
    max_features: int = 512  # Max features to explain


@dataclass
class SHAPValues:
    """SHAP values for a prediction."""

    values: np.ndarray  # Shape: (num_features,)
    base_value: float
    feature_names: list[str] | None = None
    data: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SHAPExplainer:
    """SHAP-based model explainer.

    This is a simplified implementation that computes approximate SHAP values
    using gradient-based methods. For production, consider using the official
    shap library.
    """

    def __init__(self, model: nn.Module, config: SHAPConfig | None = None):
        """Initialize SHAP explainer.

        Args:
            model: PyTorch model to explain
            config: SHAP configuration
        """
        self.model = model
        self.config = config or SHAPConfig()
        self.background_data: torch.Tensor | None = None

        LOGGER.info(f"Initialized SHAPExplainer with method={self.config.method}")

    def fit_background(self, data: torch.Tensor) -> None:
        """Fit background dataset for SHAP.

        Args:
            data: Background dataset (used for baseline)
        """
        # Sample background data
        n_samples = min(self.config.n_samples, len(data))
        indices = torch.randperm(len(data))[:n_samples]
        self.background_data = data[indices]

        LOGGER.info(f"Fitted background data with {n_samples} samples")

    def explain(
        self,
        inputs: torch.Tensor,
        target_class: int | None = None,
    ) -> SHAPValues:
        """Compute SHAP values for inputs.

        Args:
            inputs: Input tensor to explain
            target_class: Target class for multi-class models

        Returns:
            SHAP values
        """
        if self.background_data is None:
            msg = "Must call fit_background() before explain()"
            raise ValueError(msg)

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

            # Compute gradients (approximation of SHAP values)
            logits.backward(torch.ones_like(logits))

            # Gradient * input as SHAP approximation
            shap_values = (inputs.grad * inputs).detach().cpu().numpy()

        # Average across batch if multiple samples
        if len(shap_values.shape) > 1:
            shap_values = shap_values.mean(axis=0)

        # Compute base value (mean prediction on background)
        with torch.no_grad():
            base_outputs = self.model(self.background_data)
            if target_class is not None and len(base_outputs.shape) > 1:
                base_value = base_outputs[:, target_class].mean().item()
            else:
                base_value = base_outputs.mean().item()

        return SHAPValues(
            values=shap_values,
            base_value=base_value,
            data=inputs.detach().cpu().numpy(),
        )

    def explain_batch(
        self,
        inputs: torch.Tensor,
        target_class: int | None = None,
    ) -> list[SHAPValues]:
        """Compute SHAP values for a batch of inputs.

        Args:
            inputs: Batch of input tensors
            target_class: Target class for multi-class models

        Returns:
            List of SHAP values for each input
        """
        results = []

        for i in range(0, len(inputs), self.config.batch_size):
            batch = inputs[i : i + self.config.batch_size]
            shap_vals = self.explain(batch, target_class)
            results.append(shap_vals)

        return results

    def get_feature_importance(
        self,
        shap_values: SHAPValues,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Get top-k most important features.

        Args:
            shap_values: SHAP values
            top_k: Number of top features to return

        Returns:
            List of (feature_index, importance_score) tuples
        """
        # Use absolute values for importance
        importances = np.abs(shap_values.values)

        # Get top-k indices
        top_indices = np.argsort(importances)[::-1][:top_k]

        return [(int(idx), float(importances[idx])) for idx in top_indices]


def create_shap_explainer(
    model: nn.Module,
    background_data: torch.Tensor,
    method: str = "gradient",
    n_samples: int = 100,
) -> SHAPExplainer:
    """Create and fit a SHAP explainer (convenience function).

    Args:
        model: PyTorch model
        background_data: Background dataset
        method: SHAP method to use
        n_samples: Number of background samples

    Returns:
        Fitted SHAP explainer
    """
    config = SHAPConfig(method=method, n_samples=n_samples)
    explainer = SHAPExplainer(model, config)
    explainer.fit_background(background_data)

    return explainer
