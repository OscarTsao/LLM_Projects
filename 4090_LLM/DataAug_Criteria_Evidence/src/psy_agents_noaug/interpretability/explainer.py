#!/usr/bin/env python
"""Unified model explainer (Phase 22).

This module provides a unified interface for explaining model predictions
using multiple interpretability methods.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import torch
from torch import nn

from psy_agents_noaug.interpretability.attention import AttentionVisualizer
from psy_agents_noaug.interpretability.feature_importance import (
    FeatureImportanceCalculator,
    ImportanceMethod,
)
from psy_agents_noaug.interpretability.shap_explainer import SHAPExplainer

LOGGER = logging.getLogger(__name__)


@dataclass
class ExplainerConfig:
    """Configuration for model explainer."""

    methods: list[str] = field(
        default_factory=lambda: ["gradient", "shap", "attention"]
    )
    shap_n_samples: int = 100
    importance_method: ImportanceMethod = ImportanceMethod.GRADIENT
    extract_attention: bool = True
    top_k_features: int = 10


@dataclass
class Explanation:
    """Model prediction explanation."""

    prediction: float | np.ndarray
    confidence: float
    feature_importance: dict[int, float]  # feature_idx -> importance
    attention_weights: np.ndarray | None = None
    shap_values: np.ndarray | None = None
    top_tokens: list[tuple[str, float]] | None = None
    counterfactuals: list[dict[str, Any]] | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_top_features(self, k: int = 10) -> list[tuple[int, float]]:
        """Get top-k most important features.

        Args:
            k: Number of top features

        Returns:
            List of (feature_index, importance) tuples
        """
        sorted_features = sorted(
            self.feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
        )
        return sorted_features[:k]

    def get_summary(self) -> dict[str, Any]:
        """Get explanation summary.

        Returns:
            Summary dictionary
        """
        return {
            "prediction": (
                float(self.prediction)
                if isinstance(self.prediction, (int, float, np.number))
                else self.prediction.tolist()
            ),
            "confidence": self.confidence,
            "top_features": self.get_top_features(5),
            "has_attention": self.attention_weights is not None,
            "has_shap": self.shap_values is not None,
            "timestamp": self.timestamp.isoformat(),
        }


class ModelExplainer:
    """Unified model explainer."""

    def __init__(
        self,
        model: nn.Module,
        config: ExplainerConfig | None = None,
    ):
        """Initialize model explainer.

        Args:
            model: PyTorch model to explain
            config: Explainer configuration
        """
        self.model = model
        self.config = config or ExplainerConfig()

        # Initialize sub-explainers
        self.shap_explainer: SHAPExplainer | None = None
        self.attention_visualizer: AttentionVisualizer | None = None
        self.importance_calculator = FeatureImportanceCalculator(model)

        if "shap" in self.config.methods:
            from psy_agents_noaug.interpretability.shap_explainer import SHAPConfig

            shap_config = SHAPConfig(n_samples=self.config.shap_n_samples)
            self.shap_explainer = SHAPExplainer(model, shap_config)

        if "attention" in self.config.methods or self.config.extract_attention:
            self.attention_visualizer = AttentionVisualizer(model)

        LOGGER.info(f"Initialized ModelExplainer with methods={self.config.methods}")

    def fit_background(self, background_data: torch.Tensor) -> None:
        """Fit background data for SHAP.

        Args:
            background_data: Background dataset
        """
        if self.shap_explainer:
            self.shap_explainer.fit_background(background_data)

    def explain(
        self,
        inputs: torch.Tensor,
        tokens: list[str] | None = None,
        target_class: int | None = None,
    ) -> Explanation:
        """Explain a prediction.

        Args:
            inputs: Input tensor
            tokens: Optional token strings
            target_class: Target class for multi-class models

        Returns:
            Explanation
        """
        self.model.eval()

        # Get prediction
        with torch.no_grad():
            outputs = self.model(inputs)

            # Check if multi-class (last dim > 1) or binary (last dim = 1)
            is_multiclass = len(outputs.shape) > 1 and outputs.shape[-1] > 1

            if is_multiclass:
                # Multi-class
                probs = torch.softmax(outputs, dim=-1)
                if target_class is not None:
                    prediction = outputs[:, target_class].mean().item()
                    confidence = probs[:, target_class].mean().item()
                else:
                    prediction = outputs.mean(dim=0).cpu().numpy()
                    confidence = probs.max(dim=-1).values.mean().item()
            else:
                # Binary or single output
                prediction = torch.sigmoid(outputs).mean().item()
                confidence = abs(prediction - 0.5) * 2  # Distance from 0.5

        # Calculate feature importance
        importance = self.importance_calculator.gradient_importance(
            inputs, target_class
        )
        feature_importance = dict(
            zip(
                importance.feature_indices.tolist(),
                importance.importance_scores.tolist(),
            )
        )

        # Get SHAP values
        shap_values = None
        if self.shap_explainer:
            try:
                shap_result = self.shap_explainer.explain(inputs, target_class)
                shap_values = shap_result.values
            except Exception:
                LOGGER.exception("Failed to compute SHAP values")

        # Extract attention
        attention_weights = None
        if self.attention_visualizer:
            try:
                attention_list = self.attention_visualizer.extract_attention(
                    inputs, tokens
                )
                if attention_list:
                    # Use aggregated attention from all layers
                    attention_weights = self.attention_visualizer.aggregate_attention(
                        attention_list
                    )
            except Exception:
                LOGGER.exception("Failed to extract attention")

        # Create top tokens list
        top_tokens = None
        if tokens and feature_importance:
            top_features = sorted(
                feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
            )[: self.config.top_k_features]

            top_tokens = [
                (tokens[idx] if idx < len(tokens) else f"[{idx}]", score)
                for idx, score in top_features
            ]

        return Explanation(
            prediction=prediction,
            confidence=confidence,
            feature_importance=feature_importance,
            attention_weights=attention_weights,
            shap_values=shap_values,
            top_tokens=top_tokens,
        )

    def explain_batch(
        self,
        inputs: torch.Tensor,
        tokens_list: list[list[str]] | None = None,
        target_class: int | None = None,
    ) -> list[Explanation]:
        """Explain a batch of predictions.

        Args:
            inputs: Batch of input tensors
            tokens_list: Optional list of token strings per input
            target_class: Target class for multi-class models

        Returns:
            List of explanations
        """
        explanations = []

        for i in range(len(inputs)):
            input_i = inputs[i : i + 1]
            tokens = tokens_list[i] if tokens_list else None

            explanation = self.explain(input_i, tokens, target_class)
            explanations.append(explanation)

        return explanations

    def generate_counterfactuals(
        self,
        inputs: torch.Tensor,
        target_prediction: float,
        max_changes: int = 5,
        n_samples: int = 10,
    ) -> list[dict[str, Any]]:
        """Generate counterfactual explanations.

        Args:
            inputs: Input tensor
            target_prediction: Desired prediction value
            max_changes: Maximum number of features to change
            n_samples: Number of counterfactuals to generate

        Returns:
            List of counterfactual explanations
        """
        self.model.eval()

        # Get current prediction
        with torch.no_grad():
            current_output = self.model(inputs)
            current_pred = torch.sigmoid(current_output).item()

        # Get feature importance
        importance = self.importance_calculator.gradient_importance(inputs)
        top_features = importance.get_top_k(max_changes)

        counterfactuals = []

        for _ in range(n_samples):
            # Create modified input
            modified = inputs.clone()

            # Perturb top features
            changes = []
            for feat_idx, _ in top_features[: np.random.randint(1, max_changes + 1)]:
                # Random perturbation
                perturbation = np.random.randn() * 0.1
                modified.view(-1)[feat_idx] += perturbation
                changes.append(feat_idx)

            # Get new prediction
            with torch.no_grad():
                new_output = self.model(modified)
                new_pred = torch.sigmoid(new_output).item()

            # Check if closer to target
            if abs(new_pred - target_prediction) < abs(
                current_pred - target_prediction
            ):
                counterfactuals.append(
                    {
                        "original_prediction": current_pred,
                        "new_prediction": new_pred,
                        "changed_features": changes,
                        "num_changes": len(changes),
                    }
                )

        return sorted(
            counterfactuals, key=lambda x: abs(x["new_prediction"] - target_prediction)
        )


def explain_prediction(
    model: nn.Module,
    inputs: torch.Tensor,
    tokens: list[str] | None = None,
    background_data: torch.Tensor | None = None,
) -> Explanation:
    """Explain a prediction (convenience function).

    Args:
        model: PyTorch model
        inputs: Input tensor
        tokens: Optional token strings
        background_data: Optional background data for SHAP

    Returns:
        Explanation
    """
    explainer = ModelExplainer(model)

    if background_data is not None:
        explainer.fit_background(background_data)

    return explainer.explain(inputs, tokens)
