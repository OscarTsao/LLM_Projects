#!/usr/bin/env python
"""Attention visualization for transformer models (Phase 22).

This module provides tools for extracting and visualizing attention weights
from transformer-based models.
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
class AttentionWeights:
    """Attention weights from a transformer layer."""

    weights: np.ndarray  # Shape: (num_heads, seq_len, seq_len)
    layer_idx: int
    tokens: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AttentionVisualizer:
    """Visualizer for transformer attention patterns."""

    def __init__(self, model: nn.Module):
        """Initialize attention visualizer.

        Args:
            model: Transformer model
        """
        self.model = model
        self.attention_hooks: list[Any] = []
        self.attention_weights: list[torch.Tensor] = []

        LOGGER.info("Initialized AttentionVisualizer")

    def register_hooks(self) -> None:
        """Register forward hooks to capture attention weights."""
        self.attention_weights = []

        def attention_hook(module, input, output):
            """Hook to capture attention weights."""
            # For transformer layers, attention weights are typically in output[1]
            if isinstance(output, tuple) and len(output) > 1:
                attn = output[1]
                if attn is not None:
                    self.attention_weights.append(attn.detach().cpu())

        # Register hooks on attention modules
        for name, module in self.model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                hook = module.register_forward_hook(attention_hook)
                self.attention_hooks.append(hook)

        LOGGER.info(f"Registered {len(self.attention_hooks)} attention hooks")

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.attention_hooks:
            hook.remove()
        self.attention_hooks = []
        self.attention_weights = []

    def extract_attention(
        self,
        inputs: torch.Tensor,
        tokens: list[str] | None = None,
    ) -> list[AttentionWeights]:
        """Extract attention weights for inputs.

        Args:
            inputs: Input tensor
            tokens: Optional token strings for visualization

        Returns:
            List of attention weights per layer
        """
        self.register_hooks()

        try:
            # Forward pass
            self.model.eval()
            with torch.no_grad():
                _ = self.model(inputs)

            # Extract attention weights
            results = []
            for layer_idx, attn in enumerate(self.attention_weights):
                # Average across batch if needed
                if len(attn.shape) == 4:  # (batch, heads, seq, seq)
                    attn = attn[0]  # Take first in batch

                results.append(
                    AttentionWeights(
                        weights=attn.numpy(),
                        layer_idx=layer_idx,
                        tokens=tokens,
                    )
                )

            return results

        finally:
            self.remove_hooks()

    def get_head_importance(
        self,
        attention_weights: AttentionWeights,
    ) -> np.ndarray:
        """Calculate importance score for each attention head.

        Args:
            attention_weights: Attention weights

        Returns:
            Importance scores per head
        """
        # Use entropy as importance measure
        # Lower entropy = more focused attention = more important
        weights = attention_weights.weights  # (num_heads, seq_len, seq_len)

        importances = []
        for head_idx in range(weights.shape[0]):
            head_weights = weights[head_idx]  # (seq_len, seq_len)

            # Calculate entropy
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            head_weights = head_weights + epsilon
            head_weights = head_weights / head_weights.sum(axis=1, keepdims=True)

            entropy = -(head_weights * np.log(head_weights)).sum(axis=1).mean()
            importances.append(entropy)

        return np.array(importances)

    def aggregate_attention(
        self,
        attention_weights: list[AttentionWeights],
        method: str = "mean",
    ) -> np.ndarray:
        """Aggregate attention across layers.

        Args:
            attention_weights: List of attention weights from all layers
            method: Aggregation method (mean, max, last)

        Returns:
            Aggregated attention matrix
        """
        if not attention_weights:
            return np.array([])

        # Get all weight matrices
        matrices = [aw.weights for aw in attention_weights]

        # Average across heads for each layer
        matrices = [w.mean(axis=0) for w in matrices]

        if method == "mean":
            return np.mean(matrices, axis=0)
        if method == "max":
            return np.max(matrices, axis=0)
        if method == "last":
            return matrices[-1]
        msg = f"Unknown aggregation method: {method}"
        raise ValueError(msg)


def extract_attention_weights(
    model: nn.Module,
    inputs: torch.Tensor,
    tokens: list[str] | None = None,
) -> list[AttentionWeights]:
    """Extract attention weights (convenience function).

    Args:
        model: Transformer model
        inputs: Input tensor
        tokens: Optional token strings

    Returns:
        List of attention weights per layer
    """
    visualizer = AttentionVisualizer(model)
    return visualizer.extract_attention(inputs, tokens)


def visualize_attention(
    attention_weights: AttentionWeights,
    tokens: list[str] | None = None,
) -> dict[str, Any]:
    """Create attention visualization data.

    Args:
        attention_weights: Attention weights to visualize
        tokens: Token strings for axes

    Returns:
        Visualization data dictionary
    """
    weights = attention_weights.weights

    # Average across heads
    avg_weights = weights.mean(axis=0)

    # Get top attention pairs
    top_k = 10
    flat_indices = np.argsort(avg_weights.flatten())[::-1][:top_k]
    top_pairs = []

    for idx in flat_indices:
        i = idx // avg_weights.shape[1]
        j = idx % avg_weights.shape[1]
        score = avg_weights[i, j]

        pair = {
            "from_idx": int(i),
            "to_idx": int(j),
            "score": float(score),
        }

        if tokens:
            pair["from_token"] = tokens[i] if i < len(tokens) else f"[{i}]"
            pair["to_token"] = tokens[j] if j < len(tokens) else f"[{j}]"

        top_pairs.append(pair)

    return {
        "layer_idx": attention_weights.layer_idx,
        "num_heads": weights.shape[0],
        "attention_matrix": avg_weights.tolist(),
        "top_pairs": top_pairs,
        "tokens": tokens,
    }
