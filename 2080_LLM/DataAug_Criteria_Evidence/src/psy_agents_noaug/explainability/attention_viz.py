#!/usr/bin/env python
"""Attention visualization for transformer models (Phase 27).

This module provides tools for visualizing attention patterns in
transformer-based models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class AttentionWeights:
    """Attention weights from a transformer layer."""

    weights: np.ndarray  # Shape: (n_heads, seq_len, seq_len)
    tokens: list[str]
    layer_idx: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_head_attention(self, head_idx: int) -> np.ndarray:
        """Get attention weights for specific head.

        Args:
            head_idx: Head index

        Returns:
            Attention matrix for head
        """
        return self.weights[head_idx]

    def get_average_attention(self) -> np.ndarray:
        """Get average attention across all heads.

        Returns:
            Average attention matrix
        """
        return np.mean(self.weights, axis=0)

    def get_token_attention(self, token_idx: int) -> dict[str, float]:
        """Get attention scores for a specific token.

        Args:
            token_idx: Token index

        Returns:
            Dictionary mapping tokens to attention scores
        """
        # Average across heads
        avg_attention = self.get_average_attention()

        # Get attention from this token to all others
        attention_scores = avg_attention[token_idx]

        return {token: float(score) for token, score in zip(self.tokens, attention_scores)}


class AttentionVisualizer:
    """Visualizer for transformer attention patterns."""

    def __init__(self, model: Any, tokenizer: Any):
        """Initialize attention visualizer.

        Args:
            model: Transformer model
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer

        LOGGER.info("Initialized AttentionVisualizer")

    def extract_attention(
        self,
        text: str,
        layer_idx: int | None = None,
    ) -> list[AttentionWeights]:
        """Extract attention weights from model.

        Args:
            text: Input text
            layer_idx: Specific layer to extract (None for all layers)

        Returns:
            List of attention weights per layer
        """
        LOGGER.info(f"Extracting attention for text: '{text[:50]}...'")

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Forward pass with attention output
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # Extract attention weights
        attentions = outputs.attentions  # Tuple of (batch, n_heads, seq_len, seq_len)

        attention_weights = []
        for idx, attn in enumerate(attentions):
            if layer_idx is not None and idx != layer_idx:
                continue

            # Convert to numpy and remove batch dimension
            attn_np = attn[0].cpu().numpy()

            attention_weights.append(
                AttentionWeights(
                    weights=attn_np,
                    tokens=tokens,
                    layer_idx=idx,
                )
            )

        LOGGER.info(f"Extracted attention from {len(attention_weights)} layers")

        return attention_weights

    def get_important_tokens(
        self,
        text: str,
        layer_idx: int = -1,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Get most important tokens based on attention.

        Args:
            text: Input text
            layer_idx: Layer to analyze (-1 for last layer)
            top_k: Number of top tokens

        Returns:
            List of (token, importance) tuples
        """
        # Extract attention
        attentions = self.extract_attention(text, layer_idx=None)

        if not attentions:
            return []

        # Use specified layer
        if layer_idx < 0:
            layer_idx = len(attentions) + layer_idx

        attention = attentions[layer_idx]

        # Calculate token importance as sum of attention received
        avg_attention = attention.get_average_attention()
        importance = avg_attention.sum(axis=0)  # Sum along source dimension

        # Get top k
        top_indices = np.argsort(importance)[::-1][:top_k]

        return [(attention.tokens[i], float(importance[i])) for i in top_indices]

    def analyze_token_relations(
        self,
        text: str,
        token_idx: int,
        layer_idx: int = -1,
        threshold: float = 0.1,
    ) -> dict[str, Any]:
        """Analyze relationships for a specific token.

        Args:
            text: Input text
            token_idx: Token index to analyze
            layer_idx: Layer to use
            threshold: Minimum attention threshold

        Returns:
            Analysis dictionary
        """
        # Extract attention
        attentions = self.extract_attention(text, layer_idx=None)

        if not attentions:
            return {}

        # Use specified layer
        if layer_idx < 0:
            layer_idx = len(attentions) + layer_idx

        attention = attentions[layer_idx]

        # Get attention for token
        token_attention = attention.get_token_attention(token_idx)

        # Filter by threshold
        strong_relations = {
            token: score
            for token, score in token_attention.items()
            if score >= threshold
        }

        # Sort by score
        sorted_relations = sorted(
            strong_relations.items(), key=lambda x: x[1], reverse=True
        )

        return {
            "token": attention.tokens[token_idx],
            "layer": layer_idx,
            "strong_relations": sorted_relations,
            "num_relations": len(sorted_relations),
            "mean_attention": np.mean(list(token_attention.values())),
            "max_attention": max(token_attention.values()),
        }


def visualize_attention(
    model: Any,
    tokenizer: Any,
    text: str,
    layer_idx: int | None = None,
) -> list[AttentionWeights]:
    """Visualize attention patterns (convenience function).

    Args:
        model: Transformer model
        tokenizer: Tokenizer
        text: Input text
        layer_idx: Specific layer to visualize

    Returns:
        List of attention weights
    """
    visualizer = AttentionVisualizer(model, tokenizer)
    return visualizer.extract_attention(text, layer_idx)
