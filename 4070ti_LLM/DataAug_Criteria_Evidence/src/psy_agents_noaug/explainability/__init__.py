#!/usr/bin/env python
"""Model explainability and interpretability (Phase 27).

This module provides tools for explaining model predictions including:
- Feature importance analysis (permutation, gradient-based, integrated gradients)
- Attention visualization for transformer models
- Explanation aggregation and comparison across methods
- Consensus feature identification

Key Features:
- Multiple explanation methods with consistent APIs
- Support for both gradient-based and model-agnostic approaches
- Transformer attention analysis
- Explanation comparison and aggregation
"""

from __future__ import annotations

from psy_agents_noaug.explainability.attention_viz import (
    AttentionVisualizer,
    AttentionWeights,
    visualize_attention,
)
from psy_agents_noaug.explainability.explanations import (
    Explanation,
    ExplanationAggregator,
    create_explanation,
)
from psy_agents_noaug.explainability.feature_importance import (
    FeatureImportance,
    FeatureImportanceAnalyzer,
    calculate_feature_importance,
)

__all__ = [
    # Feature importance
    "FeatureImportance",
    "FeatureImportanceAnalyzer",
    "calculate_feature_importance",
    # Attention visualization
    "AttentionVisualizer",
    "AttentionWeights",
    "visualize_attention",
    # Explanations
    "Explanation",
    "ExplanationAggregator",
    "create_explanation",
]
