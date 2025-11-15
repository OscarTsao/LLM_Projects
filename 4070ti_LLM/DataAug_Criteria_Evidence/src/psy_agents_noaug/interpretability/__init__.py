#!/usr/bin/env python
"""Model Interpretability & Explainability (Phase 22).

This module provides tools for understanding and explaining model predictions:
- SHAP-based explanations
- Attention visualization for transformers
- Feature importance tracking
- Prediction explanations
- Counterfactual analysis
"""

from __future__ import annotations

from psy_agents_noaug.interpretability.attention import (
    AttentionVisualizer,
    extract_attention_weights,
    visualize_attention,
)
from psy_agents_noaug.interpretability.explainer import (
    Explanation,
    ExplainerConfig,
    ModelExplainer,
    explain_prediction,
)
from psy_agents_noaug.interpretability.feature_importance import (
    FeatureImportance,
    FeatureImportanceCalculator,
    FeatureImportanceTracker,
    ImportanceMethod,
    calculate_feature_importance,
)
from psy_agents_noaug.interpretability.shap_explainer import (
    SHAPConfig,
    SHAPExplainer,
    SHAPValues,
    create_shap_explainer,
)

__all__ = [
    # Attention visualization
    "AttentionVisualizer",
    "extract_attention_weights",
    "visualize_attention",
    # Model explainer
    "Explanation",
    "ExplainerConfig",
    "ModelExplainer",
    "explain_prediction",
    # Feature importance
    "FeatureImportance",
    "FeatureImportanceCalculator",
    "FeatureImportanceTracker",
    "ImportanceMethod",
    "calculate_feature_importance",
    # SHAP
    "SHAPConfig",
    "SHAPExplainer",
    "SHAPValues",
    "create_shap_explainer",
]
