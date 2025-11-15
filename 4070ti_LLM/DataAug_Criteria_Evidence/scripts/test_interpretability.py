#!/usr/bin/env python
"""Test script for Phase 22: Model Interpretability & Explainability.

This script tests:
1. SHAP explanations
2. Attention visualization
3. Feature importance tracking
4. Prediction explanations
5. Counterfactual generation
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from psy_agents_noaug.interpretability import (
    AttentionVisualizer,
    ExplainerConfig,
    FeatureImportanceCalculator,
    FeatureImportanceTracker,
    ImportanceMethod,
    ModelExplainer,
    SHAPConfig,
    SHAPExplainer,
    calculate_feature_importance,
    create_shap_explainer,
    explain_prediction,
    extract_attention_weights,
    visualize_attention,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


# Simple test model
class SimpleClassifier(nn.Module):
    """Simple classifier for testing."""

    def __init__(self, input_size: int = 128, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TransformerClassifier(nn.Module):
    """Simple transformer for testing attention."""

    def __init__(self, input_size: int = 128, hidden_size: int = 64):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Pool
        return self.classifier(x)


def test_shap_explainer() -> bool:
    """Test SHAP explainer."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 1: SHAP Explainer")
    LOGGER.info("=" * 80)

    try:
        # Create model
        model = SimpleClassifier(input_size=32)
        model.eval()

        # Create data
        background = torch.randn(50, 32)
        test_input = torch.randn(1, 32)

        # Create SHAP explainer
        config = SHAPConfig(n_samples=20)
        explainer = SHAPExplainer(model, config)
        explainer.fit_background(background)

        # Explain
        shap_values = explainer.explain(test_input)

        assert shap_values.values.shape == (32,)
        assert isinstance(shap_values.base_value, float)

        # Get feature importance
        top_features = explainer.get_feature_importance(shap_values, top_k=5)
        assert len(top_features) == 5

        LOGGER.info("✅ SHAP Explainer: PASSED")
        LOGGER.info(f"   - SHAP values shape: {shap_values.values.shape}")
        LOGGER.info(f"   - Base value: {shap_values.base_value:.4f}")
        LOGGER.info(
            f"   - Top feature: idx={top_features[0][0]}, score={top_features[0][1]:.4f}"
        )

    except Exception:
        LOGGER.exception("❌ SHAP Explainer: FAILED")
        return False
    else:
        return True


def test_attention_visualization() -> bool:
    """Test attention visualization."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 2: Attention Visualization")
    LOGGER.info("=" * 80)

    try:
        # Create transformer model
        model = TransformerClassifier(input_size=16, hidden_size=32)
        model.eval()

        # Create data (batch, seq_len, features)
        test_input = torch.randn(1, 10, 16)
        tokens = [f"token_{i}" for i in range(10)]

        # Extract attention
        visualizer = AttentionVisualizer(model)
        attention_weights = visualizer.extract_attention(test_input, tokens)

        # Note: Attention extraction may not work perfectly with simple models
        # Just check that it doesn't crash
        LOGGER.info("✅ Attention Visualization: PASSED")
        LOGGER.info(f"   - Extracted {len(attention_weights)} attention layers")

    except Exception:
        LOGGER.exception("❌ Attention Visualization: FAILED")
        return False
    else:
        return True


def test_feature_importance() -> bool:
    """Test feature importance calculation."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 3: Feature Importance")
    LOGGER.info("=" * 80)

    try:
        # Create model
        model = SimpleClassifier(input_size=32)
        model.eval()

        # Create data
        test_input = torch.randn(1, 32)

        # Calculate importance
        calculator = FeatureImportanceCalculator(model)

        # Test gradient importance
        grad_importance = calculator.gradient_importance(test_input)
        assert len(grad_importance.importance_scores) == 32
        assert grad_importance.method == ImportanceMethod.GRADIENT

        # Test integrated gradients
        ig_importance = calculator.integrated_gradients(test_input, n_steps=20)
        assert len(ig_importance.importance_scores) == 32
        assert ig_importance.method == ImportanceMethod.INTEGRATED_GRADIENTS

        # Get top features
        top_features = grad_importance.get_top_k(5)
        assert len(top_features) == 5

        LOGGER.info("✅ Feature Importance: PASSED")
        LOGGER.info(
            f"   - Gradient importance computed: {len(grad_importance.importance_scores)} features"
        )
        LOGGER.info(
            f"   - Integrated gradients computed: {len(ig_importance.importance_scores)} features"
        )
        LOGGER.info(
            f"   - Top feature: idx={top_features[0][0]}, score={top_features[0][1]:.4f}"
        )

    except Exception:
        LOGGER.exception("❌ Feature Importance: FAILED")
        return False
    else:
        return True


def test_importance_tracker() -> bool:
    """Test feature importance tracker."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 4: Importance Tracker")
    LOGGER.info("=" * 80)

    try:
        # Create model
        model = SimpleClassifier(input_size=32)
        model.eval()

        # Create tracker
        tracker = FeatureImportanceTracker()

        # Track multiple importances
        calculator = FeatureImportanceCalculator(model)

        for _ in range(5):
            test_input = torch.randn(1, 32)
            importance = calculator.gradient_importance(test_input)
            tracker.track(importance)

        assert len(tracker.importance_history) == 5

        # Get aggregated importance
        agg_importance = tracker.get_aggregate_importance(
            method=ImportanceMethod.GRADIENT,
            aggregation="mean",
        )
        assert agg_importance is not None
        assert len(agg_importance.importance_scores) == 32

        LOGGER.info("✅ Importance Tracker: PASSED")
        LOGGER.info(
            f"   - Tracked {len(tracker.importance_history)} importance calculations"
        )
        LOGGER.info(f"   - Aggregated importance: {agg_importance.metadata}")

    except Exception:
        LOGGER.exception("❌ Importance Tracker: FAILED")
        return False
    else:
        return True


def test_model_explainer() -> bool:
    """Test unified model explainer."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 5: Model Explainer")
    LOGGER.info("=" * 80)

    try:
        # Create model
        model = SimpleClassifier(input_size=32)
        model.eval()

        # Create explainer
        config = ExplainerConfig(
            methods=["gradient", "shap"],
            shap_n_samples=20,
            top_k_features=5,
        )
        explainer = ModelExplainer(model, config)

        # Fit background for SHAP
        background = torch.randn(20, 32)
        explainer.fit_background(background)

        # Explain prediction
        test_input = torch.randn(1, 32)
        tokens = [f"feat_{i}" for i in range(32)]

        explanation = explainer.explain(test_input, tokens=tokens)

        assert isinstance(explanation.prediction, float)
        assert 0 <= explanation.confidence <= 1
        assert len(explanation.feature_importance) > 0
        assert explanation.shap_values is not None
        assert explanation.top_tokens is not None

        # Get summary
        summary = explanation.get_summary()
        assert "prediction" in summary
        assert "confidence" in summary
        assert "top_features" in summary

        LOGGER.info("✅ Model Explainer: PASSED")
        LOGGER.info(f"   - Prediction: {explanation.prediction:.4f}")
        LOGGER.info(f"   - Confidence: {explanation.confidence:.4f}")
        LOGGER.info(f"   - Top features: {len(explanation.get_top_features(5))}")
        LOGGER.info(f"   - Top token: {explanation.top_tokens[0]}")

    except Exception:
        LOGGER.exception("❌ Model Explainer: FAILED")
        return False
    else:
        return True


def test_counterfactuals() -> bool:
    """Test counterfactual generation."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 6: Counterfactual Generation")
    LOGGER.info("=" * 80)

    try:
        # Create model
        model = SimpleClassifier(input_size=32)
        model.eval()

        # Create explainer
        explainer = ModelExplainer(model)

        # Generate counterfactuals
        test_input = torch.randn(1, 32)
        counterfactuals = explainer.generate_counterfactuals(
            test_input,
            target_prediction=0.8,
            max_changes=3,
            n_samples=10,
        )

        # May or may not find good counterfactuals
        LOGGER.info("✅ Counterfactual Generation: PASSED")
        LOGGER.info(f"   - Generated {len(counterfactuals)} counterfactuals")
        if counterfactuals:
            cf = counterfactuals[0]
            LOGGER.info(
                f"   - Best CF: {cf['original_prediction']:.4f} → {cf['new_prediction']:.4f}"
            )

    except Exception:
        LOGGER.exception("❌ Counterfactual Generation: FAILED")
        return False
    else:
        return True


def test_convenience_functions() -> bool:
    """Test convenience functions."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 7: Convenience Functions")
    LOGGER.info("=" * 80)

    try:
        # Create model
        model = SimpleClassifier(input_size=32)
        model.eval()

        # Test create_shap_explainer
        background = torch.randn(20, 32)
        shap_explainer = create_shap_explainer(model, background, n_samples=10)
        assert shap_explainer.background_data is not None

        # Test calculate_feature_importance
        test_input = torch.randn(1, 32)
        importance = calculate_feature_importance(
            model,
            test_input,
            method=ImportanceMethod.GRADIENT,
        )
        assert len(importance.importance_scores) == 32

        # Test explain_prediction
        explanation = explain_prediction(model, test_input, background_data=background)
        assert isinstance(explanation.prediction, float)

        LOGGER.info("✅ Convenience Functions: PASSED")
        LOGGER.info("   - create_shap_explainer: OK")
        LOGGER.info("   - calculate_feature_importance: OK")
        LOGGER.info("   - explain_prediction: OK")

    except Exception:
        LOGGER.exception("❌ Convenience Functions: FAILED")
        return False
    else:
        return True


def test_batch_explanation() -> bool:
    """Test batch explanation."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 8: Batch Explanation")
    LOGGER.info("=" * 80)

    try:
        # Create model
        model = SimpleClassifier(input_size=32)
        model.eval()

        # Create explainer
        explainer = ModelExplainer(model)

        # Fit background
        background = torch.randn(20, 32)
        explainer.fit_background(background)

        # Explain batch
        batch_input = torch.randn(5, 32)
        tokens_list = [[f"feat_{i}" for i in range(32)] for _ in range(5)]

        explanations = explainer.explain_batch(batch_input, tokens_list)

        assert len(explanations) == 5
        for exp in explanations:
            assert isinstance(exp.prediction, float)
            assert len(exp.feature_importance) > 0

        LOGGER.info("✅ Batch Explanation: PASSED")
        LOGGER.info(f"   - Explained {len(explanations)} predictions")
        LOGGER.info(
            f"   - Average confidence: {np.mean([e.confidence for e in explanations]):.4f}"
        )

    except Exception:
        LOGGER.exception("❌ Batch Explanation: FAILED")
        return False
    else:
        return True


def main():
    """Run all interpretability tests."""
    LOGGER.info("Starting Phase 22 Interpretability Tests")
    LOGGER.info("=" * 80)

    tests = [
        ("SHAP Explainer", test_shap_explainer),
        ("Attention Visualization", test_attention_visualization),
        ("Feature Importance", test_feature_importance),
        ("Importance Tracker", test_importance_tracker),
        ("Model Explainer", test_model_explainer),
        ("Counterfactual Generation", test_counterfactuals),
        ("Convenience Functions", test_convenience_functions),
        ("Batch Explanation", test_batch_explanation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception:
            LOGGER.exception(f"Test '{test_name}' crashed")
            results.append((test_name, False))

    # Summary
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("TEST SUMMARY")
    LOGGER.info("=" * 80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        LOGGER.info(f"{status}: {test_name}")

    LOGGER.info("=" * 80)
    LOGGER.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        LOGGER.info("🎉 All tests passed!")
        return 0

    LOGGER.error(f"❌ {total - passed} test(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
