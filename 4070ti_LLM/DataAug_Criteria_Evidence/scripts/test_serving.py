#!/usr/bin/env python
"""Test script for Phase 29: Model Serving & Deployment Infrastructure.

This script validates the model serving components including:
- Model loading from filesystem and registry
- Prediction API with request/response contracts
- Batch inference capabilities
- Monitor integration
- Model caching

Usage:
    python scripts/test_serving.py
    make test-serving
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch import nn

# Phase 29: Model Serving
from psy_agents_noaug.serving import (
    ModelLoader,
    PredictionRequest,
    PredictionResponse,
    Predictor,
    create_predictor,
)


class DummyModel(nn.Module):
    """Dummy PyTorch model for testing."""

    def __init__(self, input_dim: int = 10, output_dim: int = 2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, input: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass returning dict with logits."""
        logits = self.fc(input)
        return {"logits": logits}


def test_model_loader():
    """Test ModelLoader functionality."""
    print("\n=== Testing ModelLoader ===")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        # Create and save dummy model
        model = DummyModel()
        model_path = tmpdir / "dummy_model.pt"
        torch.save(model.state_dict(), model_path)

        print(f"✓ Created dummy model at {model_path}")

        # Test loading
        loader = ModelLoader(device="cpu")
        loaded_model = loader.load_pytorch_model(
            model_path, model_class=DummyModel, input_dim=10, output_dim=2
        )

        print("✓ Loaded model successfully")
        print("  - Device: cpu")
        print(f"  - Model type: {type(loaded_model).__name__}")

        # Test model inference
        test_input = torch.randn(1, 10)
        with torch.no_grad():
            output = loaded_model(test_input)

        assert "logits" in output, "Model output should contain 'logits'"
        print(f"✓ Model inference works (output shape: {output['logits'].shape})")

        # Test caching
        cache_key = "test_model:v1"
        loader.loaded_models[cache_key] = loaded_model
        cached = loader.get_cached_model(cache_key)

        assert cached is loaded_model, "Should retrieve cached model"
        print("✓ Model caching works")

        # Test cache clearing
        loader.clear_cache(cache_key)
        assert loader.get_cached_model(cache_key) is None, "Cache should be cleared"
        print("✓ Cache clearing works")

    print("\n✅ ModelLoader tests passed")


def test_predictor():
    """Test Predictor functionality."""
    print("\n=== Testing Predictor ===")

    # Create dummy model
    model = DummyModel()
    model.eval()

    # Create predictor
    predictor = create_predictor(model)

    print("✓ Created Predictor")

    # Test single prediction
    request = PredictionRequest(
        instance_id="test_001",
        inputs={"input": torch.randn(1, 10)},
        explain=False,
    )

    response = predictor.predict(request)

    print("✓ Single prediction completed")
    print(f"  - Instance ID: {response.instance_id}")
    print(f"  - Latency: {response.latency_ms:.2f} ms")
    print(f"  - Confidence: {response.confidence}")

    assert response.instance_id == "test_001", "Instance ID should match"
    assert response.latency_ms > 0, "Latency should be positive"
    assert response.prediction is not None, "Should have prediction"

    # Test batch prediction
    requests = [
        PredictionRequest(
            instance_id=f"test_{i:03d}",
            inputs={"input": torch.randn(1, 10)},
        )
        for i in range(5)
    ]

    responses = predictor.predict_batch(requests)

    print(f"✓ Batch prediction completed ({len(responses)} predictions)")

    assert len(responses) == 5, "Should have 5 responses"
    for i, resp in enumerate(responses):
        assert resp.instance_id == f"test_{i:03d}", f"Instance ID {i} should match"

    # Test stats
    stats = predictor.get_stats()
    print("✓ Predictor stats retrieved")
    print(f"  - Total predictions: {stats['total_predictions']}")
    print(f"  - Total errors: {stats['total_errors']}")
    print(f"  - Error rate: {stats['error_rate']:.2%}")

    assert stats["total_predictions"] == 6, "Should have 6 total predictions"
    assert stats["error_rate"] == 0.0, "Error rate should be 0"

    print("\n✅ Predictor tests passed")


def test_prediction_contracts():
    """Test PredictionRequest and PredictionResponse."""
    print("\n=== Testing Prediction Contracts ===")

    # Test PredictionRequest
    request = PredictionRequest(
        instance_id="req_001",
        inputs={"text": "sample input"},
        explain=True,
        metadata={"user_id": "user123"},
    )

    print("✓ Created PredictionRequest")
    print(f"  - Instance ID: {request.instance_id}")
    print(f"  - Explain: {request.explain}")
    print(f"  - Metadata: {request.metadata}")

    # Test PredictionResponse
    response = PredictionResponse(
        instance_id="req_001",
        prediction=1,
        confidence=0.95,
        latency_ms=15.5,
        explanation={"method": "attention", "top_features": []},
        metadata={"model_version": "v1.0"},
    )

    print("✓ Created PredictionResponse")
    print(f"  - Instance ID: {response.instance_id}")
    print(f"  - Prediction: {response.prediction}")
    print(f"  - Confidence: {response.confidence}")
    print(f"  - Latency: {response.latency_ms} ms")

    # Test to_dict
    response_dict = response.to_dict()
    assert "instance_id" in response_dict, "Should have instance_id"
    assert "prediction" in response_dict, "Should have prediction"
    assert "confidence" in response_dict, "Should have confidence"
    assert "latency_ms" in response_dict, "Should have latency_ms"
    assert "explanation" in response_dict, "Should have explanation"
    assert "timestamp" in response_dict, "Should have timestamp"

    print("✓ to_dict() conversion works")

    print("\n✅ Prediction contract tests passed")


def test_confidence_extraction():
    """Test confidence extraction from model outputs."""
    print("\n=== Testing Confidence Extraction ===")

    model = DummyModel()
    predictor = Predictor(model)

    # Test with logits dict
    logits = torch.tensor([[1.0, 2.0, 0.5]])
    outputs = {"logits": logits}

    confidence = predictor._extract_confidence(outputs)
    print(f"✓ Extracted confidence from logits dict: {confidence:.4f}")

    assert confidence is not None, "Should extract confidence"
    assert 0 <= confidence <= 1, "Confidence should be in [0, 1]"

    # Test with tensor directly
    confidence2 = predictor._extract_confidence(logits)
    print(f"✓ Extracted confidence from tensor: {confidence2:.4f}")

    print("\n✅ Confidence extraction tests passed")


def test_monitor_integration():
    """Test integration with monitors (mock)."""
    print("\n=== Testing Monitor Integration ===")

    # Create dummy monitors
    class DummyMonitor:
        def __init__(self):
            self.records = []

        def record_request(self, **kwargs):
            self.records.append(kwargs)

        def record_prediction(self, **kwargs):
            self.records.append(kwargs)

    perf_monitor = DummyMonitor()
    pred_monitor = DummyMonitor()

    # Create predictor with monitors
    model = DummyModel()
    predictor = Predictor(
        model=model,
        performance_monitor=perf_monitor,
        prediction_monitor=pred_monitor,
    )

    print("✓ Created Predictor with monitors")

    # Make prediction
    request = PredictionRequest(
        instance_id="test_monitor",
        inputs={"input": torch.randn(1, 10)},
    )

    predictor.predict(request)

    print("✓ Made prediction with monitoring")
    print(f"  - Performance records: {len(perf_monitor.records)}")
    print(f"  - Prediction records: {len(pred_monitor.records)}")

    assert len(perf_monitor.records) > 0, "Should record to performance monitor"

    print("\n✅ Monitor integration tests passed")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Phase 29: Model Serving & Deployment Infrastructure - Test Suite")
    print("=" * 70)

    try:
        test_model_loader()
        test_predictor()
        test_prediction_contracts()
        test_confidence_extraction()
        test_monitor_integration()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        raise


if __name__ == "__main__":
    main()
