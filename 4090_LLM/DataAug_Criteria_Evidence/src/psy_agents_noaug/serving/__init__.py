#!/usr/bin/env python
"""Model serving and deployment infrastructure (Phase 29).

This module provides production-ready model serving capabilities including:
- Model loading from registry or filesystem
- Prediction API with monitoring integration
- Batch inference support
- Health monitoring and validation
- Explainability integration

Key Components:
- ModelLoader: Load and cache models for serving
- Predictor: Make predictions with monitoring
- PredictionRequest/Response: Structured API contracts

Integration:
- Phase 26: Performance and prediction monitoring
- Phase 27: Explainability integration hooks
- Phase 28: Registry-based model loading

Example:
    ```python
    from psy_agents_noaug.serving import (
        ModelLoader,
        Predictor,
        PredictionRequest,
        load_model,
        create_predictor,
    )

    # Load model
    loader = ModelLoader(device="cuda")
    model = loader.load_pytorch_model("path/to/model.pt")

    # Create predictor
    predictor = create_predictor(
        model=model,
        performance_monitor=perf_monitor,
        prediction_monitor=pred_monitor,
    )

    # Make prediction
    request = PredictionRequest(
        instance_id="req_001",
        inputs={"text": "sample input"},
        explain=True,
    )
    response = predictor.predict(request)
    ```
"""

from __future__ import annotations

from psy_agents_noaug.serving.model_loader import (
    ModelLoader,
    load_model,
)
from psy_agents_noaug.serving.predictor import (
    PredictionRequest,
    PredictionResponse,
    Predictor,
    create_predictor,
)

__all__ = [
    # Model loading
    "ModelLoader",
    "load_model",
    # Prediction
    "Predictor",
    "PredictionRequest",
    "PredictionResponse",
    "create_predictor",
]
