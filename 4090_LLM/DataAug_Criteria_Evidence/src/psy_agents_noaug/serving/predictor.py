#!/usr/bin/env python
"""Prediction server with monitoring and explainability (Phase 29).

This module provides a prediction service that integrates with
monitoring, explainability, and governance features.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class PredictionRequest:
    """Prediction request."""

    instance_id: str
    inputs: dict[str, Any]
    explain: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResponse:
    """Prediction response."""

    instance_id: str
    prediction: Any
    confidence: float | None = None
    latency_ms: float = 0.0
    explanation: dict[str, Any] | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "instance_id": self.instance_id,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "explanation": self.explanation,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class Predictor:
    """Prediction service with monitoring and explainability."""

    def __init__(
        self,
        model: Any,
        preprocessor: Any | None = None,
        postprocessor: Any | None = None,
        performance_monitor: Any | None = None,
        prediction_monitor: Any | None = None,
    ):
        """Initialize predictor.

        Args:
            model: Model for predictions
            preprocessor: Input preprocessor
            postprocessor: Output postprocessor
            performance_monitor: Performance monitor
            prediction_monitor: Prediction monitor
        """
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.performance_monitor = performance_monitor
        self.prediction_monitor = prediction_monitor

        self.total_predictions = 0
        self.total_errors = 0

        LOGGER.info("Initialized Predictor")

    def predict(
        self,
        request: PredictionRequest,
    ) -> PredictionResponse:
        """Make prediction.

        Args:
            request: Prediction request

        Returns:
            Prediction response
        """
        start_time = time.time()
        success = True
        error_type = None

        try:
            # Preprocess inputs
            inputs = (
                self.preprocessor(request.inputs)
                if self.preprocessor
                else request.inputs
            )

            # Convert to tensor if needed
            if isinstance(inputs, dict):
                inputs = {k: self._to_tensor(v) for k, v in inputs.items()}
            else:
                inputs = self._to_tensor(inputs)

            # Run model
            with torch.no_grad():
                outputs = self.model(
                    **inputs if isinstance(inputs, dict) else {"input": inputs}
                )

            # Postprocess outputs
            prediction = self.postprocessor(outputs) if self.postprocessor else outputs

            # Extract confidence if available
            confidence = self._extract_confidence(outputs)

            # Generate explanation if requested
            explanation = None
            if request.explain:
                explanation = self._generate_explanation(inputs, outputs)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Create response
            response = PredictionResponse(
                instance_id=request.instance_id,
                prediction=prediction,
                confidence=confidence,
                latency_ms=latency_ms,
                explanation=explanation,
            )

            # Update counters
            self.total_predictions += 1

            # Record metrics
            self._record_metrics(latency_ms, success, prediction, confidence)

            return response  # noqa: TRY300

        except Exception as e:
            # Handle error
            success = False
            error_type = type(e).__name__
            self.total_errors += 1

            latency_ms = (time.time() - start_time) * 1000

            LOGGER.exception("Prediction error")

            # Record error metrics
            self._record_metrics(latency_ms, success, None, None, error_type)

            raise

    def predict_batch(
        self,
        requests: list[PredictionRequest],
    ) -> list[PredictionResponse]:
        """Make batch predictions.

        Args:
            requests: List of prediction requests

        Returns:
            List of prediction responses
        """
        responses = []

        for request in requests:
            try:
                response = self.predict(request)
                responses.append(response)
            except Exception:
                LOGGER.exception(f"Batch prediction error for {request.instance_id}")
                # Continue with other requests

        return responses

    def _to_tensor(self, data: Any) -> torch.Tensor:
        """Convert data to tensor.

        Args:
            data: Input data

        Returns:
            Tensor
        """
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        return torch.tensor(data)

    def _extract_confidence(self, outputs: Any) -> float | None:
        """Extract confidence from model outputs.

        Args:
            outputs: Model outputs

        Returns:
            Confidence score or None
        """
        try:
            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=-1)
                return float(torch.max(probs).item())
            if isinstance(outputs, torch.Tensor):
                probs = torch.softmax(outputs, dim=-1)
                return float(torch.max(probs).item())
        except Exception as e:
            LOGGER.debug(f"Could not extract confidence: {e}")

        return None

    def _generate_explanation(
        self,
        inputs: Any,
        outputs: Any,
    ) -> dict[str, Any] | None:
        """Generate explanation for prediction.

        Args:
            inputs: Model inputs
            outputs: Model outputs

        Returns:
            Explanation dictionary or None
        """
        # Placeholder for explainability integration
        # In production, this would integrate with Phase 27 explainability
        return {
            "method": "attention",
            "top_features": [],
            "note": "Explainability integration placeholder",
        }

    def _record_metrics(
        self,
        latency_ms: float,
        success: bool,
        prediction: Any = None,
        confidence: float | None = None,
        error_type: str | None = None,
    ) -> None:
        """Record metrics to monitors.

        Args:
            latency_ms: Latency in milliseconds
            success: Whether prediction succeeded
            prediction: Prediction value
            confidence: Confidence score
            error_type: Error type if failed
        """
        # Record to performance monitor
        if self.performance_monitor:
            try:
                self.performance_monitor.record_request(
                    latency=latency_ms / 1000,  # Convert to seconds
                    success=success,
                    error_type=error_type,
                )
            except Exception as e:
                LOGGER.debug(f"Failed to record performance metrics: {e}")

        # Record to prediction monitor
        if self.prediction_monitor and success and prediction is not None:
            try:
                # Extract numeric prediction for monitoring
                if isinstance(prediction, int | float):
                    pred_value = float(prediction)
                elif isinstance(prediction, torch.Tensor):
                    pred_value = float(prediction.item())
                else:
                    pred_value = None

                if pred_value is not None:
                    self.prediction_monitor.record_prediction(
                        prediction=pred_value,
                        confidence=confidence,
                    )
            except Exception as e:
                LOGGER.debug(f"Failed to record prediction metrics: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get predictor statistics.

        Returns:
            Statistics dictionary
        """
        error_rate = (
            self.total_errors / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )

        return {
            "total_predictions": self.total_predictions,
            "total_errors": self.total_errors,
            "error_rate": error_rate,
        }


def create_predictor(
    model: Any,
    **kwargs: Any,
) -> Predictor:
    """Create predictor (convenience function).

    Args:
        model: Model for predictions
        **kwargs: Additional predictor arguments

    Returns:
        Predictor instance
    """
    return Predictor(model, **kwargs)
