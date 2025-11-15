#!/usr/bin/env python
"""Pydantic models for API requests and responses (Phase 18).

This module defines data models for:
- Prediction requests and responses
- Health check responses
- Metrics responses
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for predictions."""

    text: str = Field(..., description="Input text for prediction", min_length=1)
    task: str = Field(
        "criteria",
        description="Task type (criteria, evidence, share, joint)",
    )
    model_version: str | None = Field(
        None,
        description="Optional model version to use",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Patient shows symptoms of depression.",
                "task": "criteria",
                "model_version": "v1.0.0",
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    prediction: Any = Field(..., description="Model prediction")
    confidence: float | None = Field(
        None,
        description="Prediction confidence score",
        ge=0.0,
        le=1.0,
    )
    task: str = Field(..., description="Task type")
    model_version: str = Field(..., description="Model version used")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": {"label": "positive", "score": 0.95},
                "confidence": 0.95,
                "task": "criteria",
                "model_version": "v1.0.0",
                "latency_ms": 45.2,
                "timestamp": "2025-01-15T10:30:00Z",
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""

    texts: list[str] = Field(
        ...,
        description="List of input texts",
        min_length=1,
        max_length=100,
    )
    task: str = Field(
        "criteria",
        description="Task type (criteria, evidence, share, joint)",
    )
    model_version: str | None = Field(
        None,
        description="Optional model version to use",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Patient shows symptoms of depression.",
                    "No significant symptoms observed.",
                ],
                "task": "criteria",
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: list[Any] = Field(..., description="List of predictions")
    count: int = Field(..., description="Number of predictions")
    total_latency_ms: float = Field(
        ...,
        description="Total latency in milliseconds",
    )
    avg_latency_ms: float = Field(
        ...,
        description="Average latency per prediction",
    )
    task: str = Field(..., description="Task type")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Batch timestamp (ISO format)")


class HealthResponse(BaseModel):
    """Response model for health checks."""

    status: str = Field(..., description="Health status (healthy, degraded, unhealthy)")
    timestamp: str = Field(..., description="Health check timestamp (ISO format)")
    checks: dict[str, Any] = Field(
        default_factory=dict,
        description="Individual check results",
    )
    uptime_seconds: float | None = Field(
        None,
        description="Server uptime in seconds",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-01-15T10:30:00Z",
                "checks": {
                    "model_loaded": {"status": "passed", "message": "Model loaded"},
                    "memory": {"status": "passed", "message": "Memory usage OK"},
                },
                "uptime_seconds": 3600.5,
            }
        }


class MetricsResponse(BaseModel):
    """Response model for metrics."""

    timestamp: str = Field(..., description="Metrics timestamp (ISO format)")
    performance: dict[str, Any] = Field(
        default_factory=dict,
        description="Performance metrics",
    )
    requests: dict[str, Any] = Field(
        default_factory=dict,
        description="Request statistics",
    )
    resources: dict[str, Any] | None = Field(
        None,
        description="Resource usage metrics",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-01-15T10:30:00Z",
                "performance": {
                    "latency_p50_ms": 45.2,
                    "latency_p95_ms": 89.5,
                    "latency_p99_ms": 120.3,
                    "throughput_rps": 250.5,
                },
                "requests": {
                    "total": 10000,
                    "success": 9950,
                    "errors": 50,
                    "error_rate": 0.005,
                },
                "resources": {
                    "cpu_percent": 45.2,
                    "memory_mb": 2048.5,
                },
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp (ISO format)")
    request_id: str | None = Field(None, description="Request ID for tracking")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid input",
                "detail": "Text field cannot be empty",
                "timestamp": "2025-01-15T10:30:00Z",
                "request_id": "req-12345",
            }
        }
