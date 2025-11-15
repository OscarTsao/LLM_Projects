#!/usr/bin/env python
"""REST API for model serving (Phase 18).

This module provides a production-ready REST API including:
- FastAPI-based model serving endpoints
- Health check and readiness endpoints
- Metrics and monitoring endpoints
- Request validation and error handling
- OpenAPI documentation

Key Features:
- High-performance async API
- Automatic request/response validation
- Built-in monitoring integration
- Swagger UI documentation
"""

from __future__ import annotations

from psy_agents_noaug.api.models import (
    HealthResponse,
    MetricsResponse,
    PredictionRequest,
    PredictionResponse,
)
from psy_agents_noaug.api.server import (
    create_app,
    run_server,
)

__all__ = [
    # Server
    "create_app",
    "run_server",
    # Models
    "PredictionRequest",
    "PredictionResponse",
    "HealthResponse",
    "MetricsResponse",
]
