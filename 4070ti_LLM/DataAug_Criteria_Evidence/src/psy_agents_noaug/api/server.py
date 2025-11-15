#!/usr/bin/env python
"""FastAPI server for model serving (Phase 18).

This module provides a production-ready REST API server with:
- Prediction endpoints (single and batch)
- Health check and readiness endpoints
- Metrics endpoint
- Error handling and validation
- Monitoring integration
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from psy_agents_noaug.api.models import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    PredictionRequest,
    PredictionResponse,
)
from psy_agents_noaug.monitoring import HealthCheck, HealthMonitor, PerformanceMonitor

LOGGER = logging.getLogger(__name__)

# Global state
app_state: dict[str, Any] = {
    "start_time": None,
    "performance_monitor": None,
    "health_monitor": None,
    "model_loaded": False,
    "model_version": "v1.0.0",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    LOGGER.info("Starting API server...")
    app_state["start_time"] = time.time()
    app_state["performance_monitor"] = PerformanceMonitor(window_size=1000)
    app_state["health_monitor"] = HealthMonitor()

    # Add health checks
    app_state["health_monitor"].add_check(
        HealthCheck(
            name="model_loaded",
            checker=lambda: app_state["model_loaded"],
            description="Check if model is loaded",
            critical=True,
        )
    )

    app_state["health_monitor"].add_check(
        HealthCheck(
            name="performance",
            checker=lambda: True,  # Always pass for now
            description="Check performance metrics",
            critical=False,
        )
    )

    # Simulate model loading
    app_state["model_loaded"] = True
    LOGGER.info("Model loaded successfully")

    yield

    # Shutdown
    LOGGER.info("Shutting down API server...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="PSY Agents API",
        description="Production API for clinical text analysis models",
        version=app_state["model_version"],
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # Error handler
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        LOGGER.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                detail=str(exc),
                timestamp=datetime.now().isoformat(),
                request_id=getattr(request.state, "request_id", None),
            ).dict(),
        )

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "PSY Agents API",
            "version": app_state["model_version"],
            "status": "running",
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
        }

    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint.

        Returns:
            Health status and check results
        """
        summary = app_state["health_monitor"].get_summary()

        uptime = None
        if app_state["start_time"]:
            uptime = time.time() - app_state["start_time"]

        return HealthResponse(
            status=summary["overall_status"],
            timestamp=summary["timestamp"],
            checks={
                check["name"]: {
                    "status": "passed" if check["passed"] else "failed",
                    "message": check["message"],
                }
                for check in summary["checks"]
            },
            uptime_seconds=uptime,
        )

    # Readiness check endpoint
    @app.get("/ready")
    async def readiness_check():
        """Readiness check endpoint.

        Returns:
            Readiness status
        """
        if not app_state["model_loaded"]:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return {"status": "ready", "timestamp": datetime.now().isoformat()}

    # Metrics endpoint
    @app.get("/metrics", response_model=MetricsResponse)
    async def get_metrics():
        """Get performance metrics.

        Returns:
            Performance and resource metrics
        """
        monitor = app_state["performance_monitor"]
        summary = monitor.get_summary()

        return MetricsResponse(
            timestamp=summary["timestamp"],
            performance={
                "latency_p50_ms": summary["latency"]["p50_ms"],
                "latency_p95_ms": summary["latency"]["p95_ms"],
                "latency_p99_ms": summary["latency"]["p99_ms"],
                "latency_mean_ms": summary["latency"]["mean_ms"],
                "throughput_rps": summary["throughput_rps"],
            },
            requests={
                "total": summary["total_requests"],
                "errors": summary["total_errors"],
                "error_rate": summary["error_rate"],
            },
            resources=summary.get("resources"),
        )

    # Prediction endpoint
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """Make a single prediction.

        Args:
            request: Prediction request

        Returns:
            Prediction response
        """
        start_time = time.time()

        try:
            # Validate task
            if request.task not in ["criteria", "evidence", "share", "joint"]:
                raise HTTPException(  # noqa: TRY301
                    status_code=400,
                    detail=f"Invalid task: {request.task}",
                )

            # Mock prediction (replace with actual model inference)
            prediction = {
                "label": "positive",
                "score": 0.85,
            }
            confidence = 0.85

            # Record metrics
            latency = time.time() - start_time
            app_state["performance_monitor"].record_request(latency, error=False)

            return PredictionResponse(
                prediction=prediction,
                confidence=confidence,
                task=request.task,
                model_version=request.model_version or app_state["model_version"],
                latency_ms=latency * 1000,
                timestamp=datetime.now().isoformat(),
            )

        except HTTPException:
            raise
        except Exception as e:
            latency = time.time() - start_time
            app_state["performance_monitor"].record_request(latency, error=True)
            LOGGER.error(f"Prediction error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    # Batch prediction endpoint
    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(request: BatchPredictionRequest):
        """Make batch predictions.

        Args:
            request: Batch prediction request

        Returns:
            Batch prediction response
        """
        start_time = time.time()

        try:
            # Validate task
            if request.task not in ["criteria", "evidence", "share", "joint"]:
                raise HTTPException(  # noqa: TRY301
                    status_code=400,
                    detail=f"Invalid task: {request.task}",
                )

            # Mock predictions (replace with actual model inference)
            predictions = [{"label": "positive", "score": 0.85} for _ in request.texts]

            # Calculate metrics
            total_latency = time.time() - start_time
            avg_latency = total_latency / len(request.texts)

            # Record metrics (record as single batch request)
            app_state["performance_monitor"].record_request(total_latency, error=False)

            return BatchPredictionResponse(
                predictions=predictions,
                count=len(predictions),
                total_latency_ms=total_latency * 1000,
                avg_latency_ms=avg_latency * 1000,
                task=request.task,
                model_version=request.model_version or app_state["model_version"],
                timestamp=datetime.now().isoformat(),
            )

        except HTTPException:
            raise
        except Exception as e:
            latency = time.time() - start_time
            app_state["performance_monitor"].record_request(latency, error=True)
            LOGGER.error(f"Batch prediction error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
):
    """Run the API server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload
    """
    import uvicorn

    LOGGER.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "psy_agents_noaug.api.server:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )
