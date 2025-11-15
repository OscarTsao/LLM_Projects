#!/usr/bin/env python
"""Performance monitoring for model inference (Phase 17).

This module provides performance monitoring including:
- Latency tracking (p50, p95, p99)
- Throughput measurement
- Resource usage (CPU, memory, GPU)
- Request rate limiting
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""

    timestamp: datetime
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_mean: float
    throughput: float  # requests per second
    error_rate: float  # fraction of errors
    total_requests: int
    total_errors: int


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""

    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: float = 0.0
    gpu_utilization: float = 0.0


class PerformanceMonitor:
    """Monitor model performance metrics."""

    def __init__(
        self,
        window_size: int = 1000,
        track_resources: bool = True,
    ):
        """Initialize performance monitor.

        Args:
            window_size: Number of recent requests to track
            track_resources: Whether to track resource usage
        """
        self.window_size = window_size
        self.track_resources = track_resources

        # Latency tracking (in seconds)
        self.latencies: deque[float] = deque(maxlen=window_size)

        # Error tracking
        self.total_requests = 0
        self.total_errors = 0

        # Throughput tracking
        self.request_timestamps: deque[float] = deque(maxlen=window_size)

        # Resource tracking
        self.resource_metrics: list[ResourceMetrics] = []

        LOGGER.info(
            "Initialized PerformanceMonitor (window_size=%d)",
            window_size,
        )

    def record_request(
        self,
        latency: float,
        error: bool = False,
    ) -> None:
        """Record a single request.

        Args:
            latency: Request latency in seconds
            error: Whether the request resulted in an error
        """
        self.latencies.append(latency)
        self.request_timestamps.append(time.time())
        self.total_requests += 1

        if error:
            self.total_errors += 1

    def get_latency_percentiles(self) -> dict[str, float]:
        """Get latency percentiles.

        Returns:
            Dict with p50, p95, p99, mean latencies
        """
        if not self.latencies:
            return {
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "mean": 0.0,
            }

        latencies_array = np.array(list(self.latencies))

        return {
            "p50": float(np.percentile(latencies_array, 50)),
            "p95": float(np.percentile(latencies_array, 95)),
            "p99": float(np.percentile(latencies_array, 99)),
            "mean": float(np.mean(latencies_array)),
        }

    def get_throughput(self) -> float:
        """Get current throughput (requests per second).

        Returns:
            Requests per second
        """
        if len(self.request_timestamps) < 2:
            return 0.0

        timestamps = list(self.request_timestamps)
        time_window = timestamps[-1] - timestamps[0]

        if time_window == 0:
            return 0.0

        return len(timestamps) / time_window

    def get_error_rate(self) -> float:
        """Get error rate.

        Returns:
            Fraction of requests that resulted in errors
        """
        if self.total_requests == 0:
            return 0.0

        return self.total_errors / self.total_requests

    def _capture_resource_metrics(self) -> ResourceMetrics:
        """Capture current resource usage.

        Returns:
            Resource metrics
        """
        import psutil

        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_mb = psutil.virtual_memory().used / (1024 * 1024)

        # GPU metrics (if available)
        gpu_memory_mb = 0.0
        gpu_utilization = 0.0

        try:
            import torch

            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                # Note: GPU utilization requires nvidia-ml-py
                # We'll just use memory as a proxy for now
                gpu_utilization = (
                    gpu_memory_mb
                    / (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
                ) * 100
        except ImportError:
            pass

        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            gpu_utilization=gpu_utilization,
        )

    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics snapshot.

        Returns:
            Performance metrics
        """
        latency_stats = self.get_latency_percentiles()

        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            latency_p50=latency_stats["p50"],
            latency_p95=latency_stats["p95"],
            latency_p99=latency_stats["p99"],
            latency_mean=latency_stats["mean"],
            throughput=self.get_throughput(),
            error_rate=self.get_error_rate(),
            total_requests=self.total_requests,
            total_errors=self.total_errors,
        )

        # Optionally capture resource metrics
        if self.track_resources:
            resource_metrics = self._capture_resource_metrics()
            self.resource_metrics.append(resource_metrics)

        return metrics

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all metrics.

        Returns:
            Summary dict
        """
        metrics = self.get_metrics()

        summary = {
            "timestamp": metrics.timestamp.isoformat(),
            "latency": {
                "p50_ms": metrics.latency_p50 * 1000,
                "p95_ms": metrics.latency_p95 * 1000,
                "p99_ms": metrics.latency_p99 * 1000,
                "mean_ms": metrics.latency_mean * 1000,
            },
            "throughput_rps": metrics.throughput,
            "error_rate": metrics.error_rate,
            "total_requests": metrics.total_requests,
            "total_errors": metrics.total_errors,
        }

        # Add resource metrics if available
        if self.resource_metrics:
            latest_resources = self.resource_metrics[-1]
            summary["resources"] = {
                "cpu_percent": latest_resources.cpu_percent,
                "memory_mb": latest_resources.memory_mb,
                "gpu_memory_mb": latest_resources.gpu_memory_mb,
                "gpu_utilization": latest_resources.gpu_utilization,
            }

        return summary

    def reset(self) -> None:
        """Reset all metrics."""
        self.latencies.clear()
        self.request_timestamps.clear()
        self.total_requests = 0
        self.total_errors = 0
        self.resource_metrics.clear()

        LOGGER.info("Reset performance metrics")


def monitor_performance(
    latencies: list[float],
    errors: list[bool] | None = None,
    window_size: int = 1000,
) -> PerformanceMetrics:
    """Monitor performance metrics (convenience function).

    Args:
        latencies: List of request latencies
        errors: List of error indicators
        window_size: Window size for tracking

    Returns:
        Performance metrics
    """
    monitor = PerformanceMonitor(window_size=window_size)

    if errors is None:
        errors = [False] * len(latencies)

    for latency, error in zip(latencies, errors, strict=False):
        monitor.record_request(latency, error)

    return monitor.get_metrics()
