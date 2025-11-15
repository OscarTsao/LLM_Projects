#!/usr/bin/env python
"""Performance monitoring for model inference (Phase 26).

This module provides tools for monitoring model performance including latency,
throughput, resource usage, and error rates.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a time window."""

    # Latency metrics (seconds)
    mean_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    max_latency: float

    # Throughput metrics
    requests_per_second: float
    total_requests: int

    # Error metrics
    error_rate: float
    total_errors: int

    # Resource metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    # Time window
    window_start: datetime = field(default_factory=datetime.now)
    window_end: datetime = field(default_factory=datetime.now)

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary.

        Returns:
            Summary dictionary
        """
        return {
            "latency": {
                "mean": self.mean_latency,
                "p50": self.p50_latency,
                "p95": self.p95_latency,
                "p99": self.p99_latency,
                "max": self.max_latency,
            },
            "throughput": {
                "requests_per_second": self.requests_per_second,
                "total_requests": self.total_requests,
            },
            "errors": {
                "error_rate": self.error_rate,
                "total_errors": self.total_errors,
            },
            "resources": {
                "memory_mb": self.memory_usage_mb,
                "cpu_percent": self.cpu_usage_percent,
            },
            "window": {
                "start": self.window_start.isoformat(),
                "end": self.window_end.isoformat(),
                "duration_seconds": (self.window_end - self.window_start).total_seconds(),
            },
        }


@dataclass
class RequestRecord:
    """Single request record."""

    timestamp: datetime
    latency: float  # seconds
    success: bool
    error_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Monitor model performance metrics."""

    def __init__(
        self,
        window_size: int = 1000,  # Keep last N requests
        aggregation_interval: int = 60,  # Aggregate every N seconds
    ):
        """Initialize performance monitor.

        Args:
            window_size: Number of recent requests to keep
            aggregation_interval: Seconds between metric aggregations
        """
        self.window_size = window_size
        self.aggregation_interval = aggregation_interval

        # Request history
        self.requests: deque[RequestRecord] = deque(maxlen=window_size)

        # Current metrics
        self.current_metrics: PerformanceMetrics | None = None
        self.last_aggregation: datetime = datetime.now()

        # Counters
        self.total_requests = 0
        self.total_errors = 0

        LOGGER.info(
            f"Initialized PerformanceMonitor "
            f"(window_size={window_size}, interval={aggregation_interval}s)"
        )

    def record_request(
        self,
        latency: float,
        success: bool = True,
        error_type: str | None = None,
        **metadata: Any,
    ) -> None:
        """Record a request.

        Args:
            latency: Request latency in seconds
            success: Whether request succeeded
            error_type: Type of error if failed
            **metadata: Additional metadata
        """
        record = RequestRecord(
            timestamp=datetime.now(),
            latency=latency,
            success=success,
            error_type=error_type,
            metadata=metadata,
        )

        self.requests.append(record)
        self.total_requests += 1

        if not success:
            self.total_errors += 1

        # Check if we should aggregate
        if self._should_aggregate():
            self._aggregate_metrics()

    def _should_aggregate(self) -> bool:
        """Check if metrics should be aggregated.

        Returns:
            True if aggregation is due
        """
        elapsed = (datetime.now() - self.last_aggregation).total_seconds()
        return elapsed >= self.aggregation_interval

    def _aggregate_metrics(self) -> None:
        """Aggregate metrics from request history."""
        if not self.requests:
            return

        # Get requests in current window
        now = datetime.now()
        window_start = now - timedelta(seconds=self.aggregation_interval)

        recent_requests = [
            r for r in self.requests if r.timestamp >= window_start
        ]

        if not recent_requests:
            return

        # Calculate latency metrics
        latencies = sorted([r.latency for r in recent_requests])
        n = len(latencies)

        mean_latency = sum(latencies) / n
        p50_latency = latencies[int(n * 0.50)]
        p95_latency = latencies[int(n * 0.95)]
        p99_latency = latencies[int(n * 0.99)]
        max_latency = latencies[-1]

        # Calculate throughput
        duration = (now - window_start).total_seconds()
        requests_per_second = len(recent_requests) / duration if duration > 0 else 0.0

        # Calculate error rate
        errors = sum(1 for r in recent_requests if not r.success)
        error_rate = errors / len(recent_requests) if recent_requests else 0.0

        # Create metrics
        self.current_metrics = PerformanceMetrics(
            mean_latency=mean_latency,
            p50_latency=p50_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            max_latency=max_latency,
            requests_per_second=requests_per_second,
            total_requests=len(recent_requests),
            error_rate=error_rate,
            total_errors=errors,
            window_start=window_start,
            window_end=now,
        )

        self.last_aggregation = now

        LOGGER.debug(
            f"Aggregated metrics: "
            f"mean_latency={mean_latency:.3f}s, "
            f"rps={requests_per_second:.1f}, "
            f"error_rate={error_rate:.2%}"
        )

    def get_current_metrics(self) -> PerformanceMetrics | None:
        """Get current aggregated metrics.

        Returns:
            Current metrics or None if not yet aggregated
        """
        # Force aggregation if due
        if self._should_aggregate():
            self._aggregate_metrics()

        return self.current_metrics

    def get_metrics_for_window(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> PerformanceMetrics | None:
        """Get metrics for specific time window.

        Args:
            start_time: Window start (None for all available)
            end_time: Window end (None for now)

        Returns:
            Metrics for window or None if no data
        """
        if not self.requests:
            return None

        end_time = end_time or datetime.now()
        start_time = start_time or self.requests[0].timestamp

        # Filter requests in window
        windowed_requests = [
            r
            for r in self.requests
            if start_time <= r.timestamp <= end_time
        ]

        if not windowed_requests:
            return None

        # Calculate metrics
        latencies = sorted([r.latency for r in windowed_requests])
        n = len(latencies)

        duration = (end_time - start_time).total_seconds()
        errors = sum(1 for r in windowed_requests if not r.success)

        return PerformanceMetrics(
            mean_latency=sum(latencies) / n,
            p50_latency=latencies[int(n * 0.50)],
            p95_latency=latencies[int(n * 0.95)],
            p99_latency=latencies[int(n * 0.99)],
            max_latency=latencies[-1],
            requests_per_second=len(windowed_requests) / duration if duration > 0 else 0.0,
            total_requests=len(windowed_requests),
            error_rate=errors / len(windowed_requests),
            total_errors=errors,
            window_start=start_time,
            window_end=end_time,
        )

    def get_error_summary(self) -> dict[str, Any]:
        """Get summary of errors.

        Returns:
            Error summary with counts by type
        """
        error_counts: dict[str, int] = {}

        for request in self.requests:
            if not request.success and request.error_type:
                error_counts[request.error_type] = (
                    error_counts.get(request.error_type, 0) + 1
                )

        return {
            "total_errors": self.total_errors,
            "total_requests": self.total_requests,
            "overall_error_rate": (
                self.total_errors / self.total_requests
                if self.total_requests > 0
                else 0.0
            ),
            "errors_by_type": error_counts,
        }

    def reset(self) -> None:
        """Reset all metrics and history."""
        self.requests.clear()
        self.current_metrics = None
        self.last_aggregation = datetime.now()
        self.total_requests = 0
        self.total_errors = 0

        LOGGER.info("Reset PerformanceMonitor")


class LatencyTracker:
    """Context manager for tracking request latency."""

    def __init__(self, monitor: PerformanceMonitor):
        """Initialize latency tracker.

        Args:
            monitor: Performance monitor to record to
        """
        self.monitor = monitor
        self.start_time: float | None = None
        self.success = True
        self.error_type: str | None = None

    def __enter__(self) -> LatencyTracker:
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """End timing and record."""
        if self.start_time is None:
            return

        latency = time.time() - self.start_time

        # Check for exception
        if exc_type is not None:
            self.success = False
            self.error_type = exc_type.__name__

        self.monitor.record_request(
            latency=latency,
            success=self.success,
            error_type=self.error_type,
        )

    def mark_error(self, error_type: str) -> None:
        """Mark request as failed.

        Args:
            error_type: Type of error
        """
        self.success = False
        self.error_type = error_type


def track_latency(monitor: PerformanceMonitor) -> LatencyTracker:
    """Create latency tracker context manager.

    Args:
        monitor: Performance monitor

    Returns:
        Latency tracker context manager

    Example:
        with track_latency(monitor):
            result = model.predict(data)
    """
    return LatencyTracker(monitor)
