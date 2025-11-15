#!/usr/bin/env python
"""Experiment tracking and metrics aggregation (Phase 21).

This module provides:
- Experiment result tracking
- Metric aggregation
- Time-series analysis
- Conversion tracking
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class MetricEvent:
    """Metric tracking event."""

    experiment_id: str
    variant_id: str
    metric_name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperimentTracker:
    """Tracker for experiment metrics."""

    def __init__(self):
        """Initialize experiment tracker."""
        self.events: list[MetricEvent] = []
        self.variant_assignments: dict[str, str] = {}  # user_id -> variant_id
        LOGGER.info("Initialized ExperimentTracker")

    def track_event(self, event: MetricEvent) -> None:
        """Track a metric event.

        Args:
            event: Metric event to track
        """
        self.events.append(event)

        # Track variant assignment
        if event.user_id:
            key = f"{event.experiment_id}:{event.user_id}"
            self.variant_assignments[key] = event.variant_id

    def get_events(
        self,
        experiment_id: str,
        variant_id: str | None = None,
        metric_name: str | None = None,
    ) -> list[MetricEvent]:
        """Get filtered events.

        Args:
            experiment_id: Experiment identifier
            variant_id: Filter by variant (optional)
            metric_name: Filter by metric (optional)

        Returns:
            List of matching events
        """
        events = [e for e in self.events if e.experiment_id == experiment_id]

        if variant_id:
            events = [e for e in events if e.variant_id == variant_id]

        if metric_name:
            events = [e for e in events if e.metric_name == metric_name]

        return events

    def get_variant_assignment(self, experiment_id: str, user_id: str) -> str | None:
        """Get variant assignment for a user.

        Args:
            experiment_id: Experiment identifier
            user_id: User identifier

        Returns:
            Assigned variant ID or None
        """
        key = f"{experiment_id}:{user_id}"
        return self.variant_assignments.get(key)


class MetricAggregator:
    """Aggregator for experiment metrics."""

    def __init__(self, tracker: ExperimentTracker):
        """Initialize metric aggregator.

        Args:
            tracker: Experiment tracker
        """
        self.tracker = tracker
        LOGGER.info("Initialized MetricAggregator")

    def aggregate_metrics(
        self, experiment_id: str, metric_name: str
    ) -> dict[str, dict[str, float]]:
        """Aggregate metrics by variant.

        Args:
            experiment_id: Experiment identifier
            metric_name: Metric to aggregate

        Returns:
            Dictionary of variant_id -> {mean, std, count, etc.}
        """
        events = self.tracker.get_events(experiment_id, metric_name=metric_name)

        # Group by variant
        variant_values: dict[str, list[float]] = defaultdict(list)
        for event in events:
            variant_values[event.variant_id].append(event.value)

        # Aggregate
        results = {}
        for variant_id, values in variant_values.items():
            if not values:
                continue

            mean = sum(values) / len(values)
            variance = (
                sum((x - mean) ** 2 for x in values) / len(values)
                if len(values) > 1
                else 0
            )
            std = variance**0.5

            results[variant_id] = {
                "mean": mean,
                "std": std,
                "min": min(values),
                "max": max(values),
                "count": len(values),
                "sum": sum(values),
            }

        return results

    def calculate_conversion_rate(
        self, experiment_id: str, conversion_metric: str = "conversion"
    ) -> dict[str, float]:
        """Calculate conversion rates by variant.

        Args:
            experiment_id: Experiment identifier
            conversion_metric: Metric name for conversions

        Returns:
            Dictionary of variant_id -> conversion_rate
        """
        events = self.tracker.get_events(experiment_id)

        # Count unique users and conversions per variant
        variant_users: dict[str, set[str]] = defaultdict(set)
        variant_conversions: dict[str, set[str]] = defaultdict(set)

        for event in events:
            if event.user_id:
                variant_users[event.variant_id].add(event.user_id)

                if event.metric_name == conversion_metric and event.value > 0:
                    variant_conversions[event.variant_id].add(event.user_id)

        # Calculate rates
        rates = {}
        for variant_id in variant_users:
            total_users = len(variant_users[variant_id])
            converted_users = len(variant_conversions.get(variant_id, set()))
            rates[variant_id] = (
                converted_users / total_users if total_users > 0 else 0.0
            )

        return rates

    def get_time_series(
        self,
        experiment_id: str,
        variant_id: str,
        metric_name: str,
        window_hours: int = 24,
    ) -> list[tuple[datetime, float]]:
        """Get time-series data for a metric.

        Args:
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            metric_name: Metric name
            window_hours: Time window in hours

        Returns:
            List of (timestamp, value) tuples
        """
        events = self.tracker.get_events(
            experiment_id, variant_id=variant_id, metric_name=metric_name
        )

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        return [(e.timestamp, e.value) for e in events]


# Convenience function
def track_conversion(
    experiment_id: str,
    variant_id: str,
    user_id: str,
    converted: bool = True,
) -> MetricEvent:
    """Track a conversion event (convenience function).

    Args:
        experiment_id: Experiment identifier
        variant_id: Variant identifier
        user_id: User identifier
        converted: Whether user converted

    Returns:
        Created metric event
    """
    return MetricEvent(
        experiment_id=experiment_id,
        variant_id=variant_id,
        metric_name="conversion",
        value=1.0 if converted else 0.0,
        user_id=user_id,
    )
