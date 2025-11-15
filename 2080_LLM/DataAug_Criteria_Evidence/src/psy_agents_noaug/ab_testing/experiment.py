#!/usr/bin/env python
"""Experiment configuration and management (Phase 21).

This module provides:
- Experiment definition
- Variant configuration
- Experiment lifecycle management
- Success metrics definition
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

LOGGER = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    """Experiment status."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class Variant:
    """Experiment variant."""

    id: str
    name: str
    description: str
    allocation: float  # Traffic allocation (0.0 to 1.0)
    model_version: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    name: str
    description: str
    hypothesis: str
    primary_metric: str  # Metric to optimize
    secondary_metrics: list[str] = field(default_factory=list)
    minimum_sample_size: int = 1000
    significance_level: float = 0.05  # Alpha
    minimum_detectable_effect: float = 0.05  # Minimum effect size
    max_duration_days: int = 30


class Experiment:
    """A/B testing experiment."""

    def __init__(
        self,
        id: str,
        config: ExperimentConfig,
        variants: list[Variant],
    ):
        """Initialize experiment.

        Args:
            id: Experiment identifier
            config: Experiment configuration
            variants: List of variants
        """
        self.id = id
        self.config = config
        self.variants = {v.id: v for v in variants}
        self.status = ExperimentStatus.DRAFT
        self.created_at = datetime.now()
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None

        # Metrics storage
        self.metrics: dict[str, dict[str, list[float]]] = {v.id: {} for v in variants}

        LOGGER.info(f"Created experiment {id} with {len(variants)} variants")

    def start(self) -> None:
        """Start the experiment."""
        if self.status != ExperimentStatus.DRAFT:
            msg = f"Cannot start experiment in {self.status.value} status"
            raise ValueError(msg)

        # Validate allocations sum to 1.0
        total_allocation = sum(v.allocation for v in self.variants.values())
        if abs(total_allocation - 1.0) > 0.01:
            msg = f"Allocations sum to {total_allocation}, must sum to 1.0"
            raise ValueError(msg)

        self.status = ExperimentStatus.RUNNING
        self.started_at = datetime.now()
        LOGGER.info(f"Started experiment {self.id}")

    def pause(self) -> None:
        """Pause the experiment."""
        if self.status != ExperimentStatus.RUNNING:
            msg = f"Cannot pause experiment in {self.status.value} status"
            raise ValueError(msg)

        self.status = ExperimentStatus.PAUSED
        LOGGER.info(f"Paused experiment {self.id}")

    def resume(self) -> None:
        """Resume the experiment."""
        if self.status != ExperimentStatus.PAUSED:
            msg = f"Cannot resume experiment in {self.status.value} status"
            raise ValueError(msg)

        self.status = ExperimentStatus.RUNNING
        LOGGER.info(f"Resumed experiment {self.id}")

    def complete(self, winner: str | None = None) -> None:
        """Complete the experiment.

        Args:
            winner: Winning variant ID (if any)
        """
        if self.status not in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            msg = f"Cannot complete experiment in {self.status.value} status"
            raise ValueError(msg)

        self.status = ExperimentStatus.COMPLETED
        self.ended_at = datetime.now()

        if winner:
            LOGGER.info(f"Completed experiment {self.id} with winner: {winner}")
        else:
            LOGGER.info(f"Completed experiment {self.id} with no winner")

    def record_metric(self, variant_id: str, metric_name: str, value: float) -> None:
        """Record a metric value for a variant.

        Args:
            variant_id: Variant identifier
            metric_name: Metric name
            value: Metric value
        """
        if variant_id not in self.variants:
            msg = f"Unknown variant: {variant_id}"
            raise ValueError(msg)

        if metric_name not in self.metrics[variant_id]:
            self.metrics[variant_id][metric_name] = []

        self.metrics[variant_id][metric_name].append(value)

    def get_variant_metrics(self, variant_id: str) -> dict[str, list[float]]:
        """Get all metrics for a variant.

        Args:
            variant_id: Variant identifier

        Returns:
            Dictionary of metric_name -> values
        """
        return self.metrics.get(variant_id, {})

    def get_summary(self) -> dict[str, Any]:
        """Get experiment summary.

        Returns:
            Summary dictionary
        """
        return {
            "id": self.id,
            "name": self.config.name,
            "status": self.status.value,
            "variants": [
                {"id": v.id, "name": v.name, "allocation": v.allocation}
                for v in self.variants.values()
            ],
            "primary_metric": self.config.primary_metric,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
        }


class ExperimentManager:
    """Manager for experiments."""

    def __init__(self):
        """Initialize experiment manager."""
        self.experiments: dict[str, Experiment] = {}
        LOGGER.info("Initialized ExperimentManager")

    def create_experiment(
        self,
        id: str,
        config: ExperimentConfig,
        variants: list[Variant],
    ) -> Experiment:
        """Create a new experiment.

        Args:
            id: Experiment identifier
            config: Experiment configuration
            variants: List of variants

        Returns:
            Created experiment
        """
        if id in self.experiments:
            msg = f"Experiment {id} already exists"
            raise ValueError(msg)

        experiment = Experiment(id, config, variants)
        self.experiments[id] = experiment

        return experiment

    def get_experiment(self, id: str) -> Experiment | None:
        """Get an experiment.

        Args:
            id: Experiment identifier

        Returns:
            Experiment if found
        """
        return self.experiments.get(id)

    def list_experiments(
        self, status: ExperimentStatus | None = None
    ) -> list[Experiment]:
        """List experiments.

        Args:
            status: Filter by status (optional)

        Returns:
            List of experiments
        """
        experiments = list(self.experiments.values())

        if status:
            experiments = [e for e in experiments if e.status == status]

        return sorted(experiments, key=lambda e: e.created_at, reverse=True)

    def delete_experiment(self, id: str) -> bool:
        """Delete an experiment.

        Args:
            id: Experiment identifier

        Returns:
            True if deleted
        """
        if id in self.experiments:
            del self.experiments[id]
            LOGGER.info(f"Deleted experiment {id}")
            return True

        return False


# Convenience function
def create_experiment(
    name: str,
    variants: list[str],
    primary_metric: str = "accuracy",
) -> Experiment:
    """Create an experiment (convenience function).

    Args:
        name: Experiment name
        variants: List of variant names
        primary_metric: Primary metric to optimize

    Returns:
        Created experiment
    """
    config = ExperimentConfig(
        name=name,
        description=f"A/B test for {name}",
        hypothesis=f"Test {len(variants)} variants",
        primary_metric=primary_metric,
    )

    variant_objs = [
        Variant(
            id=f"variant-{i}",
            name=v,
            description=f"Variant {v}",
            allocation=1.0 / len(variants),
        )
        for i, v in enumerate(variants)
    ]

    return Experiment(id=f"exp-{name}", config=config, variants=variant_objs)
