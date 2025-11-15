"""Adaptive resource allocation for efficient HPO.

This module provides utilities for:
- Estimating trial budgets based on time constraints
- Dynamic resource allocation (epochs, batch sizes)
- Budget optimization for successive halving
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class ResourceBudget:
    """Resource budget estimation for HPO."""

    min_resource: int
    max_resource: int
    n_trials: int
    reduction_factor: int
    estimated_total_trials: int
    estimated_savings: float  # Fraction of resources saved vs no pruning


def estimate_trial_budget(
    *,
    total_time_hours: float,
    avg_trial_minutes: float,
    min_resource: int = 1,
    max_resource: int = 27,
    reduction_factor: int = 3,
    pruning_efficiency: float = 0.5,
) -> ResourceBudget:
    """Estimate optimal trial budget for Hyperband/successive halving.

    This function helps you plan your HPO run by estimating how many
    trials you can afford given your time and resource constraints.

    Args:
        total_time_hours: Total available time for HPO
        avg_trial_minutes: Average time per trial at max_resource
        min_resource: Minimum resource (e.g., 1 epoch)
        max_resource: Maximum resource (e.g., 27 epochs)
        reduction_factor: Halving rate (typically 3 or 4)
        pruning_efficiency: Expected fraction of trials pruned early (0.3-0.7)

    Returns:
        Resource budget with trial count and savings estimate

    Example:
        >>> # 8 hours available, 15 min per trial at max epochs
        >>> budget = estimate_trial_budget(
        ...     total_time_hours=8.0,
        ...     avg_trial_minutes=15.0,
        ...     max_resource=27,
        ... )
        >>> print(f"Can run ~{budget.n_trials} trials")
        >>> print(f"Estimated savings: {budget.estimated_savings:.1%}")
    """
    # Convert time to minutes
    total_time_minutes = total_time_hours * 60

    # Without pruning: max trials = total_time / trial_time
    max_trials_no_pruning = int(total_time_minutes / avg_trial_minutes)

    # With Hyperband: trials are distributed across brackets
    # Each bracket has different resource allocations
    # Average resource per trial is approximately max_resource / reduction_factor
    avg_resource_with_pruning = max_resource / (reduction_factor**pruning_efficiency)
    avg_trial_minutes_with_pruning = avg_trial_minutes * (
        avg_resource_with_pruning / max_resource
    )

    # Estimated trials with pruning
    estimated_trials = int(total_time_minutes / avg_trial_minutes_with_pruning)

    # Resource savings
    total_resource_no_pruning = max_trials_no_pruning * max_resource
    total_resource_with_pruning = estimated_trials * avg_resource_with_pruning
    savings = 1.0 - (total_resource_with_pruning / total_resource_no_pruning)

    LOGGER.info(
        "Budget estimation: %.1fh available, %.1f min/trial @ max_resource=%d",
        total_time_hours,
        avg_trial_minutes,
        max_resource,
    )
    LOGGER.info(
        "  Without pruning: ~%d trials (%.1f total hours)",
        max_trials_no_pruning,
        max_trials_no_pruning * avg_trial_minutes / 60,
    )
    LOGGER.info(
        "  With pruning: ~%d trials (%.1f avg resource, %.1f%% savings)",
        estimated_trials,
        avg_resource_with_pruning,
        savings * 100,
    )

    return ResourceBudget(
        min_resource=min_resource,
        max_resource=max_resource,
        n_trials=estimated_trials,
        reduction_factor=reduction_factor,
        estimated_total_trials=estimated_trials,
        estimated_savings=savings,
    )


class AdaptiveResourceAllocator:
    """Dynamically allocate resources based on trial performance.

    This class implements adaptive scheduling that adjusts resource
    allocation (e.g., epochs) based on early trial performance.

    Strategy:
    - Start with min_resource for all trials
    - Gradually increase resources for promising trials
    - Use early stopping for unpromising trials
    """

    def __init__(
        self,
        min_resource: int = 1,
        max_resource: int = 27,
        reduction_factor: int = 3,
        performance_threshold: float = 0.7,
    ):
        """Initialize adaptive resource allocator.

        Args:
            min_resource: Minimum resource allocation
            max_resource: Maximum resource allocation
            reduction_factor: Multiplicative increase factor
            performance_threshold: Percentile threshold for promotion (0-1)
        """
        self.min_resource = min_resource
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor
        self.performance_threshold = performance_threshold

        # Track trial history
        self._trial_resources: dict[int, int] = {}
        self._trial_scores: dict[int, list[float]] = {}

        LOGGER.info(
            "Adaptive allocator: min=%d, max=%d, factor=%d, threshold=%.2f",
            min_resource,
            max_resource,
            reduction_factor,
            performance_threshold,
        )

    def get_initial_resource(self, trial_number: int) -> int:
        """Get initial resource allocation for a new trial.

        Args:
            trial_number: Trial number

        Returns:
            Initial resource (typically min_resource)
        """
        self._trial_resources[trial_number] = self.min_resource
        self._trial_scores[trial_number] = []
        return self.min_resource

    def should_continue(
        self,
        trial_number: int,
        current_score: float,
        all_scores: list[float],
    ) -> tuple[bool, int]:
        """Decide whether to continue a trial and with how many resources.

        Args:
            trial_number: Trial number
            current_score: Current performance score (higher is better)
            all_scores: All scores from completed trials at this resource level

        Returns:
            (should_continue, next_resource) tuple
        """
        current_resource = self._trial_resources.get(trial_number, self.min_resource)
        self._trial_scores[trial_number].append(current_score)

        # Reached max resource - stop here
        if current_resource >= self.max_resource:
            LOGGER.debug(
                "Trial #%d reached max_resource=%d, stopping",
                trial_number,
                self.max_resource,
            )
            return False, current_resource

        # Not enough data yet - continue with same resource
        if len(all_scores) < 3:
            LOGGER.debug("Trial #%d: not enough data, continuing", trial_number)
            return True, current_resource

        # Compute performance percentile
        sorted_scores = sorted(all_scores, reverse=True)
        rank = sum(1 for s in sorted_scores if s > current_score)
        percentile = 1.0 - (rank / len(sorted_scores))

        LOGGER.debug(
            "Trial #%d: score=%.4f, percentile=%.2f, resource=%d",
            trial_number,
            current_score,
            percentile,
            current_resource,
        )

        # Decision: promote if in top percentile
        if percentile >= self.performance_threshold:
            next_resource = min(
                current_resource * self.reduction_factor, self.max_resource
            )
            self._trial_resources[trial_number] = next_resource
            LOGGER.info(
                "Trial #%d promoted: %d → %d epochs (percentile=%.2f)",
                trial_number,
                current_resource,
                next_resource,
                percentile,
            )
            return True, next_resource
        LOGGER.info(
            "Trial #%d pruned: score=%.4f < threshold (percentile=%.2f)",
            trial_number,
            current_score,
            percentile,
        )
        return False, current_resource

    def get_resource_schedule(self, n_rungs: int | None = None) -> list[int]:
        """Get the full resource schedule (rungs) for successive halving.

        Args:
            n_rungs: Number of rungs (auto-computed if None)

        Returns:
            List of resource values [min_resource, ..., max_resource]

        Example:
            >>> allocator = AdaptiveResourceAllocator(
            ...     min_resource=1,
            ...     max_resource=27,
            ...     reduction_factor=3,
            ... )
            >>> schedule = allocator.get_resource_schedule()
            >>> print(schedule)  # [1, 3, 9, 27]
        """
        if n_rungs is None:
            # Auto-compute rungs
            n_rungs = (
                int(
                    math.log(self.max_resource / self.min_resource)
                    / math.log(self.reduction_factor)
                )
                + 1
            )

        schedule = []
        current = self.min_resource

        for _ in range(n_rungs):
            schedule.append(current)
            current = min(current * self.reduction_factor, self.max_resource)

        # Ensure max_resource is included
        if schedule[-1] < self.max_resource:
            schedule.append(self.max_resource)

        LOGGER.info("Resource schedule: %s", schedule)
        return schedule

    def estimate_trial_distribution(self, n_initial_trials: int) -> dict[int, int]:
        """Estimate how many trials will reach each resource level.

        Uses geometric decay based on promotion threshold.

        Args:
            n_initial_trials: Number of trials starting at min_resource

        Returns:
            Dictionary {resource: n_trials} for each rung

        Example:
            >>> allocator = AdaptiveResourceAllocator(
            ...     performance_threshold=0.7,  # Top 30% advance
            ... )
            >>> dist = allocator.estimate_trial_distribution(100)
            >>> print(dist)  # {1: 100, 3: 30, 9: 9, 27: 3}
        """
        schedule = self.get_resource_schedule()
        distribution = {}

        current_trials = n_initial_trials
        for resource in schedule:
            distribution[resource] = current_trials
            # Next rung gets only top trials
            current_trials = int(current_trials * (1.0 - self.performance_threshold))

        LOGGER.info("Trial distribution estimate:")
        for resource, n_trials in distribution.items():
            LOGGER.info("  Resource %d: %d trials", resource, n_trials)

        return distribution

    def get_statistics(self) -> dict[str, Any]:
        """Get allocator statistics.

        Returns:
            Dictionary with allocation statistics
        """
        if not self._trial_resources:
            return {
                "n_trials": 0,
                "avg_resource": 0.0,
                "max_resource_reached": 0,
            }

        resources = list(self._trial_resources.values())
        return {
            "n_trials": len(resources),
            "avg_resource": sum(resources) / len(resources),
            "max_resource_reached": max(resources),
            "min_resource_used": min(resources),
            "resource_distribution": {
                r: sum(1 for x in resources if x == r) for r in set(resources)
            },
        }
