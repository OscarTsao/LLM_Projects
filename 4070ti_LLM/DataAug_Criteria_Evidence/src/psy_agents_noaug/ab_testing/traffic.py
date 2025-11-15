#!/usr/bin/env python
"""Traffic splitting and routing (Phase 21).

This module provides:
- Traffic splitting strategies
- Sticky session management
- Request routing
- Load balancing across variants
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any

LOGGER = logging.getLogger(__name__)


class SplitStrategy(str, Enum):
    """Traffic split strategies."""

    UNIFORM = "uniform"  # Equal split
    WEIGHTED = "weighted"  # Weighted by allocation
    STICKY = "sticky"  # Consistent assignment by user ID
    RANDOM = "random"  # Pure random


@dataclass
class TrafficAllocation:
    """Traffic allocation for a variant."""

    variant_id: str
    weight: float  # Weight (0.0 to 1.0)
    sticky: bool = False  # Use sticky sessions


class TrafficSplitter:
    """Traffic splitter for A/B testing."""

    def __init__(
        self,
        strategy: SplitStrategy = SplitStrategy.UNIFORM,
        seed: int | None = None,
    ):
        """Initialize traffic splitter.

        Args:
            strategy: Splitting strategy
            seed: Random seed for reproducibility
        """
        self.strategy = strategy
        self.random = random.Random(seed)
        self.allocations: dict[str, TrafficAllocation] = {}
        self.sticky_cache: dict[str, str] = {}
        LOGGER.info(f"Initialized TrafficSplitter with {strategy.value} strategy")

    def configure_allocation(
        self,
        experiment_id: str,
        allocations: list[TrafficAllocation],
    ) -> None:
        """Configure traffic allocation for an experiment.

        Args:
            experiment_id: Experiment identifier
            allocations: List of traffic allocations
        """
        # Normalize weights
        total_weight = sum(a.weight for a in allocations)
        for allocation in allocations:
            key = f"{experiment_id}:{allocation.variant_id}"
            self.allocations[key] = TrafficAllocation(
                variant_id=allocation.variant_id,
                weight=allocation.weight / total_weight,
                sticky=allocation.sticky,
            )

        LOGGER.info(f"Configured allocation for experiment {experiment_id}")

    def assign_variant(
        self,
        experiment_id: str,
        user_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str | None:
        """Assign a variant to a user.

        Args:
            experiment_id: Experiment identifier
            user_id: User identifier for sticky sessions
            context: Additional context for assignment

        Returns:
            Assigned variant ID or None
        """
        # Get allocations for experiment
        exp_allocations = [
            (k.split(":")[1], v)
            for k, v in self.allocations.items()
            if k.startswith(f"{experiment_id}:")
        ]

        if not exp_allocations:
            return None

        # Check sticky cache
        if user_id and self.strategy == SplitStrategy.STICKY:
            cache_key = f"{experiment_id}:{user_id}"
            if cache_key in self.sticky_cache:
                return self.sticky_cache[cache_key]

        # Assign variant based on strategy
        if self.strategy == SplitStrategy.UNIFORM:
            variant_id = self._uniform_split(exp_allocations)

        elif self.strategy == SplitStrategy.WEIGHTED:
            variant_id = self._weighted_split(exp_allocations)

        elif self.strategy == SplitStrategy.STICKY:
            variant_id = self._sticky_split(
                experiment_id, user_id or "anonymous", exp_allocations
            )

        else:  # RANDOM
            variant_id = self._random_split(exp_allocations)

        # Cache for sticky sessions
        if user_id and self.strategy == SplitStrategy.STICKY:
            cache_key = f"{experiment_id}:{user_id}"
            self.sticky_cache[cache_key] = variant_id

        return variant_id

    def _uniform_split(self, allocations: list[tuple[str, TrafficAllocation]]) -> str:
        """Uniform split across variants.

        Args:
            allocations: List of (variant_id, allocation) tuples

        Returns:
            Selected variant ID
        """
        return self.random.choice([variant_id for variant_id, _ in allocations])

    def _weighted_split(self, allocations: list[tuple[str, TrafficAllocation]]) -> str:
        """Weighted split based on allocation weights.

        Args:
            allocations: List of (variant_id, allocation) tuples

        Returns:
            Selected variant ID
        """
        # Create cumulative distribution
        cumsum = 0.0
        thresholds = []
        for variant_id, allocation in allocations:
            cumsum += allocation.weight
            thresholds.append((cumsum, variant_id))

        # Random selection
        rand = self.random.random()
        for threshold, variant_id in thresholds:
            if rand <= threshold:
                return variant_id

        # Fallback to last variant
        return allocations[-1][0]

    def _sticky_split(
        self,
        experiment_id: str,
        user_id: str,
        allocations: list[tuple[str, TrafficAllocation]],
    ) -> str:
        """Sticky split - consistent assignment per user.

        Args:
            experiment_id: Experiment ID
            user_id: User ID
            allocations: List of (variant_id, allocation) tuples

        Returns:
            Selected variant ID
        """
        # Hash user ID to get consistent assignment
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        normalized = (hash_value % 10000) / 10000.0

        # Use weighted distribution with hash
        cumsum = 0.0
        for variant_id, allocation in allocations:
            cumsum += allocation.weight
            if normalized <= cumsum:
                return variant_id

        return allocations[-1][0]

    def _random_split(self, allocations: list[tuple[str, TrafficAllocation]]) -> str:
        """Pure random split.

        Args:
            allocations: List of (variant_id, allocation) tuples

        Returns:
            Selected variant ID
        """
        weights = [allocation.weight for _, allocation in allocations]
        variant_ids = [variant_id for variant_id, _ in allocations]
        return self.random.choices(variant_ids, weights=weights, k=1)[0]

    def get_allocation_stats(self, experiment_id: str) -> dict[str, float]:
        """Get allocation statistics for an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Dictionary of variant_id -> weight
        """
        stats = {}
        for key, allocation in self.allocations.items():
            if key.startswith(f"{experiment_id}:"):
                variant_id = key.split(":")[1]
                stats[variant_id] = allocation.weight

        return stats

    def clear_sticky_cache(self, experiment_id: str | None = None) -> None:
        """Clear sticky session cache.

        Args:
            experiment_id: Clear cache for specific experiment (or all if None)
        """
        if experiment_id:
            keys_to_remove = [
                k for k in self.sticky_cache.keys() if k.startswith(f"{experiment_id}:")
            ]
            for key in keys_to_remove:
                del self.sticky_cache[key]
        else:
            self.sticky_cache.clear()

        LOGGER.info(f"Cleared sticky cache for {experiment_id or 'all experiments'}")


# Convenience function
def split_traffic(
    experiment_id: str,
    variants: list[str],
    user_id: str | None = None,
    strategy: SplitStrategy = SplitStrategy.UNIFORM,
) -> str:
    """Split traffic across variants (convenience function).

    Args:
        experiment_id: Experiment identifier
        variants: List of variant IDs
        user_id: User identifier for sticky sessions
        strategy: Splitting strategy

    Returns:
        Selected variant ID
    """
    splitter = TrafficSplitter(strategy=strategy)

    # Configure equal allocation
    allocations = [
        TrafficAllocation(variant_id=v, weight=1.0 / len(variants)) for v in variants
    ]
    splitter.configure_allocation(experiment_id, allocations)

    return splitter.assign_variant(experiment_id, user_id) or variants[0]
