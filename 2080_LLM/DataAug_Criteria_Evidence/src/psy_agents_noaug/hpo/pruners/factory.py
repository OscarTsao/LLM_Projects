"""Pruner factory functions with smart defaults and best practices.

This module provides easy-to-use factory functions for creating pruners
with sensible defaults based on problem characteristics and resource budgets.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from optuna.pruners import (
    BasePruner,
    HyperbandPruner,
    MedianPruner,
    NopPruner,
    PercentilePruner,
    SuccessiveHalvingPruner,
)

LOGGER = logging.getLogger(__name__)


def create_advanced_pruner(
    pruner_type: Literal[
        "hyperband", "successive_halving", "median", "percentile", "none"
    ] = "hyperband",
    *,
    min_resource: int = 1,
    max_resource: int = 10,
    reduction_factor: int = 3,
    n_startup_trials: int = 5,
    n_warmup_steps: int = 0,
    interval_steps: int = 1,
    percentile: float = 25.0,
    **kwargs,
) -> BasePruner:
    """Create an advanced pruner with smart defaults.

    Args:
        pruner_type: Type of pruner to create
        min_resource: Minimum resource allocation (e.g., epochs)
        max_resource: Maximum resource allocation
        reduction_factor: Halving rate for successive halving
        n_startup_trials: Number of trials before pruning starts
        n_warmup_steps: Steps before pruning can occur
        interval_steps: Interval between pruning checks
        percentile: Percentile threshold for pruning (percentile pruner)
        **kwargs: Additional pruner-specific arguments

    Returns:
        Configured pruner instance

    Raises:
        ValueError: If pruner_type is unknown
    """
    if pruner_type == "hyperband":
        return create_hyperband_pruner(
            min_resource=min_resource,
            max_resource=max_resource,
            reduction_factor=reduction_factor,
            **kwargs,
        )
    if pruner_type == "successive_halving":
        return SuccessiveHalvingPruner(
            min_resource=min_resource,
            reduction_factor=reduction_factor,
            min_early_stopping_rate=0,
        )
    if pruner_type == "median":
        return create_median_pruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
            interval_steps=interval_steps,
        )
    if pruner_type == "percentile":
        return create_percentile_pruner(
            percentile=percentile,
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
            interval_steps=interval_steps,
        )
    if pruner_type == "none":
        return NopPruner()
    raise ValueError(f"Unknown pruner type: {pruner_type}")


def create_hyperband_pruner(
    min_resource: int = 1,
    max_resource: int = 10,
    reduction_factor: int = 3,
    bootstrap_count: int = 0,
) -> HyperbandPruner:
    """Create Hyperband pruner with sensible defaults.

    Hyperband is an aggressive pruning algorithm that allocates resources
    using successive halving across multiple brackets. It's particularly
    effective when:
    - You have many hyperparameters to tune
    - Trials are expensive
    - You want fast convergence

    Algorithm:
    - Creates multiple "brackets" with different resource allocations
    - Uses successive halving within each bracket
    - Prunes aggressively early, keeps promising trials longer

    Best for:
    - Large search spaces (>5 hyperparameters)
    - Expensive trials (>5 min/trial)
    - When you need results quickly

    Args:
        min_resource: Minimum resource (e.g., 1 epoch)
        max_resource: Maximum resource (e.g., 27 epochs for 3^3)
        reduction_factor: Halving rate (typically 3 or 4)
        bootstrap_count: Number of initial trials to complete fully

    Returns:
        Configured HyperbandPruner

    Example:
        >>> pruner = create_hyperband_pruner(
        ...     min_resource=1,
        ...     max_resource=27,  # 3^3
        ...     reduction_factor=3,
        ... )
    """
    LOGGER.info(
        "Creating Hyperband pruner: min=%d, max=%d, reduction=%d",
        min_resource,
        max_resource,
        reduction_factor,
    )

    return HyperbandPruner(
        min_resource=min_resource,
        max_resource=max_resource,
        reduction_factor=reduction_factor,
        bootstrap_count=bootstrap_count,
    )


def create_median_pruner(
    n_startup_trials: int = 5,
    n_warmup_steps: int = 0,
    interval_steps: int = 1,
) -> MedianPruner:
    """Create Median pruner with sensible defaults.

    Median pruner stops trials whose intermediate value is worse than
    the median of all completed trials at the same step.

    Best for:
    - Medium search spaces (3-5 hyperparameters)
    - When you want moderate pruning (not too aggressive)
    - Stable optimization landscapes

    Args:
        n_startup_trials: Number of trials to complete before pruning starts
        n_warmup_steps: Steps to wait before pruning can occur
        interval_steps: Check for pruning every N steps

    Returns:
        Configured MedianPruner

    Example:
        >>> pruner = create_median_pruner(
        ...     n_startup_trials=10,
        ...     n_warmup_steps=2,
        ...     interval_steps=1,
        ... )
    """
    LOGGER.info(
        "Creating Median pruner: startup=%d, warmup=%d, interval=%d",
        n_startup_trials,
        n_warmup_steps,
        interval_steps,
    )

    return MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps,
        interval_steps=interval_steps,
    )


def create_percentile_pruner(
    percentile: float = 25.0,
    n_startup_trials: int = 5,
    n_warmup_steps: int = 0,
    interval_steps: int = 1,
) -> PercentilePruner:
    """Create Percentile pruner with sensible defaults.

    Percentile pruner stops trials whose intermediate value is worse than
    the percentile of all completed trials at the same step.

    More aggressive than median pruner (percentile < 50).

    Best for:
    - When you want aggressive pruning (percentile=10-25)
    - Large trial budgets (want to focus on top performers)
    - Expensive trials

    Args:
        percentile: Percentile threshold (0-100, lower = more aggressive)
        n_startup_trials: Number of trials to complete before pruning
        n_warmup_steps: Steps to wait before pruning can occur
        interval_steps: Check for pruning every N steps

    Returns:
        Configured PercentilePruner

    Example:
        >>> pruner = create_percentile_pruner(
        ...     percentile=10.0,  # Keep only top 10%
        ...     n_startup_trials=20,
        ... )
    """
    LOGGER.info(
        "Creating Percentile pruner: percentile=%.1f, startup=%d",
        percentile,
        n_startup_trials,
    )

    return PercentilePruner(
        percentile=percentile,
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps,
        interval_steps=interval_steps,
    )


def get_pruner_info(pruner: BasePruner) -> dict[str, Any]:
    """Get information about a pruner.

    Args:
        pruner: Pruner instance

    Returns:
        Dictionary with pruner information
    """
    info = {
        "type": type(pruner).__name__,
        "aggressive": isinstance(pruner, (HyperbandPruner, PercentilePruner)),
        "moderate": isinstance(pruner, (MedianPruner, SuccessiveHalvingPruner)),
        "disabled": isinstance(pruner, NopPruner),
    }

    # Add type-specific info
    if isinstance(pruner, HyperbandPruner):
        info["min_resource"] = pruner._min_resource
        info["max_resource"] = pruner._max_resource
        info["reduction_factor"] = pruner._reduction_factor
    elif isinstance(pruner, MedianPruner):
        info["n_startup_trials"] = pruner._n_startup_trials
        info["n_warmup_steps"] = pruner._n_warmup_steps
    elif isinstance(pruner, PercentilePruner):
        info["percentile"] = pruner._percentile
        info["n_startup_trials"] = pruner._n_startup_trials

    return info
