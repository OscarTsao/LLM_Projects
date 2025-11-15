"""Optuna pruner helpers."""

from __future__ import annotations

from typing import Literal

import optuna


def create_pruner(
    strategy: Literal["asha", "median", "none"],
    *,
    min_resource: int,
    max_resource: int,
    reduction_factor: int = 3,
    n_startup_trials: int = 5,
) -> optuna.pruners.BasePruner | None:
    """Instantiate a pruner by name."""

    if strategy == "none":
        return None

    if strategy == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=max(1, min_resource - 1),
            interval_steps=1,
        )

    # Default to ASHA (a SuccessiveHalving variant)
    return optuna.pruners.SuccessiveHalvingPruner(
        min_resource=min_resource,
        reduction_factor=max(2, reduction_factor),
        min_early_stopping_rate=0,
    )


def stage_pruner(stage: str, epochs: int) -> optuna.pruners.BasePruner:
    """Return a recommended pruner for multi-stage HPO."""

    stage = stage.upper()
    if stage == "S0":  # aggressive pruning for coarse search
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=4,
            min_early_stopping_rate=0,
        )
    if stage == "S1":
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=2,
            reduction_factor=3,
            min_early_stopping_rate=0,
        )
    if stage == "S2":
        return optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=max(1, epochs // 3),
            interval_steps=1,
        )
    return optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=max(1, epochs // 4),
        interval_steps=1,
    )

