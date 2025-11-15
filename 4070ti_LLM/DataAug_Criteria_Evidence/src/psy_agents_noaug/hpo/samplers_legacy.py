"""Optuna sampler factories."""

from __future__ import annotations

from typing import Literal

import optuna


def create_sampler(
    *,
    multi_objective: bool,
    seed: int,
    sampler: Literal["auto", "tpe", "nsga2"] = "auto",
) -> optuna.samplers.BaseSampler:
    """Return an Optuna sampler configured for the optimisation strategy."""

    if sampler == "nsga2" or multi_objective:
        return optuna.samplers.NSGAIISampler(seed=seed)

    if sampler == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
            multivariate=True,
            group=True,
            constant_liar=True,
        )

    # Auto mode picks NSGAII for multi-objective, otherwise TPE.
    return optuna.samplers.TPESampler(
        seed=seed,
        multivariate=True,
        group=True,
        constant_liar=True,
    )


