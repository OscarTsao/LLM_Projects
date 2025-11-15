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
            multivariate=False,
            group=False,
            constant_liar=False,
        )

    # Auto mode picks NSGAII for multi-objective, otherwise conservative TPE.
    return optuna.samplers.TPESampler(
        seed=seed,
        multivariate=False,
        group=False,
        constant_liar=False,
    )
