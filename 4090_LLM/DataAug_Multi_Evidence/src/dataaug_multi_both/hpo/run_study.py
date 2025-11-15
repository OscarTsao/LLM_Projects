"""Utilities to execute multi-stage Optuna studies.

This module wires together reusable helpers that the two-stage driver can call.
The functions are objective-agnostic so we can plug in either the real training
loop or deterministic simulation logic for tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import optuna


ScoreFn = Callable[[optuna.Trial, Dict[str, Any], "StageSettings"], float]


@dataclass(frozen=True)
class StageSettings:
    """Configuration for one optimization stage."""

    stage_name: str
    search_space: Dict[str, Dict[str, Any]]
    sampler: optuna.samplers.BaseSampler
    pruner: optuna.pruners.BasePruner
    n_trials: int
    timeout: Optional[int]
    plateau_patience: int
    epochs: int
    study_name: Optional[str] = None
    storage: Optional[str] = None
    direction: str = "maximize"
    frozen_params: Optional[Dict[str, Any]] = None
    progress_callback: Optional[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]] = None


@dataclass(frozen=True)
class StageResult:
    """Result bundle returned by :func:`run_stage`."""

    settings: StageSettings
    study: optuna.Study
    best_trial: optuna.trial.FrozenTrial
    completed_trials: List[optuna.trial.FrozenTrial]


class PlateauStopper:
    """Study callback that stops optimization after a plateau."""

    def __init__(self, patience: int, epsilon: float = 1e-12) -> None:
        self.patience = patience
        self.epsilon = epsilon
        self.best_value: Optional[float] = None
        self._stalled = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        value = trial.value
        if value is None:
            return

        if self.best_value is None or value > self.best_value + self.epsilon:
            self.best_value = value
            self._stalled = 0
        else:
            self._stalled += 1
            if self._stalled >= self.patience:
                study.stop()


def suggest_parameters(
    trial: optuna.Trial,
    search_space: Dict[str, Dict[str, Any]],
    frozen: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sample a flat set of parameters from ``search_space``."""

    params: Dict[str, Any] = {}
    if frozen:
        params.update(frozen)

    for name, config in search_space.items():
        if frozen and name in frozen:
            continue

        param_type = config.get("type")
        if param_type == "categorical":
            params[name] = trial.suggest_categorical(name, config["choices"])  # type: ignore[arg-type]
        elif param_type in {"float", "loguniform"}:
            low = float(config.get("low", config.get("min")))
            high = float(config.get("high", config.get("max")))
            log = param_type == "loguniform" or bool(config.get("log", False))
            params[name] = trial.suggest_float(name, low, high, log=log)
        elif param_type == "int":
            low = int(config.get("low", config.get("min")))
            high = int(config.get("high", config.get("max")))
            step = config.get("step")
            log = bool(config.get("log", False))
            if step is not None:
                params[name] = trial.suggest_int(name, low, high, step=int(step), log=log)
            else:
                params[name] = trial.suggest_int(name, low, high, log=log)
        else:  # pragma: no cover - guard for future extensions
            raise ValueError(f"Unsupported parameter type: {param_type} for {name}")

    return params


def run_stage(settings: StageSettings, objective: ScoreFn) -> StageResult:
    """Execute one Optuna stage and return the best result."""

    study = optuna.create_study(
        direction=settings.direction,
        sampler=settings.sampler,
        pruner=settings.pruner,
        storage=settings.storage,
        study_name=settings.study_name,
        load_if_exists=False,
    )

    stopper = PlateauStopper(settings.plateau_patience)
    callbacks: List[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]] = [stopper]
    if settings.progress_callback is not None:
        callbacks.append(settings.progress_callback)

    def _objective(trial: optuna.Trial) -> float:
        params = suggest_parameters(trial, settings.search_space, settings.frozen_params)
        trial.set_user_attr("epochs", settings.epochs)
        return objective(trial, params, settings)

    study.optimize(
        _objective,
        n_trials=settings.n_trials,
        timeout=settings.timeout,
        callbacks=callbacks,
    )

    completed = [
        t for t in study.get_trials(deepcopy=False)
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]

    if not completed:
        raise RuntimeError(f"Stage '{settings.stage_name}' produced no completed trials.")

    best = max(completed, key=lambda t: t.value)
    return StageResult(settings=settings, study=study, best_trial=best, completed_trials=completed)


def select_top_trials(
    trials: Iterable[optuna.trial.FrozenTrial],
    k: int,
) -> List[optuna.trial.FrozenTrial]:
    """Return the top-``k`` completed trials sorted by score."""

    sorted_trials = sorted(
        (t for t in trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None),
        key=lambda t: t.value,
        reverse=True,
    )
    return sorted_trials[:k]


def split_budget(total: int, parts: int) -> List[int]:
    """Split ``total`` into ``parts`` buckets (difference at most one)."""

    if parts <= 0:
        return []
    base = total // parts
    remainder = total % parts
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


__all__ = [
    "PlateauStopper",
    "StageResult",
    "StageSettings",
    "run_stage",
    "select_top_trials",
    "split_budget",
    "suggest_parameters",
]
