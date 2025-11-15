from __future__ import annotations

import time
from typing import Optional

import optuna

from src.hpo.optuna_utils import BestState, estimate_eta, format_progress, update_best_state


class PlateauStopper:
    def __init__(self, patience_trials: int = 120, direction: str = "maximize") -> None:
        self.patience = int(patience_trials)
        self.direction = direction
        self.state = BestState()

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:  # noqa: D401
        value = trial.value
        if value is None:
            return
        improved = update_best_state(self.state, float(value), trial.number, self.direction)
        if improved:
            return
        if trial.number - self.state.last_improved_idx >= self.patience:
            study.stop()


class TelemetryCallback:
    def __init__(self, total_trials: int, direction: str = "maximize") -> None:
        self.total = int(total_trials)
        self.direction = direction
        self.started = 0
        self.completed_or_pruned = 0
        self.start_time = time.time()
        self.best = BestState()

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        # Called at trial end
        self.completed_or_pruned += 1
        if trial.value is not None:
            update_best_state(self.best, float(trial.value), trial.number, self.direction)
        eta = estimate_eta(self.start_time, self.completed_or_pruned, self.total)
        msg = format_progress(trial.number + 1, self.total, self.completed_or_pruned, max(self.completed_or_pruned, 1), time.time() - self.start_time, eta, self.best.best_value, self.best.best_trial_id)
        print(msg)

