from __future__ import annotations

import optuna
from optuna.pruners import HyperbandPruner, PercentilePruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState


def build_sampler(stage: str) -> optuna.samplers.BaseSampler:
    if stage == "A":
        return TPESampler(multivariate=True, group=True, n_startup_trials=60, n_ei_candidates=128, seed=42)
    return TPESampler(multivariate=True, group=True, n_startup_trials=30, n_ei_candidates=128, seed=123)

def build_pruner(stage: str, max_epochs: int) -> optuna.pruners.BasePruner:
    if stage == "A":
        return HyperbandPruner(min_resource=1, max_resource=max_epochs, reduction_factor=3)
    return PercentilePruner(25.0, n_startup_trials=10, n_warmup_steps=2)

class PlateauStopper:
    """Stop study if best value hasn't improved for `patience_trials`."""
    def __init__(self, patience_trials: int = 120):
        self.patience = patience_trials
    def __call__(self, study: optuna.Study, trial: optuna.FrozenTrial):
        # If no trial has completed yet, Optuna raises when accessing best_value.
        # In that case, do nothing and wait until a first COMPLETE trial exists.
        try:
            current_best = float(study.best_value)  # may raise if no COMPLETE trials
        except Exception:
            return

        best = study.user_attrs.get("best_value", None)
        last_improve_at = study.user_attrs.get("last_improve_at", None)

        # Initialize tracking once a first COMPLETE trial exists
        if best is None or last_improve_at is None:
            study.set_user_attr("best_value", current_best)
            study.set_user_attr("last_improve_at", trial.number)
            return

        # Update when improvement happens
        if current_best > float(best) + 1e-12:
            study.set_user_attr("best_value", current_best)
            study.set_user_attr("last_improve_at", trial.number)
            return

        # Otherwise, check for plateau
        try:
            stalled = int(trial.number) - int(last_improve_at)
        except Exception:
            # Fallback: if stored attr becomes invalid, reset tracking.
            study.set_user_attr("best_value", current_best)
            study.set_user_attr("last_improve_at", trial.number)
            return

        if stalled >= self.patience:
            study.stop()
