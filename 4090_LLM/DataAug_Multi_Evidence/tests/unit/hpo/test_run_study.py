from __future__ import annotations

import optuna

from dataaug_multi_both.hpo.run_study import (
    StageSettings,
    run_stage,
    select_top_trials,
    split_budget,
)


def _constant_objective(value: float = 1.0):
    def _objective(trial: optuna.Trial, params, settings: StageSettings) -> float:  # type: ignore[override]
        return value

    return _objective


SIMPLE_SPACE = {
    "x": {"type": "float", "low": 0.0, "high": 1.0},
}


def test_plateau_stopper_triggers() -> None:
    settings = StageSettings(
        stage_name="test",
        search_space=SIMPLE_SPACE,
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=optuna.pruners.NopPruner(),
        n_trials=10,
        timeout=None,
        plateau_patience=2,
        epochs=1,
    )
    result = run_stage(settings, _constant_objective())
    # First trial improves, subsequent ones plateau -> stop quickly.
    assert len(result.completed_trials) <= 3


def test_select_top_trials_orders_by_value() -> None:
    settings = StageSettings(
        stage_name="rank",
        search_space=SIMPLE_SPACE,
        sampler=optuna.samplers.RandomSampler(seed=1),
        pruner=optuna.pruners.NopPruner(),
        n_trials=5,
        timeout=None,
        plateau_patience=5,
        epochs=1,
    )

    def objective(trial: optuna.Trial, params, stage: StageSettings) -> float:  # type: ignore[override]
        return float(trial.number)

    result = run_stage(settings, objective)
    top = select_top_trials(result.study.trials, k=2)
    assert [t.value for t in top] == sorted([t.value for t in top], reverse=True)


def test_split_budget_distribution() -> None:
    assert split_budget(10, 3) == [4, 3, 3]
    assert split_budget(2, 4) == [1, 1, 0, 0]
