from __future__ import annotations

import optuna

from src.dataaug_multi_both.hpo.samplers_pruners import PlateauStopper, build_pruner, build_sampler


def test_build_sampler_and_pruner_configs():
    sampler_a = build_sampler("A")
    sampler_b = build_sampler("B")
    assert sampler_a != sampler_b
    pruner_a = build_pruner("A", max_epochs=5)
    pruner_b = build_pruner("B", max_epochs=40)
    assert pruner_a != pruner_b


def test_plateau_stopper_triggers_after_patience():
    stopper = PlateauStopper(patience_trials=2)
    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        trial.report(0.5, step=0)
        return 0.5

    study.optimize(objective, n_trials=5, callbacks=[stopper])
    assert len(study.trials) < 5
