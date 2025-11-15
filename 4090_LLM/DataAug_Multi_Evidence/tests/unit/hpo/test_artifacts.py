from __future__ import annotations

import optuna
from pathlib import Path

from dataaug_multi_both.hpo.artifacts import export_stage_artifacts
from dataaug_multi_both.hpo.run_study import StageResult, StageSettings


def _create_dummy_stage(tmp_path: Path) -> StageResult:
    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0.0, 1.0)
        return x

    study.optimize(objective, n_trials=3)
    settings = StageSettings(
        stage_name="stage_test",
        search_space={"x": {"type": "float", "low": 0.0, "high": 1.0}},
        sampler=optuna.samplers.RandomSampler(seed=42),
        pruner=optuna.pruners.NopPruner(),
        n_trials=3,
        timeout=None,
        plateau_patience=5,
        epochs=1,
        frozen_params={"frozen": 1},
    )
    best = study.best_trial
    return StageResult(settings=settings, study=study, best_trial=best, completed_trials=study.trials)


def test_export_stage_artifacts(tmp_path):
    stage_result = _create_dummy_stage(tmp_path)
    export_stage_artifacts(stage_result, tmp_path / "artifacts", buffer=None)
    assert (tmp_path / "artifacts/best_params.json").exists()
    assert (tmp_path / "artifacts/resolved_config.json").exists()
