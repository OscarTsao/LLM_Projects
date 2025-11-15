from __future__ import annotations

from pathlib import Path

import optuna

from src.dataaug_multi_both.hpo import run_stage1, run_stage2


def test_two_stage_smoke(tmp_path) -> None:
    storage_path = tmp_path / "optuna.db"
    storage = f"sqlite:///{storage_path}"
    experiments_dir = tmp_path / "experiments"

    stage1 = run_stage1(
        storage=storage,
        n_trials=2,
        n_jobs=1,
        experiments_dir=experiments_dir,
    )
    assert len(stage1.trials) >= 2

    stage2 = run_stage2(
        storage=storage,
        stage1_study=stage1.study_name,
        n_trials=2,
        n_jobs=1,
        experiments_dir=experiments_dir,
        top_k=1,
    )
    assert len(stage2.trials) >= 2

    for trial in stage1.get_trials(deepcopy=False):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            report_path = trial.user_attrs.get("evaluation_report")
            if report_path:
                assert Path(report_path).exists()
