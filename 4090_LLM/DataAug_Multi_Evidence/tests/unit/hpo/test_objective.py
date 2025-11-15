from __future__ import annotations

from pathlib import Path

import optuna

from dataaug_multi_both.hpo import ObjectiveConfig, build_objective
from dataaug_multi_both.hpo.run_study import StageSettings, run_stage


def test_objective_runs_with_synthetic_dataset(tmp_path) -> None:
    cfg = ObjectiveConfig(
        output_root=Path(tmp_path) / "hpo",
        synthetic_train_size=16,
        synthetic_val_size=8,
        synthetic_seq_len=32,
        use_synthetic=True,
    )
    objective = build_objective(cfg)

    search_space = {
        "train_batch_size": {"type": "categorical", "choices": [4]},
        "train_grad_accum": {"type": "categorical", "choices": [1]},
        "train_max_length": {"type": "categorical", "choices": [32]},
        "train_grad_clip": {"type": "float", "low": 0.5, "high": 1.0},
    }

    settings = StageSettings(
        stage_name="unit_stage",
        search_space=search_space,
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=optuna.pruners.NopPruner(),
        n_trials=2,
        timeout=None,
        plateau_patience=5,
        epochs=1,
    )

    result = run_stage(settings, objective)
    assert len(result.completed_trials) >= 1
    assert result.best_trial.value is not None
