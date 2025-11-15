from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import optuna
import pytest

from dataaug_multi_both.hpo.driver import run_two_stage_hpo
from dataaug_multi_both.hpo.run_study import prepare_stage_b_controls, run_optuna_stage
from dataaug_multi_both.hpo.space import define_search_space


def test_prepare_stage_b_controls_freezes_structural_keys(test_config):
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    cfg = define_search_space(trial, base_config=test_config)
    frozen, narrowed = prepare_stage_b_controls(cfg)

    assert "loss.cw" in frozen
    combo_label = frozen["aug.simple.combo"]
    if combo_label == "none":
        combo_count = 0
    else:
        combo_count = len(combo_label.split("|"))
    assert combo_count <= 2
    assert "opt.lr_enc" in narrowed
    assert narrowed["opt.lr_enc"]["low"] <= cfg["optim"]["lr_encoder"] <= narrowed["opt.lr_enc"]["high"]


def test_run_optuna_stage_structure_selection(monkeypatch, tmp_path: Path, test_config):
    storage = f"sqlite:///{tmp_path / 'stage.db'}"

    def fake_run_training_job(cfg: Mapping[str, Any], *, trial=None, resume=False):
        struct_idx = trial.user_attrs.get("structure_index", 0) if trial else 0
        return {
            "objective": 0.5 + struct_idx,
            "metrics": {"val_ev_macro_f1": 0.5 + struct_idx},
            "checkpoint_path": "",
            "evaluation_report_path": "",
            "epochs_run": cfg["train"]["num_epochs"],
            "pruned_at_epoch": 0,
        }

    monkeypatch.setattr(
        "dataaug_multi_both.hpo.run_study.run_training_job",
        fake_run_training_job,
    )

    structure_pool = [
        (
            {
                "loss.cw": "none",
                "aug.simple.combo": "none",
                "aug.simple.compose": "one_of",
                "aug.ta.combo": "none",
                "aug.compose_cross_family": "serial_only",
            },
            {},
        ),
        (
            {
                "loss.cw": "inverse_freq",
                "aug.simple.combo": "EDA",
                "aug.simple.compose": "one_of",
                "aug.ta.combo": "none",
                "aug.compose_cross_family": "serial_only",
            },
            {
                "aug.simple.EDA.p_token": {"low": 0.05, "high": 0.1},
                "aug.simple.EDA.tpe": {"low": 0, "high": 1},
            },
        ),
    ]

    stage_study = run_optuna_stage(
        test_config,
        study_name="test_stage",
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
        n_trials=2,
        timeout=None,
        stage_label="stage_b",
        stage_epochs=1,
        global_seed=42,
        objective_metric="val_ev_macro_f1",
        structure_pool=structure_pool,
        plateau_patience=None,
        study_direction="maximize",
        enqueue_params=None,
    )

    assert all("structure.id" in t.params for t in stage_study.trials)
    assert stage_study.best_trial.params["structure.id"] in {0, 1}


@pytest.mark.unit
def test_two_stage_driver_uses_defaults(tmp_path: Path, monkeypatch):
    storage = f"sqlite:///{tmp_path / 'driver.db'}"
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "dataaug_multi_both.hpo.driver._ensure_artifact_dir",
        lambda stage: tmp_path / "artifacts" / stage,
    )
    monkeypatch.setattr(
        "dataaug_multi_both.hpo.driver._materialize_stage_artifacts",
        lambda stage, study: None,
    )

    def fake_run_training_job(cfg: Mapping[str, Any], *, trial=None, resume=False):
        struct_idx = trial.user_attrs.get("structure_index", 0) if trial else 0
        return {
            "objective": 0.7 + 0.1 * struct_idx,
            "metrics": {"val_ev_macro_f1": 0.7 + 0.1 * struct_idx},
            "checkpoint_path": "",
            "evaluation_report_path": "",
            "epochs_run": cfg["train"]["num_epochs"],
            "pruned_at_epoch": 0,
        }

    monkeypatch.setattr(
        "dataaug_multi_both.hpo.run_study.run_training_job",
        fake_run_training_job,
    )

    summary = run_two_stage_hpo(
        model="hf-internal-testing/tiny-random-roberta",
        trials_a=2,
        epochs_a=1,
        trials_b=1,
        epochs_b=1,
        k_top=1,
        global_seed=13,
        timeout=None,
        config_overrides={
            "mlflow": {
                "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
                "artifact_location": str(tmp_path / "mlruns"),
                "buffer": {"dir": str(tmp_path / "mlflow_buffer")},
            },
            "hpo": {"storage": storage},
            "augmentation": {
                "simple": {
                    "enabled_mask": {
                        "EDA": False,
                        "CharSwap": False,
                        "Embedding": False,
                        "BackTranslation": False,
                        "CheckList": False,
                        "CLARE": False,
                    }
                },
                "ta": {
                    "enabled_mask": {
                        "TextFoolerJin2019": False,
                        "PWWSRen2019": False,
                        "DeepWordBugGao2018": False,
                        "HotFlipEbrahimi2017": False,
                        "IGAWang2019": False,
                        "Kuleshov2017": False,
                        "CheckList2020": False,
                        "BAEGarg2019": False,
                    }
                },
                "compose_cross_family": "serial_only",
            },
        },
    )

    assert summary["stage_b_best"]["value"] is not None
