from __future__ import annotations

from copy import deepcopy

from dataaug_multi_both.training.train_loop import run_training_job


def test_early_stopping_triggers(monkeypatch, test_config):
    cfg = deepcopy(test_config)
    cfg["train"]["num_epochs"] = 8
    cfg["train"]["early_stopping"]["patience"] = 2
    cfg["train"]["early_stopping"]["min_delta"] = 0.0

    def constant_metrics(*args, split: str, **kwargs):
        return {
            f"{split}_ev_macro_f1": 0.5,
            f"{split}_ev_accuracy": 0.5,
            f"{split}_cri_macro_f1": 0.5,
            f"{split}_cri_accuracy": 0.5,
            f"{split}_macro_f1_mean": 0.5,
        }

    monkeypatch.setattr(
        "dataaug_multi_both.training.train_loop.compute_metrics",
        constant_metrics,
    )

    result = run_training_job(cfg)
    assert result["epochs_run"] <= cfg["train"]["early_stopping"]["patience"] + 1
