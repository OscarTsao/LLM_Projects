from __future__ import annotations

from typing import Any

import optuna
import pytest
import torch

from dataaug_multi_both.training import train_loop


def test_cuda_oom_marks_trial_pruned(monkeypatch, test_config):
    class CrashModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(1, 1)

        def forward(self, *args, **kwargs):  # pragma: no cover - intentional failure path
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(
        "dataaug_multi_both.training.train_loop.build_multitask_model",
        lambda cfg: CrashModel(),
    )

    captured_params: list[dict[str, Any]] = []

    def capture_params(params: dict[str, Any]) -> None:
        captured_params.append(params)

    monkeypatch.setattr(
        "dataaug_multi_both.training.train_loop.log_params_safe",
        capture_params,
    )

    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    with pytest.raises(optuna.TrialPruned):
        train_loop.run_training_job(test_config, trial=trial)

    flattened = {k: v for entry in captured_params for k, v in entry.items()}
    assert "oom.batch_size" in flattened
    assert "oom.model_name" in flattened
