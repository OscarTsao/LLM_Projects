from __future__ import annotations

import optuna
from optuna.trial import TrialState, create_trial

from src.dataaug_multi_both.hpo import narrow_stage2_space


def _build_trial(value: float, params: dict, user_attrs: dict | None = None) -> optuna.trial.FrozenTrial:
    distributions = {}
    for key, param_value in params.items():
        if isinstance(param_value, float):
            low = max(param_value * 0.9, 1e-12)
            high = param_value * 1.1
            distributions[key] = optuna.distributions.UniformDistribution(low=low, high=high)
        elif isinstance(param_value, int):
            distributions[key] = optuna.distributions.IntUniformDistribution(low=param_value, high=param_value)
        else:
            distributions[key] = optuna.distributions.CategoricalDistribution(choices=[param_value])
    return create_trial(
        params=params,
        distributions=distributions,
        value=value,
        state=TrialState.COMPLETE,
        user_attrs=user_attrs or {},
    )


def test_narrow_stage2_space_extracts_top_k_ranges() -> None:
    trials = [
        _build_trial(
            value=0.8,
            params={
                "learning_rate": 1e-4,
                "optimizer": "adam",
                "max_seq_length": 384,
                "batching_combo": "bs16_ga2_eff32",
            },
            user_attrs={"batch_size": 16, "gradient_accumulation_steps": 2},
        ),
        _build_trial(
            value=0.7,
            params={
                "learning_rate": 2e-4,
                "optimizer": "adamw",
                "max_seq_length": 256,
                "batching_combo": "bs32_ga1_eff32",
            },
            user_attrs={"batch_size": 32, "gradient_accumulation_steps": 1},
        ),
        _build_trial(
            value=0.5,
            params={
                "learning_rate": 5e-5,
                "optimizer": "sgd",
                "max_seq_length": 128,
                "batching_combo": "bs8_ga2_eff16",
            },
            user_attrs={"batch_size": 8, "gradient_accumulation_steps": 2},
        ),
    ]

    narrowed = narrow_stage2_space(trials, top_k=2)
    lr_low, lr_high = narrowed.float_ranges["learning_rate"]
    assert lr_low >= 5e-6 and lr_high <= 3e-3
    assert set(narrowed.categorical_choices["optimizer"]) <= {"adam", "adamw"}
    assert "max_seq_length" in narrowed.categorical_choices
    assert "batch_size" in narrowed.categorical_choices
    assert "gradient_accumulation_steps" in narrowed.categorical_choices
