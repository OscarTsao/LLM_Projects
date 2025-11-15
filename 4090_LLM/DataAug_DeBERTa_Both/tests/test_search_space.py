from __future__ import annotations

import optuna
from optuna.samplers import RandomSampler

from src.dataaug_multi_both.hpo import ALLOWED_EFFECTIVE_BATCH_SIZES, Stage, sample_parameters
from src.dataaug_multi_both.hpo.search_space_v2 import NarrowedSpace


def _create_study(seed: int = 0) -> optuna.Study:
    return optuna.create_study(direction="maximize", sampler=RandomSampler(seed=seed))


def test_stage1_sampling_enforces_effective_batch_constraint() -> None:
    study = _create_study(seed=11)
    trial = study.ask()
    params = sample_parameters(trial, Stage.STAGE1)
    product = params["batch_size"] * params["gradient_accumulation_steps"]
    assert product in ALLOWED_EFFECTIVE_BATCH_SIZES
    assert "decision_thresholds_per_class" not in params
    study.tell(trial, 0.1)


def test_stage2_sampling_uses_narrowing_constraints() -> None:
    narrowing = NarrowedSpace()
    narrowing.merge_float("learning_rate", (1e-5, 1e-4))
    narrowing.merge_categorical("optimizer", ["adam"])
    narrowing.merge_categorical("max_seq_length", [384])
    narrowing.merge_categorical("batch_size", [32])
    narrowing.merge_categorical("gradient_accumulation_steps", [2])

    study = _create_study(seed=17)
    trial = study.ask()
    params = sample_parameters(trial, Stage.STAGE2, narrowing=narrowing)
    assert 1e-5 <= params["learning_rate"] <= 1e-4
    assert params["optimizer"] == "adam"
    assert params["batch_size"] == 32
    assert params["gradient_accumulation_steps"] == 2
    assert params["max_seq_length"] in (384, 512)
    study.tell(trial, 0.2)


def test_stage2_assigns_threshold_mode() -> None:
    study = _create_study(seed=23)
    trial = study.ask()
    params = sample_parameters(trial, Stage.STAGE2)
    assert "decision_thresholds_per_class" in params
    assert isinstance(params["decision_thresholds_per_class"], bool)
    study.tell(trial, 0.3)
