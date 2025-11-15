from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from src.dataaug_multi_both.training.checkpoint_manager import (
    CheckpointManager,
    CheckpointMetadata,
    CheckpointRetentionPolicy,
)
from src.dataaug_multi_both.training.trainer import Trainer, TrainerConfig, TrainingState, seed_everything


class MlflowStub:
    def __init__(self) -> None:
        self.tags: dict[str, str] = {}

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value


def _build_manager(base_dir: Path) -> CheckpointManager:
    policy = CheckpointRetentionPolicy(keep_last_n=2, keep_best_k=1, max_total_size_gb=0.5)
    compatibility = CheckpointMetadata(
        code_version="abc123",
        model_signature="encoder:v1",
        head_configuration="head:v1",
    )
    return CheckpointManager(trial_dir=base_dir, policy=policy, compatibility=compatibility)


@pytest.mark.unit
def test_seed_everything_is_deterministic() -> None:
    seed_everything(42)
    first_values = (
        random.random(),
        float(np.random.random()),
    )

    seed_everything(42)
    second_values = (
        random.random(),
        float(np.random.random()),
    )

    assert first_values == second_values


@pytest.mark.unit
def test_prepare_without_resume_returns_zero_state(workspace_tmp_path: Path) -> None:
    manager = _build_manager(workspace_tmp_path)
    trainer = Trainer(
        config=TrainerConfig(trial_id="trial-1", optimization_metric="val_f1", resume_if_available=False),
        checkpoint_manager=manager,
    )

    state = trainer.prepare()

    assert state.epoch == 0
    assert state.global_step == 0


@pytest.mark.unit
def test_resume_returns_checkpoint_state(workspace_tmp_path: Path) -> None:
    manager = _build_manager(workspace_tmp_path)
    trainer = Trainer(
        config=TrainerConfig(trial_id="trial-1", optimization_metric="val_f1"),
        checkpoint_manager=manager,
    )

    # Save a checkpoint and ensure the trainer loads it
    trainer.save_state(TrainingState(epoch=3, global_step=90, best_metric=0.75), metric_value=0.75)

    # New trainer instance to simulate new process
    trainer2 = Trainer(
        config=TrainerConfig(trial_id="trial-1", optimization_metric="val_f1"),
        checkpoint_manager=_build_manager(workspace_tmp_path),
    )

    resumed = trainer2.resume()

    assert resumed.epoch == 3
    assert resumed.global_step == 90
    assert resumed.best_metric == 0.75


@pytest.mark.unit
def test_resume_is_idempotent(workspace_tmp_path: Path) -> None:
    manager = _build_manager(workspace_tmp_path)
    trainer = Trainer(
        config=TrainerConfig(trial_id="trial-1", optimization_metric="val_f1"),
        checkpoint_manager=manager,
    )

    trainer.save_state(TrainingState(epoch=1, global_step=10), metric_value=0.1)

    first = trainer.resume()
    second = trainer.resume()

    assert first.epoch == second.epoch == 1
    assert first.global_step == second.global_step == 10


@pytest.mark.unit
def test_seed_tags_written_to_mlflow(workspace_tmp_path: Path) -> None:
    manager = _build_manager(workspace_tmp_path)
    mlflow_stub = MlflowStub()
    trainer = Trainer(
        config=TrainerConfig(trial_id="trial-1", optimization_metric="val_f1", resume_if_available=False),
        checkpoint_manager=manager,
        mlflow_client=mlflow_stub,
    )

    trainer.prepare()

    assert "seed.python" in mlflow_stub.tags
    assert "seed.numpy" in mlflow_stub.tags
