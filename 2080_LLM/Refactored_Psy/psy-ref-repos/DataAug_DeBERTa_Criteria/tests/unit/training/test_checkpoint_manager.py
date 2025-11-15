from __future__ import annotations

from pathlib import Path

import pytest

from src.dataaug_multi_both.training.checkpoint_manager import (
    CheckpointCompatibilityError,
    CheckpointManager,
    CheckpointMetadata,
    CheckpointRetentionPolicy,
    StorageCapacityError,
    StorageStats,
)


def _build_manager(base_dir: Path) -> CheckpointManager:
    policy = CheckpointRetentionPolicy(keep_last_n=2, keep_best_k=1, max_total_size_gb=0.001)
    compatibility = CheckpointMetadata(
        code_version="deadbeef",
        model_signature="encoder:v1",
        head_configuration="criteria-head:v1",
    )
    return CheckpointManager(trial_dir=base_dir, policy=policy, compatibility=compatibility)


def _dummy_state(epoch: int) -> dict[str, int]:
    return {"epoch": epoch, "value": epoch * 2}


@pytest.mark.unit
def test_save_and_load_checkpoint_roundtrip(workspace_tmp_path: Path) -> None:
    manager = _build_manager(workspace_tmp_path)

    manager.save_checkpoint(state=_dummy_state(1), epoch=1, metric_value=0.5)
    manager.save_checkpoint(state=_dummy_state(2), epoch=2, metric_value=0.6)

    record, payload = manager.load_latest_checkpoint()

    assert record.epoch == 2
    assert payload["epoch"] == 2
    assert payload["value"] == 4


@pytest.mark.unit
def test_pruning_respects_retention_policy(workspace_tmp_path: Path) -> None:
    policy = CheckpointRetentionPolicy(keep_last_n=1, keep_best_k=1, max_total_size_gb=0.0002)
    compatibility = CheckpointMetadata(
        code_version="deadbeef",
        model_signature="encoder:v1",
        head_configuration="criteria-head:v1",
    )
    manager = CheckpointManager(trial_dir=workspace_tmp_path, policy=policy, compatibility=compatibility)

    # Save checkpoints with increasing metric to mark newer as best
    manager.save_checkpoint(state=_dummy_state(1), epoch=1, metric_value=0.1)
    manager.save_checkpoint(state=_dummy_state(2), epoch=2, metric_value=0.5)
    manager.save_checkpoint(state=_dummy_state(3), epoch=3, metric_value=0.2)

    checkpoint_files = sorted(workspace_tmp_path.glob("checkpoint_epoch*.pt"))
    assert len(checkpoint_files) == 2  # keep_last_n=1 + keep_best_k=1
    epochs = {path.stem.split("epoch")[-1] for path in checkpoint_files}
    assert epochs == {"0002", "0003"}


@pytest.mark.unit
def test_co_best_checkpoints_preserved(workspace_tmp_path: Path) -> None:
    policy = CheckpointRetentionPolicy(keep_last_n=1, keep_best_k=1, max_total_size_gb=0.001)
    compatibility = CheckpointMetadata(
        code_version="deadbeef",
        model_signature="encoder:v1",
        head_configuration="criteria-head:v1",
    )
    manager = CheckpointManager(trial_dir=workspace_tmp_path, policy=policy, compatibility=compatibility)

    manager.save_checkpoint(state=_dummy_state(1), epoch=1, metric_value=0.8)
    manager.save_checkpoint(state=_dummy_state(2), epoch=2, metric_value=0.8)

    best_epochs = [record.epoch for record in manager.best_checkpoints()]
    assert set(best_epochs) == {1, 2}


@pytest.mark.unit
def test_preflight_storage_error(workspace_tmp_path: Path) -> None:
    manager = _build_manager(workspace_tmp_path)

    storage_stats = StorageStats(available_bytes=100_000_000, total_bytes=1_000_000_000)

    with pytest.raises(StorageCapacityError) as err:
        manager.preflight_check(estimated_checkpoint_size_bytes=800_000_000, storage_stats=storage_stats)

    message = str(err.value)
    assert "Storage exhaustion detected" in message
    assert "hydra override" in message


@pytest.mark.unit
def test_incompatible_checkpoint_rejected(workspace_tmp_path: Path) -> None:
    manager = _build_manager(workspace_tmp_path)
    manager.save_checkpoint(state=_dummy_state(1), epoch=1, metric_value=0.3)

    incompatible_manager = CheckpointManager(
        trial_dir=workspace_tmp_path,
        policy=manager.policy,
        compatibility=CheckpointMetadata(
            code_version="other",
            model_signature="encoder:v1",
            head_configuration="criteria-head:v1",
        ),
    )

    with pytest.raises(CheckpointCompatibilityError):
        incompatible_manager.load_latest_checkpoint()
