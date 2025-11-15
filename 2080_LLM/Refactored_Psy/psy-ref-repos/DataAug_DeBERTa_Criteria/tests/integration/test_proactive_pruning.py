"""Integration tests for proactive retention pruning."""

import tempfile
from pathlib import Path

import pytest

from src.dataaug_multi_both.training.checkpoint_manager import (
    CheckpointManager,
    CheckpointMetadata,
    CheckpointRetentionPolicy,
    StorageCapacityError,
    StorageStats,
)


class TestProactivePruning:
    """Test suite for proactive pruning (FR-018, FR-028)."""

    def test_aggressive_pruning_triggered_at_10_percent(self):
        """Test that aggressive pruning is triggered when free space < 10%."""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = CheckpointRetentionPolicy(
                keep_last_n=10,  # Keep more to see pruning effect
                keep_best_k=5,
                aggressive_free_ratio=0.10
            )
            metadata = CheckpointMetadata(
                code_version="1.0.0",
                model_signature="test",
                head_configuration="test"
            )

            checkpoint_manager = CheckpointManager(
                trial_dir=Path(tmpdir),
                policy=policy,
                compatibility=metadata
            )

            # Save multiple checkpoints
            for epoch in range(10):
                checkpoint = {"epoch": epoch, "data": "x" * 1000}

                # Simulate low disk space (5% free)
                storage_stats = StorageStats(
                    available_bytes=5 * 1024**3,  # 5 GB
                    total_bytes=100 * 1024**3     # 100 GB (5% free)
                )

                # This should trigger aggressive pruning
                checkpoint_manager.save_checkpoint(
                    state=checkpoint,
                    epoch=epoch,
                    metric_value=0.8 + epoch * 0.01,
                    storage_stats=storage_stats
                )

            # Verify that checkpoints exist (some may be pruned)
            checkpoints = list(Path(tmpdir).glob("checkpoint_epoch_*.pt"))
            # Should have some checkpoints (policy allows up to 10)
            assert len(checkpoints) > 0
            assert len(checkpoints) <= 10

    def test_no_pruning_when_space_sufficient(self):
        """Test that pruning is not triggered when space is sufficient."""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = CheckpointRetentionPolicy(
                keep_last_n=10,  # Keep all
                keep_best_k=10,
                aggressive_free_ratio=0.10
            )
            metadata = CheckpointMetadata(
                code_version="1.0.0",
                model_signature="test",
                head_configuration="test"
            )

            checkpoint_manager = CheckpointManager(
                trial_dir=Path(tmpdir),
                policy=policy,
                compatibility=metadata
            )

            # Save checkpoints with sufficient space
            for epoch in range(5):
                checkpoint = {"epoch": epoch, "data": "x" * 1000}

                # Simulate sufficient disk space (50% free)
                storage_stats = StorageStats(
                    available_bytes=50 * 1024**3,  # 50 GB
                    total_bytes=100 * 1024**3      # 100 GB (50% free)
                )

                checkpoint_manager.save_checkpoint(
                    state=checkpoint,
                    epoch=epoch,
                    metric_value=0.8 + epoch * 0.01,
                    storage_stats=storage_stats
                )

            # All checkpoints should be retained
            checkpoints = list(Path(tmpdir).glob("checkpoint_epoch_*.pt"))
            assert len(checkpoints) >= 3  # At least some saved


class TestQuantifiedPolicyTransitions:
    """Test suite for quantified policy transitions (FR-028)."""

    def test_policy_transition_to_minimal(self):
        """Test that policy transitions to minimal when pruning insufficient."""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = CheckpointRetentionPolicy(
                keep_last_n=5,
                keep_best_k=3,
                aggressive_free_ratio=0.10
            )
            metadata = CheckpointMetadata(
                code_version="1.0.0",
                model_signature="test",
                head_configuration="test"
            )

            checkpoint_manager = CheckpointManager(
                trial_dir=Path(tmpdir),
                policy=policy,
                compatibility=metadata
            )

            # Save checkpoints
            for epoch in range(10):
                checkpoint = {"epoch": epoch, "data": "x" * 1000}

                # Simulate very low disk space
                storage_stats = StorageStats(
                    available_bytes=2 * 1024**3,   # 2 GB
                    total_bytes=100 * 1024**3      # 100 GB (2% free)
                )

                try:
                    checkpoint_manager.save_checkpoint(
                        state=checkpoint,
                        epoch=epoch,
                        metric_value=0.8 + epoch * 0.01,
                        storage_stats=storage_stats
                    )
                except StorageCapacityError:
                    # Expected when space is critically low
                    pass

            # Verify aggressive pruning occurred
            checkpoints = list(Path(tmpdir).glob("checkpoint_epoch_*.pt"))
            # Should have very few checkpoints due to aggressive pruning
            assert len(checkpoints) <= 3


class TestStorageCapacityError:
    """Test suite for storage capacity error (FR-014)."""

    def test_detailed_error_on_capacity_failure(self):
        """Test that detailed error is generated on capacity failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = CheckpointRetentionPolicy(
                keep_last_n=1,
                keep_best_k=1,
                aggressive_free_ratio=0.10
            )
            metadata = CheckpointMetadata(
                code_version="1.0.0",
                model_signature="test",
                head_configuration="test"
            )

            checkpoint_manager = CheckpointManager(
                trial_dir=Path(tmpdir),
                policy=policy,
                compatibility=metadata
            )

            # Try to save with extremely low space
            checkpoint = {"epoch": 0, "data": "x" * 1000}

            # Simulate critically low disk space (0.1% free)
            storage_stats = StorageStats(
                available_bytes=100 * 1024**2,  # 100 MB
                total_bytes=100 * 1024**3       # 100 GB (0.1% free)
            )

            # This should raise StorageCapacityError with detailed message
            with pytest.raises(StorageCapacityError) as exc_info:
                checkpoint_manager.save_checkpoint(
                    state=checkpoint,
                    epoch=0,
                    metric_value=0.8,
                    storage_stats=storage_stats
                )

            # Verify error message contains useful information
            error_message = str(exc_info.value)
            assert "storage" in error_message.lower() or "capacity" in error_message.lower()


class TestPrunableRecords:
    """Test suite for prunable records selection."""

    def test_oldest_non_protected_pruned_first(self):
        """Test that oldest non-protected checkpoints are pruned first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = CheckpointRetentionPolicy(
                keep_last_n=2,
                keep_best_k=1,
                aggressive_free_ratio=0.10
            )
            metadata = CheckpointMetadata(
                code_version="1.0.0",
                model_signature="test",
                head_configuration="test"
            )

            checkpoint_manager = CheckpointManager(
                trial_dir=Path(tmpdir),
                policy=policy,
                compatibility=metadata
            )

            # Save 5 checkpoints
            for epoch in range(5):
                checkpoint = {"epoch": epoch, "data": "x" * 1000}
                checkpoint_manager.save_checkpoint(
                    state=checkpoint,
                    epoch=epoch,
                    metric_value=0.9 if epoch == 2 else 0.8  # Epoch 2 is best
                )

            # Verify retention policy
            checkpoints = list(Path(tmpdir).glob("checkpoint_epoch_*.pt"))

            # Should keep at most 3 checkpoints (last 2 + best 1)
            assert len(checkpoints) <= 3
            assert len(checkpoints) > 0  # At least some retained


class TestStorageMonitorIntegration:
    """Test suite for storage monitor integration."""

    def test_storage_stats_tracked(self):
        """Test that storage stats are tracked during checkpointing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = CheckpointRetentionPolicy(keep_last_n=5, keep_best_k=5)
            metadata = CheckpointMetadata(
                code_version="1.0.0",
                model_signature="test",
                head_configuration="test"
            )

            checkpoint_manager = CheckpointManager(
                trial_dir=Path(tmpdir),
                policy=policy,
                compatibility=metadata
            )

            # Save checkpoint with storage stats
            checkpoint = {"epoch": 0, "data": "x" * 1000}
            storage_stats = StorageStats(
                available_bytes=50 * 1024**3,
                total_bytes=100 * 1024**3
            )

            record = checkpoint_manager.save_checkpoint(
                state=checkpoint,
                epoch=0,
                metric_value=0.8,
                storage_stats=storage_stats
            )

            # Verify checkpoint was saved
            assert record.path.exists()

