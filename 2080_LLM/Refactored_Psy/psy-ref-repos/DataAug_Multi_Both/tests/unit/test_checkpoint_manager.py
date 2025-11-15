"""Unit tests for checkpoint manager."""

import pytest
import tempfile
import torch
from pathlib import Path
from src.dataaug_multi_both.training.checkpoint_manager import (
    CheckpointRetentionPolicy,
    CheckpointMetadata,
    CheckpointManager,
    CheckpointError,
    CheckpointCompatibilityError,
    CheckpointCorruptionError
)


class TestCheckpointRetentionPolicy:
    """Test suite for CheckpointRetentionPolicy."""
    
    def test_policy_defaults(self):
        """Test that policy has correct defaults."""
        policy = CheckpointRetentionPolicy()
        assert policy.keep_last_n == 1
        assert policy.keep_best_k == 1
        assert policy.max_total_size_gb is None
        assert policy.aggressive_free_ratio == 0.10
    
    def test_policy_custom_values(self):
        """Test that policy accepts custom values."""
        policy = CheckpointRetentionPolicy(
            keep_last_n=3,
            keep_best_k=2,
            max_total_size_gb=5.0,
            aggressive_free_ratio=0.15
        )
        assert policy.keep_last_n == 3
        assert policy.keep_best_k == 2
        assert policy.max_total_size_gb == 5.0
        assert policy.aggressive_free_ratio == 0.15


class TestCheckpointMetadata:
    """Test suite for CheckpointMetadata."""
    
    def test_metadata_creation(self):
        """Test that metadata can be created."""
        metadata = CheckpointMetadata(
            code_version="1.0.0",
            model_signature="bert-base",
            head_configuration="dual-agent"
        )
        assert metadata.code_version == "1.0.0"
        assert metadata.model_signature == "bert-base"
        assert metadata.head_configuration == "dual-agent"
        assert metadata.extra is None
    
    def test_metadata_with_extra(self):
        """Test metadata with extra fields."""
        metadata = CheckpointMetadata(
            code_version="1.0.0",
            model_signature="bert-base",
            head_configuration="dual-agent",
            extra={"author": "test", "experiment": "exp001"}
        )
        assert metadata.extra["author"] == "test"
        assert metadata.extra["experiment"] == "exp001"


class TestCheckpointManager:
    """Test suite for CheckpointManager."""
    
    def create_dummy_checkpoint(self, epoch: int = 0) -> dict:
        """Create a dummy checkpoint for testing."""
        return {
            "epoch": epoch,
            "model_state_dict": {"layer1.weight": torch.randn(10, 10)},
            "optimizer_state_dict": {"param_groups": []},
            "metrics": {"f1": 0.85, "accuracy": 0.90}
        }
    
    def test_manager_initialization(self):
        """Test that manager can be initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = CheckpointRetentionPolicy()
            metadata = CheckpointMetadata(
                code_version="1.0.0",
                model_signature="test",
                head_configuration="test"
            )

            manager = CheckpointManager(
                trial_dir=Path(tmpdir),
                policy=policy,
                compatibility=metadata
            )

            assert manager.trial_dir == Path(tmpdir)
            assert manager.policy == policy
            assert manager.compatibility == metadata
    
    def test_save_checkpoint(self):
        """Test saving a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = CheckpointRetentionPolicy(keep_last_n=5, keep_best_k=5)
            metadata = CheckpointMetadata(
                code_version="1.0.0",
                model_signature="test",
                head_configuration="test"
            )

            manager = CheckpointManager(
                trial_dir=Path(tmpdir),
                policy=policy,
                compatibility=metadata
            )

            checkpoint = self.create_dummy_checkpoint(epoch=0)

            record = manager.save_checkpoint(checkpoint, epoch=0, metric_value=0.85)

            assert record.path.exists()
            assert record.path.suffix == ".pt"
            assert record.epoch == 0
    
    def test_load_checkpoint(self):
        """Test loading a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = CheckpointRetentionPolicy(keep_last_n=5, keep_best_k=5)
            metadata = CheckpointMetadata(
                code_version="1.0.0",
                model_signature="test",
                head_configuration="test"
            )

            manager = CheckpointManager(
                trial_dir=Path(tmpdir),
                policy=policy,
                compatibility=metadata
            )

            # Save checkpoint
            checkpoint = self.create_dummy_checkpoint(epoch=0)
            record = manager.save_checkpoint(checkpoint, epoch=0, metric_value=0.85)

            # Load checkpoint
            loaded = manager.load_checkpoint(record.path)

            assert "model_state_dict" in loaded
            assert "optimizer_state_dict" in loaded
    
    def test_retention_policy_keep_last_n(self):
        """Test that retention policy keeps last N checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = CheckpointRetentionPolicy(keep_last_n=2, keep_best_k=0)
            metadata = CheckpointMetadata(
                code_version="1.0.0",
                model_signature="test",
                head_configuration="test"
            )

            manager = CheckpointManager(
                trial_dir=Path(tmpdir),
                policy=policy,
                compatibility=metadata
            )

            # Save 5 checkpoints
            for epoch in range(5):
                checkpoint = self.create_dummy_checkpoint(epoch=epoch)
                manager.save_checkpoint(checkpoint, epoch=epoch, metric_value=0.8 + epoch * 0.01)

            # Should only keep last 2
            checkpoints = list(Path(tmpdir).glob("checkpoint_epoch_*.pt"))
            assert len(checkpoints) == 2
    
    def test_retention_policy_keep_best_k(self):
        """Test that retention policy keeps best K checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = CheckpointRetentionPolicy(keep_last_n=0, keep_best_k=2)
            metadata = CheckpointMetadata(
                code_version="1.0.0",
                model_signature="test",
                head_configuration="test"
            )

            manager = CheckpointManager(
                trial_dir=Path(tmpdir),
                policy=policy,
                compatibility=metadata
            )

            # Save checkpoints with different metrics
            metrics_list = [0.80, 0.85, 0.90, 0.75, 0.88]
            for epoch, metric_value in enumerate(metrics_list):
                checkpoint = self.create_dummy_checkpoint(epoch=epoch)
                manager.save_checkpoint(checkpoint, epoch=epoch, metric_value=metric_value)

            # Should keep best 2 (0.90 and 0.88)
            checkpoints = list(Path(tmpdir).glob("checkpoint_epoch_*.pt"))
            assert len(checkpoints) == 2
    
    def test_get_best_checkpoint(self):
        """Test getting best checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = CheckpointRetentionPolicy(keep_last_n=5, keep_best_k=5)
            metadata = CheckpointMetadata(
                code_version="1.0.0",
                model_signature="test",
                head_configuration="test"
            )

            manager = CheckpointManager(
                trial_dir=Path(tmpdir),
                policy=policy,
                compatibility=metadata
            )

            # Save checkpoints
            best_epoch = 2
            for epoch in range(5):
                checkpoint = self.create_dummy_checkpoint(epoch=epoch)
                metric_value = 0.90 if epoch == best_epoch else 0.80
                manager.save_checkpoint(checkpoint, epoch=epoch, metric_value=metric_value)

            best_records = manager.best_checkpoints()
            assert len(best_records) > 0
            assert best_records[0].epoch == best_epoch
    
    def test_checkpoint_corruption_detection(self):
        """Test that corrupted checkpoints are detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = CheckpointRetentionPolicy(keep_last_n=5, keep_best_k=5)
            metadata = CheckpointMetadata(
                code_version="1.0.0",
                model_signature="test",
                head_configuration="test"
            )

            manager = CheckpointManager(
                trial_dir=Path(tmpdir),
                policy=policy,
                compatibility=metadata
            )

            # Save checkpoint
            checkpoint = self.create_dummy_checkpoint()
            record = manager.save_checkpoint(checkpoint, epoch=0, metric_value=0.85)

            # Corrupt the checkpoint
            with open(record.path, 'ab') as f:
                f.write(b'corrupted_data')

            # Loading should raise CheckpointCorruptionError
            with pytest.raises(CheckpointCorruptionError):
                manager.load_checkpoint(record.path)

