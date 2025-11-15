"""Integration tests for checkpoint resume functionality."""

import tempfile
from pathlib import Path

from src.dataaug_multi_both.training.checkpoint_manager import (
    CheckpointManager,
    CheckpointMetadata,
    CheckpointRetentionPolicy,
)
from src.dataaug_multi_both.training.trainer import Trainer, TrainerConfig, TrainingState


class TestResumeFromCheckpoint:
    """Test suite for resume from checkpoint (FR-004)."""

    def test_resume_from_zero_checkpoints(self):
        """Test resume when no checkpoints exist (FR-030)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                trial_id="test_trial",
                optimization_metric="f1",
                resume_if_available=True
            )

            policy = CheckpointRetentionPolicy(keep_last_n=2, keep_best_k=1)
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

            trainer = Trainer(
                config=config,
                checkpoint_manager=checkpoint_manager
            )

            # Resume should start from epoch 0
            state = trainer.resume()

            assert state.epoch == 0
            assert state.global_step == 0
            assert state.best_metric is None

    def test_resume_from_existing_checkpoint(self):
        """Test resume from existing checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                trial_id="test_trial",
                optimization_metric="f1",
                resume_if_available=True
            )

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

            trainer = Trainer(
                config=config,
                checkpoint_manager=checkpoint_manager
            )

            # Save a checkpoint
            state = TrainingState(
                epoch=5,
                global_step=1000,
                best_metric=0.85,
                metrics={"f1": 0.85}
            )
            trainer.save_state(state, metric_value=0.85)

            # Create new trainer and resume
            trainer2 = Trainer(
                config=config,
                checkpoint_manager=checkpoint_manager
            )

            resumed_state = trainer2.resume()

            assert resumed_state.epoch == 5
            assert resumed_state.global_step == 1000
            assert resumed_state.best_metric == 0.85

    def test_resume_idempotent(self):
        """Test that resume is idempotent (FR-031)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                trial_id="test_trial",
                optimization_metric="f1",
                resume_if_available=True
            )

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

            trainer = Trainer(
                config=config,
                checkpoint_manager=checkpoint_manager
            )

            # Save checkpoint
            state = TrainingState(epoch=3, global_step=500, best_metric=0.80)
            trainer.save_state(state, metric_value=0.80)

            # Resume multiple times
            state1 = trainer.resume()
            state2 = trainer.resume()
            state3 = trainer.resume()

            # All should be identical
            assert state1.epoch == state2.epoch == state3.epoch == 3
            assert state1.global_step == state2.global_step == state3.global_step == 500

    def test_prepare_with_resume_disabled(self):
        """Test prepare when resume is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                trial_id="test_trial",
                optimization_metric="f1",
                resume_if_available=False
            )

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

            trainer = Trainer(
                config=config,
                checkpoint_manager=checkpoint_manager
            )

            # Save a checkpoint
            state = TrainingState(epoch=5, global_step=1000)
            trainer.save_state(state, metric_value=0.85)

            # Prepare should start fresh (not resume)
            prepared_state = trainer.prepare()

            assert prepared_state.epoch == 0
            assert prepared_state.global_step == 0

    def test_prepare_with_resume_enabled(self):
        """Test prepare when resume is enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                trial_id="test_trial",
                optimization_metric="f1",
                resume_if_available=True
            )

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

            trainer = Trainer(
                config=config,
                checkpoint_manager=checkpoint_manager
            )

            # Save a checkpoint
            state = TrainingState(epoch=5, global_step=1000, best_metric=0.85)
            trainer.save_state(state, metric_value=0.85)

            # Create new trainer and prepare
            trainer2 = Trainer(
                config=config,
                checkpoint_manager=checkpoint_manager
            )

            prepared_state = trainer2.prepare()

            # Should resume from checkpoint
            assert prepared_state.epoch == 5
            assert prepared_state.global_step == 1000


class TestCheckpointCompatibility:
    """Test suite for checkpoint compatibility validation (FR-025)."""

    def test_compatible_checkpoint_loads(self):
        """Test that compatible checkpoint loads successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                trial_id="test_trial",
                optimization_metric="f1"
            )

            policy = CheckpointRetentionPolicy(keep_last_n=5, keep_best_k=5)
            metadata = CheckpointMetadata(
                code_version="1.0.0",
                model_signature="bert-base",
                head_configuration="dual-agent"
            )

            checkpoint_manager = CheckpointManager(
                trial_dir=Path(tmpdir),
                policy=policy,
                compatibility=metadata
            )

            trainer = Trainer(
                config=config,
                checkpoint_manager=checkpoint_manager
            )

            # Save checkpoint
            state = TrainingState(epoch=1, global_step=100)
            trainer.save_state(state, metric_value=0.80)

            # Load with same metadata (compatible)
            checkpoint_manager2 = CheckpointManager(
                trial_dir=Path(tmpdir),
                policy=policy,
                compatibility=metadata
            )

            trainer2 = Trainer(
                config=config,
                checkpoint_manager=checkpoint_manager2
            )

            # Should load successfully
            resumed_state = trainer2.resume()
            assert resumed_state.epoch == 1


class TestMetricDuplication:
    """Test suite for preventing metric duplication on resume."""

    def test_no_metric_duplication_on_resume(self):
        """Test that metrics are not duplicated after resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                trial_id="test_trial",
                optimization_metric="f1"
            )

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

            trainer = Trainer(
                config=config,
                checkpoint_manager=checkpoint_manager
            )

            # Save checkpoint with metrics
            state = TrainingState(
                epoch=3,
                global_step=500,
                metrics={"f1": 0.85, "accuracy": 0.90}
            )
            trainer.save_state(state, metric_value=0.85)

            # Resume
            resumed_state = trainer.resume()

            # Metrics should be preserved
            assert resumed_state.metrics["f1"] == 0.85
            assert resumed_state.metrics["accuracy"] == 0.90

            # Global step should continue from where it left off
            assert resumed_state.global_step == 500

