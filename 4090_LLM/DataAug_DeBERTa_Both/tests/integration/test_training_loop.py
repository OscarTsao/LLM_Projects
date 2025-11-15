"""Integration tests for training loop with checkpointing."""

import pytest
import tempfile
import torch
from pathlib import Path
from src.dataaug_multi_both.training.trainer import (
    Trainer,
    TrainerConfig,
    TrainingState,
    seed_everything,
    build_worker_init_fn
)
from src.dataaug_multi_both.training.checkpoint_manager import (
    CheckpointManager,
    CheckpointRetentionPolicy,
    CheckpointMetadata
)


class TestTrainingLoopIntegration:
    """Integration tests for training loop."""
    
    def test_seed_everything_deterministic(self):
        """Test that seeding produces deterministic results."""
        # Seed and generate random numbers
        seed_everything(42)
        torch_rand_1 = torch.rand(10)
        
        # Seed again with same seed
        seed_everything(42)
        torch_rand_2 = torch.rand(10)
        
        # Should be identical
        assert torch.allclose(torch_rand_1, torch_rand_2)
    
    def test_worker_init_fn_deterministic(self):
        """Test that worker init function produces deterministic results."""
        worker_init_fn = build_worker_init_fn(42)
        
        # Initialize worker 0
        worker_init_fn(0)
        torch_rand_1 = torch.rand(5)
        
        # Re-initialize worker 0
        worker_init_fn(0)
        torch_rand_2 = torch.rand(5)
        
        # Should be identical
        assert torch.allclose(torch_rand_1, torch_rand_2)
    
    def test_worker_init_fn_different_workers(self):
        """Test that different workers get different seeds."""
        worker_init_fn = build_worker_init_fn(42)
        
        # Initialize worker 0
        worker_init_fn(0)
        torch_rand_0 = torch.rand(5)
        
        # Initialize worker 1
        worker_init_fn(1)
        torch_rand_1 = torch.rand(5)
        
        # Should be different
        assert not torch.allclose(torch_rand_0, torch_rand_1)
    
    def test_trainer_initialization(self):
        """Test that trainer can be initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                trial_id="test_trial",
                optimization_metric="f1",
                seed=42,
                max_epochs=5
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
            
            assert trainer.config.trial_id == "test_trial"
            assert trainer.config.max_epochs == 5
    
    def test_training_state_creation(self):
        """Test creating training state."""
        state = TrainingState(
            epoch=5,
            global_step=1000,
            best_metric=0.85,
            metrics={"f1": 0.85, "accuracy": 0.90}
        )
        
        assert state.epoch == 5
        assert state.global_step == 1000
        assert state.best_metric == 0.85
        assert state.metrics["f1"] == 0.85


class TestDeterministicSeeding:
    """Test suite for deterministic seeding."""
    
    def test_seed_everything_includes_all_frameworks(self):
        """Test that all frameworks are seeded."""
        seeds = seed_everything(42)
        
        assert "python" in seeds
        assert "numpy" in seeds
        assert "torch" in seeds
        
        # CUDA seed only if CUDA available
        if torch.cuda.is_available():
            assert "torch_cuda" in seeds
    
    def test_seed_everything_reproducible_numpy(self):
        """Test that NumPy operations are reproducible."""
        import numpy as np
        
        seed_everything(42)
        arr1 = np.random.rand(10)
        
        seed_everything(42)
        arr2 = np.random.rand(10)
        
        assert np.allclose(arr1, arr2)
    
    def test_seed_everything_reproducible_python(self):
        """Test that Python random operations are reproducible."""
        import random
        
        seed_everything(42)
        vals1 = [random.random() for _ in range(10)]
        
        seed_everything(42)
        vals2 = [random.random() for _ in range(10)]
        
        assert vals1 == vals2


class TestCheckpointingIntegration:
    """Integration tests for checkpointing during training."""
    
    def test_checkpoint_manager_integration(self):
        """Test that checkpoint manager integrates with trainer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                trial_id="test_trial",
                optimization_metric="f1",
                max_epochs=3
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
            
            # Verify checkpoint manager is accessible
            assert trainer.checkpoint_manager == checkpoint_manager
    
    def test_resume_if_available_flag(self):
        """Test that resume_if_available flag is respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                trial_id="test_trial",
                optimization_metric="f1",
                resume_if_available=True
            )
            
            policy = CheckpointRetentionPolicy()
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
            
            assert trainer.config.resume_if_available is True


class TestMLflowIntegration:
    """Test suite for MLflow integration."""
    
    def test_trainer_with_mlflow_client(self):
        """Test that trainer accepts MLflow client."""
        from unittest.mock import Mock
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                trial_id="test_trial",
                optimization_metric="f1"
            )
            
            policy = CheckpointRetentionPolicy()
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
            
            mlflow_client = Mock()
            
            trainer = Trainer(
                config=config,
                checkpoint_manager=checkpoint_manager,
                mlflow_client=mlflow_client
            )
            
            assert trainer.mlflow_client == mlflow_client
    
    def test_log_seeds_to_mlflow_flag(self):
        """Test that log_seeds_to_mlflow flag is configurable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                trial_id="test_trial",
                optimization_metric="f1",
                log_seeds_to_mlflow=False
            )
            
            policy = CheckpointRetentionPolicy()
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
            
            assert trainer.config.log_seeds_to_mlflow is False

