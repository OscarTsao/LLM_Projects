"""Unit tests for trainer."""

import numpy as np
import pytest
import torch
from src.dataaug_multi_both.training.checkpoint_manager import (
    CheckpointManager,
    CheckpointMetadata,
    CheckpointRetentionPolicy,
)
from src.dataaug_multi_both.training.trainer import (
    Trainer,
    TrainerConfig,
    TrainingState,
    seed_everything,
)


class TestTrainerConfig:
    """Test suite for TrainerConfig."""
    
    def test_config_creation(self):
        """Test that config can be created."""
        config = TrainerConfig(
            trial_id="trial_001",
            optimization_metric="validation_f1_macro"
        )
        assert config.trial_id == "trial_001"
        assert config.optimization_metric == "validation_f1_macro"
        assert config.seed == 1337  # default
        assert config.max_epochs == 1  # default
    
    def test_config_custom_values(self):
        """Test that config accepts custom values."""
        config = TrainerConfig(
            trial_id="trial_002",
            optimization_metric="validation_accuracy",
            seed=42,
            max_epochs=10,
            gradient_accumulation_steps=4
        )
        assert config.seed == 42
        assert config.max_epochs == 10
        assert config.gradient_accumulation_steps == 4


class TestTrainingState:
    """Test suite for TrainingState."""
    
    def test_state_creation(self):
        """Test that training state can be created."""
        state = TrainingState(epoch=0, global_step=0)
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.best_metric is None
        assert state.metrics == {}
    
    def test_state_with_metrics(self):
        """Test training state with metrics."""
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


class TestSeedEverything:
    """Test suite for seed_everything function."""
    
    def test_seed_everything_returns_dict(self):
        """Test that seed_everything returns seed dictionary."""
        seeds = seed_everything(42)
        assert isinstance(seeds, dict)
        assert "python" in seeds
        assert "numpy" in seeds
    
    def test_seed_everything_deterministic(self):
        """Test that seeding produces deterministic results."""
        # Seed and generate random numbers
        seed_everything(42)
        torch_rand_1 = torch.rand(5)
        
        # Seed again with same seed
        seed_everything(42)
        torch_rand_2 = torch.rand(5)
        
        # Should be identical
        assert torch.allclose(torch_rand_1, torch_rand_2)
    
    def test_seed_everything_different_seeds(self):
        """Test that different seeds produce different results."""
        seed_everything(42)
        torch_rand_1 = torch.rand(5)
        
        seed_everything(123)
        torch_rand_2 = torch.rand(5)
        
        # Should be different
        assert not torch.allclose(torch_rand_1, torch_rand_2)
    
    def test_seed_everything_includes_torch_seed(self):
        """Test that torch seed is included in returned dict."""
        seeds = seed_everything(42)
        assert "torch" in seeds
        assert seeds["torch"] == 42
    
    def test_seed_everything_includes_cuda_seed(self):
        """Test that CUDA seed is included if CUDA available."""
        seeds = seed_everything(42)
        # CUDA seed should be present if CUDA is available
        if torch.cuda.is_available():
            assert "torch_cuda" in seeds


class TestTrainerThresholding:
    """Tests for Trainer threshold utilities."""

    def test_tune_validation_threshold(self, tmp_path):
        config = TrainerConfig(trial_id="trial-test", optimization_metric="validation_macro_f1")
        checkpoint_manager = CheckpointManager(
            trial_dir=tmp_path,
            policy=CheckpointRetentionPolicy(),
            compatibility=CheckpointMetadata(
                code_version="ci",
                model_signature="simulation",
                head_configuration="test",
            ),
        )
        trainer = Trainer(config=config, checkpoint_manager=checkpoint_manager)

        logits = np.array([[2.0, -1.0], [0.5, 0.1]])
        labels = np.array([[1, 0], [0, 1]])
        outcome = trainer.tune_validation_threshold(logits, labels, per_class=True)
        assert "macro_f1" in outcome
        assert "thresholds" in outcome
        assert outcome["threshold_mode"] == "per_class"
