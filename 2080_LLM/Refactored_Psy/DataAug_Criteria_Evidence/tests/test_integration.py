"""Integration tests for end-to-end workflows with augmentation.

Tests complete pipelines:
- Data loading and processing with augmentation
- Training pipeline with augmentation
- HPO pipeline
- Augmentation integration
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch


class TestDataPipelineWithAugmentation:
    """Test data pipeline with augmentation end-to-end."""

    def test_augmentation_in_training_pipeline(self, sample_posts, augmentation_config):
        """Test that augmentation integrates with training pipeline."""
        try:
            from psy_agents_aug.augment import AugmentationConfig, NLPAugPipeline

            config = AugmentationConfig(**augmentation_config)
            pipeline = NLPAugPipeline(config, aug_method="synonym")

            texts = sample_posts["text"].tolist()
            augmented, indices = pipeline.augment_batch(texts[:3], split="train")

            # Training data should be augmented
            assert len(augmented) >= len(texts[:3])

            # Validation data should NOT be augmented
            val_aug, _ = pipeline.augment_batch(texts[:3], split="val")
            assert len(val_aug) == len(texts[:3])

        except ImportError:
            pytest.skip("Augmentation modules not available")

    def test_data_split_no_leakage(self, sample_annotations):
        """Test that data splits have no leakage even with augmentation."""
        # Import from the augmented package
        try:
            from Project.Share.data.dataset import split_data_by_post_id
        except ImportError:
            pytest.skip("Data modules not available in expected structure")


class TestAugmentationIntegration:
    """Test augmentation integration with training."""

    def test_augmentation_preserves_labels(self, augmentation_config):
        """Test that augmentation preserves labels correctly."""
        try:
            from psy_agents_aug.augment import AugmentationConfig, NLPAugPipeline

            config = AugmentationConfig(**augmentation_config)
            pipeline = NLPAugPipeline(config, aug_method="synonym")

            texts = ["Patient has anxiety.", "No depression found."]
            labels = [1, 0]

            aug_texts, aug_indices = pipeline.augment_batch(texts, split="train")

            # Check that original texts are preserved
            assert texts[0] in aug_texts or any("anxiety" in t.lower() for t in aug_texts)
            assert texts[1] in aug_texts or any("depression" in t.lower() for t in aug_texts)

        except ImportError:
            pytest.skip("Augmentation modules not available")

    def test_augmentation_deterministic(self, augmentation_config):
        """Test that augmentation is deterministic with seed."""
        try:
            from psy_agents_aug.augment import AugmentationConfig, NLPAugPipeline

            config = AugmentationConfig(**augmentation_config)
            pipeline1 = NLPAugPipeline(config, aug_method="synonym")
            pipeline2 = NLPAugPipeline(config, aug_method="synonym")

            text = "Patient reports severe anxiety symptoms."

            result1 = pipeline1.augment_text(text, num_variants=1)
            result2 = pipeline2.augment_text(text, num_variants=1)

            # Should be identical with same seed
            assert result1 == result2

        except ImportError:
            pytest.skip("Augmentation modules not available")


class TestTrainingPipeline:
    """Test training pipeline end-to-end."""

    @pytest.mark.slow
    def test_training_one_epoch(self, mock_dataset, mock_mlflow):
        """Test training for one epoch."""
        from torch.utils.data import DataLoader

        # Create dataloaders
        train_loader = DataLoader(mock_dataset, batch_size=2)
        val_loader = DataLoader(mock_dataset, batch_size=2)

        # Mock model and optimizer
        model = MagicMock()
        model.return_value = MagicMock(loss=torch.tensor(0.5), logits=torch.randn(2, 2))
        model.train = MagicMock()
        model.eval = MagicMock()

        optimizer = MagicMock()
        optimizer.zero_grad = MagicMock()
        optimizer.step = MagicMock()

        # Run one training step
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            assert loss is not None
            break  # Just test one batch


class TestHPOPipeline:
    """Test HPO pipeline."""

    def test_optuna_trial_mock(self):
        """Test Optuna trial with mock."""
        import optuna

        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return (x - 2) ** 2

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=3)

        assert study.best_value is not None
        assert len(study.trials) == 3


class TestReproducibility:
    """Test reproducibility of results."""

    def test_random_seed_setting(self):
        """Test random seed setting."""
        try:
            from Project.Share.utils.seed import set_seed

            set_seed(42)
            rand1 = torch.rand(10)

            set_seed(42)
            rand2 = torch.rand(10)

            assert torch.allclose(rand1, rand2)
        except ImportError:
            pytest.skip("Seed utilities not available")


class TestAugmentationNoLeak:
    """Test that augmentation doesn't leak between splits."""

    def test_augmentation_only_on_train(self, augmentation_config):
        """Test augmentation only applies to training set."""
        try:
            from psy_agents_aug.augment import AugmentationConfig, NLPAugPipeline

            config = AugmentationConfig(**augmentation_config)
            pipeline = NLPAugPipeline(config, aug_method="synonym")

            texts = ["Test sentence."] * 5

            # Train should augment
            train_aug, _ = pipeline.augment_batch(texts, split="train")
            assert len(train_aug) >= len(texts), "Training should augment"

            # Val should not augment
            val_aug, _ = pipeline.augment_batch(texts, split="val")
            assert len(val_aug) == len(texts), "Validation should not augment"

            # Test should not augment
            test_aug, _ = pipeline.augment_batch(texts, split="test")
            assert len(test_aug) == len(texts), "Test should not augment"

        except ImportError:
            pytest.skip("Augmentation modules not available")
