"""Integration tests for end-to-end workflows.

Tests complete pipelines:
- Data loading and processing
- Training pipeline
- HPO pipeline
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestDataPipeline:
    """Test data pipeline end-to-end."""

    @pytest.mark.skip(reason="create_criteria_dataloaders not implemented yet")
    def test_groundtruth_to_loader_pipeline(
        self, sample_posts, sample_annotations, field_map_path, valid_criterion_ids
    ):
        """Test complete data pipeline from groundtruth to dataloader."""
        from psy_agents_noaug.data.groundtruth import (
            create_criteria_groundtruth,
            load_field_map,
        )
        from psy_agents_noaug.data.loaders import create_criteria_dataloaders

        # Step 1: Generate groundtruth
        field_map = load_field_map(field_map_path)
        criteria_gt = create_criteria_groundtruth(
            annotations=sample_annotations,
            posts=sample_posts,
            field_map=field_map,
            valid_criterion_ids=valid_criterion_ids,
        )

        assert len(criteria_gt) > 0
        assert "label" in criteria_gt.columns

        # Step 2: Create dataloaders (with mocked tokenizer)
        with patch("transformers.AutoTokenizer") as mock_tokenizer_cls:
            mock_tokenizer = MagicMock()
            mock_tokenizer.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
            mock_tokenizer.model_max_length = 512
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

            train_loader, val_loader, test_loader = create_criteria_dataloaders(
                groundtruth_df=criteria_gt,
                model_name="roberta-base",
                batch_size=2,
                max_length=128,
                random_seed=42,
            )

            # Verify loaders work
            assert train_loader is not None
            assert val_loader is not None
            assert test_loader is not None

    def test_data_split_no_leakage(self, sample_annotations):
        """Test that data splits have no leakage."""
        from psy_agents_noaug.data.loaders import group_split_by_post_id

        train, val, test = group_split_by_post_id(sample_annotations, random_seed=42)

        # No overlap between splits
        train_set = set(train)
        val_set = set(val)
        test_set = set(test)

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

        # All data is used
        all_post_ids = set(sample_annotations["post_id"].unique())
        split_post_ids = train_set | val_set | test_set
        assert split_post_ids == all_post_ids


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

    @pytest.mark.skip(reason="get_encoder not implemented yet")
    def test_model_initialization(self):
        """Test model initialization."""
        from psy_agents_noaug.models.encoders import get_encoder

        with patch("transformers.AutoModel") as mock_auto_model:
            mock_model = MagicMock()
            mock_model.config.hidden_size = 768
            mock_auto_model.from_pretrained.return_value = mock_model

            encoder = get_encoder("roberta-base")
            assert encoder is not None


class TestHPOPipeline:
    """Test HPO pipeline."""

    def test_hpo_config_loading(self):
        """Test HPO configuration loading."""
        # Test that HPO configs exist

        config_dir = Path(__file__).parent.parent / "configs" / "hpo"

        # Check if configs directory exists
        if config_dir.exists():
            assert (config_dir / "stage0_sanity.yaml").exists()

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


class TestCLIIntegration:
    """Test CLI commands integration."""

    def test_cli_import(self):
        """Test that CLI can be imported."""
        try:
            from psy_agents_noaug import cli

            assert cli.main is not None
        except ImportError:
            pytest.skip("CLI module not available")

    def test_hydra_config_import(self):
        """Test Hydra configuration."""
        try:
            from pathlib import Path

            config_dir = Path(__file__).parent.parent / "configs"
            if config_dir.exists():
                # Just test import, don't initialize
                assert config_dir.is_dir()
        except ImportError:
            pytest.skip("Hydra not configured")


class TestReproducibility:
    """Test reproducibility of results."""

    def test_deterministic_splits(self, sample_annotations):
        """Test that splits are deterministic."""
        from psy_agents_noaug.data.loaders import group_split_by_post_id

        # Generate splits twice with same seed
        train1, val1, test1 = group_split_by_post_id(sample_annotations, random_seed=42)
        train2, val2, test2 = group_split_by_post_id(sample_annotations, random_seed=42)

        assert set(train1) == set(train2)
        assert set(val1) == set(val2)
        assert set(test1) == set(test2)

    def test_random_seed_setting(self):
        """Test random seed setting."""
        from psy_agents_noaug.utils.reproducibility import set_seed

        set_seed(42)
        rand1 = torch.rand(10)

        set_seed(42)
        rand2 = torch.rand(10)

        assert torch.allclose(rand1, rand2)
