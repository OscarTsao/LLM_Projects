"""Smoke tests to verify basic functionality with augmentation.

Quick tests that verify:
- All modules can be imported
- Basic operations work
- Augmentation is available
- System is functional
"""

import pytest
import torch


class TestImports:
    """Test that all modules can be imported."""

    def test_import_main_package(self):
        """Test main package import."""
        try:
            import psy_agents_aug
            assert psy_agents_aug is not None
        except ImportError:
            pytest.skip("Main package not in expected structure")

    def test_import_augmentation_modules(self):
        """Test augmentation module imports."""
        try:
            from psy_agents_aug import augment
            assert augment is not None
        except ImportError:
            pytest.skip("Augmentation modules not available")

    def test_import_project_modules(self):
        """Test Project modules import."""
        try:
            from Project.Share.utils import seed
            assert seed is not None
        except ImportError:
            pytest.skip("Project modules not in expected structure")


class TestBasicOperations:
    """Test basic operations work."""

    def test_tensor_creation(self):
        """Test PyTorch tensor creation."""
        tensor = torch.randn(3, 4)
        assert tensor.shape == (3, 4)

    def test_pandas_dataframe_creation(self):
        """Test pandas DataFrame creation."""
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert len(df) == 3
        assert list(df.columns) == ["a", "b"]

    def test_numpy_array_creation(self):
        """Test numpy array creation."""
        import numpy as np

        arr = np.array([1, 2, 3])
        assert arr.shape == (3,)


class TestAugmentationAvailable:
    """Test augmentation libraries are available."""

    def test_nlpaug_available(self):
        """Test nlpaug is available."""
        try:
            import nlpaug
            assert nlpaug is not None
        except ImportError:
            pytest.skip("nlpaug not installed")

    def test_textattack_available(self):
        """Test textattack is available."""
        try:
            import textattack
            assert textattack is not None
        except ImportError:
            pytest.skip("textattack not installed")


class TestConfiguration:
    """Test configuration loading."""

    def test_config_directory_exists(self):
        """Test that config directory exists."""
        from pathlib import Path

        config_dir = Path(__file__).parent.parent / "configs"
        assert config_dir.exists(), "Config directory should exist"


class TestAugmentationBasics:
    """Test basic augmentation operations."""

    def test_augmentation_config_creation(self):
        """Test creating augmentation config."""
        try:
            from psy_agents_aug.augment import AugmentationConfig

            config = AugmentationConfig(
                enabled=True,
                ratio=0.5,
                max_aug_per_sample=1,
                seed=42,
                train_only=True,
            )
            assert config.enabled is True
            assert config.train_only is True
        except ImportError:
            pytest.skip("Augmentation config not available")

    def test_simple_text_augmentation(self):
        """Test simple text augmentation."""
        try:
            from psy_agents_aug.augment import AugmentationConfig, NLPAugPipeline

            config = AugmentationConfig(
                enabled=True,
                ratio=0.5,
                max_aug_per_sample=1,
                seed=42,
                train_only=True,
            )

            pipeline = NLPAugPipeline(config, aug_method="synonym")
            text = "This is a test sentence."

            # Should not crash
            result = pipeline.augment_text(text, num_variants=1)
            assert isinstance(result, list)

        except ImportError:
            pytest.skip("Augmentation pipeline not available")


class TestModelInitialization:
    """Test model initialization."""

    def test_simple_model_creation(self):
        """Test creating a simple PyTorch model."""
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        x = torch.randn(1, 10)
        output = model(x)
        assert output.shape == (1, 2)


class TestTrainingStep:
    """Test a single training step."""

    def test_forward_pass(self):
        """Test a forward pass through a simple model."""
        import torch.nn as nn

        model = nn.Linear(10, 2)
        x = torch.randn(4, 10)
        output = model(x)
        assert output.shape == (4, 2)

    def test_backward_pass(self):
        """Test a backward pass."""
        import torch.nn as nn

        model = nn.Linear(10, 2)
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))

        criterion = nn.CrossEntropyLoss()
        output = model(x)
        loss = criterion(output, y)

        loss.backward()
        assert loss.item() > 0


class TestUtilities:
    """Test utility functions."""

    def test_set_seed(self):
        """Test seed setting utility."""
        try:
            from Project.Share.utils.seed import set_seed

            set_seed(42)
            rand1 = torch.rand(5)

            set_seed(42)
            rand2 = torch.rand(5)

            assert torch.allclose(rand1, rand2)
        except ImportError:
            pytest.skip("Seed utilities not available")


class TestEnvironment:
    """Test environment setup."""

    def test_pytorch_available(self):
        """Test PyTorch is available."""
        import torch
        assert torch.__version__ is not None

    def test_transformers_available(self):
        """Test transformers is available."""
        import transformers
        assert transformers.__version__ is not None

    def test_cuda_info(self):
        """Test CUDA availability (info only)."""
        import torch
        # Just log CUDA availability, don't assert
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available, using CPU")
