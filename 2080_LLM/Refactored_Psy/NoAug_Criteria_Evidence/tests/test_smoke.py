"""Smoke tests to verify basic functionality.

Quick tests that verify:
- All modules can be imported
- Basic operations work
- System is functional
"""

import pytest
import torch


class TestImports:
    """Test that all modules can be imported."""

    def test_import_main_package(self):
        """Test main package import."""
        import psy_agents_noaug
        assert psy_agents_noaug is not None

    def test_import_data_modules(self):
        """Test data module imports."""
        from psy_agents_noaug.data import datasets, groundtruth, loaders
        assert datasets is not None
        assert groundtruth is not None
        assert loaders is not None

    def test_import_model_modules(self):
        """Test model module imports."""
        from psy_agents_noaug.models import encoders
        assert encoders is not None

    def test_import_training_modules(self):
        """Test training module imports."""
        from psy_agents_noaug.training import evaluate, train_loop
        assert evaluate is not None
        assert train_loop is not None

    def test_import_utils_modules(self):
        """Test utils module imports."""
        from psy_agents_noaug.utils import logging, reproducibility
        assert logging is not None
        assert reproducibility is not None


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


class TestConfiguration:
    """Test configuration loading."""

    def test_config_directory_exists(self):
        """Test that config directory exists."""
        from pathlib import Path

        config_dir = Path(__file__).parent.parent / "configs"
        assert config_dir.exists(), "Config directory should exist"

    def test_main_config_exists(self):
        """Test that main config exists."""
        from pathlib import Path

        main_config = Path(__file__).parent.parent / "configs" / "config.yaml"
        assert main_config.exists(), "Main config.yaml should exist"


class TestDataLoading:
    """Test basic data loading."""

    def test_load_sample_data(self, sample_posts, sample_annotations):
        """Test loading sample data."""
        assert len(sample_posts) > 0
        assert len(sample_annotations) > 0
        assert "post_id" in sample_posts.columns
        assert "post_id" in sample_annotations.columns


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
        from psy_agents_noaug.utils.reproducibility import set_seed

        set_seed(42)
        rand1 = torch.rand(5)

        set_seed(42)
        rand2 = torch.rand(5)

        assert torch.allclose(rand1, rand2)

    def test_logging_setup(self):
        """Test logging setup."""
        from psy_agents_noaug.utils.logging import get_logger

        logger = get_logger(__name__)
        assert logger is not None


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
