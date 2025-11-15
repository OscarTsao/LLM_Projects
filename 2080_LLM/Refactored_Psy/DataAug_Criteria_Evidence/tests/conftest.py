"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch
import yaml


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    data_dir = tmp_path_factory.mktemp("test_data")
    return data_dir


@pytest.fixture(scope="session")
def mock_mlflow():
    """Mock MLflow for testing without logging."""
    import mlflow
    
    # Store original functions
    original_start_run = mlflow.start_run
    original_log_param = mlflow.log_param
    original_log_metric = mlflow.log_metric
    original_log_artifact = mlflow.log_artifact
    
    # Create mocks
    mlflow.start_run = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    mlflow.log_param = MagicMock()
    mlflow.log_metric = MagicMock()
    mlflow.log_artifact = MagicMock()
    
    yield mlflow
    
    # Restore original functions
    mlflow.start_run = original_start_run
    mlflow.log_param = original_log_param
    mlflow.log_metric = original_log_metric
    mlflow.log_artifact = original_log_artifact


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Create a sample configuration for testing."""
    return {
        "task": {
            "name": "criteria",
            "num_classes": 2,
            "label_field": "label",
        },
        "model": {
            "name": "roberta_base",
            "pretrained_name": "roberta-base",
            "dropout": 0.1,
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 2e-5,
            "num_epochs": 3,
            "max_grad_norm": 1.0,
            "warmup_steps": 100,
        },
        "data": {
            "max_length": 128,
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
        },
        "augmentation": {
            "enabled": True,
            "ratio": 0.5,
            "max_aug_per_sample": 1,
            "seed": 42,
            "train_only": True,
        },
    }


@pytest.fixture
def sample_posts() -> pd.DataFrame:
    """Create sample posts DataFrame."""
    return pd.DataFrame({
        "post_id": ["post1", "post2", "post3", "post4", "post5"],
        "text": [
            "Patient reports anxiety symptoms.",
            "No significant psychiatric history.",
            "Depression and mood disturbances noted.",
            "Patient appears stable.",
            "Severe anxiety and panic attacks.",
        ],
    })


@pytest.fixture
def sample_annotations() -> pd.DataFrame:
    """Create sample annotations DataFrame."""
    return pd.DataFrame({
        "post_id": ["post1", "post1", "post2", "post3", "post4", "post5"],
        "criterion_id": ["A", "B", "A", "C", "A", "B"],
        "status": ["positive", "negative", "negative", "positive", "negative", "positive"],
        "cases": [
            '[{"text": "anxiety symptoms", "start_char": 15, "end_char": 31}]',
            "[]",
            "[]",
            '[{"text": "Depression", "start_char": 0, "end_char": 10}]',
            "[]",
            '[{"text": "panic attacks", "start_char": 20, "end_char": 33}]',
        ],
    })


@pytest.fixture
def valid_criterion_ids():
    """Return valid criterion IDs."""
    return {"A", "B", "C"}


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    tokenizer.model_max_length = 512
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = MagicMock()
    model.return_value = MagicMock(
        logits=torch.randn(1, 2),
        loss=torch.tensor(0.5),
    )
    model.eval = MagicMock()
    model.train = MagicMock()
    return model


@pytest.fixture
def temp_config_file(tmp_path, sample_config):
    """Create a temporary config file."""
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config, f)
    return config_file


@pytest.fixture
def augmentation_config():
    """Create augmentation configuration."""
    return {
        "enabled": True,
        "ratio": 0.5,
        "max_aug_per_sample": 1,
        "seed": 42,
        "train_only": True,
    }


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test."""
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def mock_dataset():
    """Create a mock dataset."""
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)
    dataset.__getitem__ = MagicMock(return_value={
        "input_ids": torch.tensor([1, 2, 3, 4, 5]),
        "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
        "labels": torch.tensor(1),
    })
    return dataset


@pytest.fixture(scope="function")
def clean_environment():
    """Clean environment variables before and after tests."""
    # Store original values
    original_env = dict(os.environ)
    
    # Clear MLflow and Optuna related env vars
    test_vars = [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "OPTUNA_STORAGE",
    ]
    
    for var in test_vars:
        os.environ.pop(var, None)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
