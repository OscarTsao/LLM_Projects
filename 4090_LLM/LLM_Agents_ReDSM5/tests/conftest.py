"""
Pytest configuration and shared fixtures for ReDSM5 test suite.
"""

import os
import sys
from pathlib import Path
from typing import List

import pytest
import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def tiny_model_id():
    """Return a tiny LLM model ID for CPU testing."""
    return "hf-internal-testing/tiny-random-LlamaForSequenceClassification"


@pytest.fixture
def label_list():
    """Load label list from test fixtures."""
    labels_path = Path(__file__).parent / "fixtures" / "labels.yaml"
    with open(labels_path) as f:
        config = yaml.safe_load(f)
    labels = config.get("labels", [])
    drop_labels = set(config.get("drop_labels", []))
    return [label for label in labels if label not in drop_labels]


@pytest.fixture
def synthetic_data_path(tmp_path):
    """Create synthetic test data and return the directory path."""
    from tests.fixtures.data import generate_synthetic_dataset

    data_dir = tmp_path / "synthetic_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate 100 synthetic samples
    generate_synthetic_dataset(data_dir, num_samples=100, seed=42)

    return data_dir


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _is_bitsandbytes_available():
    """Check if bitsandbytes is available."""
    try:
        import bitsandbytes
        return True
    except ImportError:
        return False


# Conditional skip markers
skip_if_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

skip_if_no_bitsandbytes = pytest.mark.skipif(
    not _is_bitsandbytes_available(),
    reason="bitsandbytes not available"
)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "cuda: marks tests that require CUDA")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "smoke: marks tests as smoke tests")
