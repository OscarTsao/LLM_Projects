"""
Test that all required LLM modules can be imported correctly.
"""

import pytest


def test_import_data():
    """Test data module imports."""
    from src import data
    assert hasattr(data, 'load_label_list')
    assert hasattr(data, 'prepare_datasets')
    assert hasattr(data, 'DatasetBundle')
    assert hasattr(data, 'MultiLabelDataCollator')


def test_import_losses():
    """Test losses module imports."""
    from src import losses
    assert hasattr(losses, 'build_loss_fn')


def test_import_metrics():
    """Test metrics module imports."""
    from src import metrics
    assert hasattr(metrics, 'compute_metrics_bundle')


def test_import_models():
    """Test models module imports."""
    from src import models
    assert hasattr(models, 'build_model')


def test_import_thresholds():
    """Test thresholds module imports."""
    from src import thresholds
    assert hasattr(thresholds, 'grid_search_thresholds')


def test_transformers_available():
    """Test that transformers library is available."""
    import transformers
    assert transformers.__version__ is not None
