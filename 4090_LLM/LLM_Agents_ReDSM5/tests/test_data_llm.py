"""
Test data loading and preprocessing for LLM text classification.
"""

import pytest
import pandas as pd
from pathlib import Path


def test_load_label_list(label_list):
    """Test that label list loads correctly from fixtures."""
    assert len(label_list) == 9
    assert 'depressed_mood' in label_list
    assert 'suicidality' in label_list


def test_collator_import():
    """Test that MultiLabelDataCollator can be imported."""
    from src.data import MultiLabelDataCollator
    assert MultiLabelDataCollator is not None


def test_synthetic_data_generation(synthetic_data_path):
    """Test that synthetic data is generated correctly."""
    train_path = synthetic_data_path / "train.jsonl"
    assert train_path.exists()

    # Read first line
    df = pd.read_json(train_path, lines=True)
    assert len(df) > 0
    assert 'text' in df.columns

    # Check that label columns exist
    label_names = [
        "depressed_mood", "diminished_interest", "weight_appetite_change",
        "sleep_disturbance", "psychomotor", "fatigue",
        "worthlessness_guilt", "concentration_indecision", "suicidality"
    ]
    for label in label_names:
        assert label in df.columns
