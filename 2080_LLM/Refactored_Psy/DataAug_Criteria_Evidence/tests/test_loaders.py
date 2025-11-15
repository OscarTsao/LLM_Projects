"""Tests for data loaders with STRICT validation."""

import json
from pathlib import Path

import pandas as pd
import pytest

from psy_agents_aug.data.loaders import (
    DSMCriteriaLoader,
    ReDSM5DataLoader,
    group_split_by_post_id,
    load_splits_json,
    save_splits_json,
)


@pytest.fixture
def sample_dsm_criteria(tmp_path):
    """Create sample DSM criteria JSON."""
    criteria = [
        {"id": "A", "text": "Criterion A description"},
        {"id": "B", "text": "Criterion B description"},
        {"id": "C", "text": "Criterion C description"},
    ]
    
    dsm_path = tmp_path / "dsm_criteria.json"
    with open(dsm_path, "w") as f:
        json.dump(criteria, f)
    
    return dsm_path


@pytest.fixture
def sample_field_map():
    """Create sample field map."""
    return {
        'posts': {
            'post_id': 'post_id',
            'text': 'text'
        },
        'annotations': {
            'post_id': 'post_id',
            'criterion_id': 'criterion_id',
            'status': 'status',
            'cases': 'cases'
        },
        'validation': {
            'strict_mode': True
        }
    }


@pytest.fixture
def sample_csv_data(tmp_path, sample_field_map):
    """Create sample CSV data."""
    # Create posts.csv
    posts = pd.DataFrame({
        "post_id": ["post1", "post2", "post3"],
        "text": ["Sample text 1", "Sample text 2", "Sample text 3"],
    })
    posts.to_csv(tmp_path / "posts.csv", index=False)
    
    # Create annotations.csv
    annotations = pd.DataFrame({
        "post_id": ["post1", "post1", "post2", "post3"],
        "criterion_id": ["A", "B", "A", "C"],
        "status": ["positive", "negative", "1", "0"],
        "cases": ['[{"text": "ev1"}]', '[]', '[{"text": "ev2"}]', '[]'],
    })
    annotations.to_csv(tmp_path / "annotations.csv", index=False)
    
    return tmp_path


def test_dsm_criteria_loader(sample_dsm_criteria):
    """Test DSM criteria loader."""
    loader = DSMCriteriaLoader(sample_dsm_criteria)
    
    criteria = loader.load_criteria()
    assert len(criteria) == 3, "Should load all criteria"
    assert all("id" in c and "text" in c for c in criteria), "All criteria should have id and text"


def test_dsm_criteria_loader_get_by_id(sample_dsm_criteria):
    """Test getting criterion by ID."""
    loader = DSMCriteriaLoader(sample_dsm_criteria)
    
    criterion = loader.get_criterion_by_id("A")
    assert criterion is not None, "Should find criterion A"
    assert criterion["id"] == "A", "Should return correct criterion"
    
    criterion = loader.get_criterion_by_id("X")
    assert criterion is None, "Should return None for non-existent ID"


def test_dsm_criteria_loader_get_all_ids(sample_dsm_criteria):
    """Test getting all criterion IDs."""
    loader = DSMCriteriaLoader(sample_dsm_criteria)
    
    ids = loader.get_all_criterion_ids()
    assert ids == ["A", "B", "C"], "Should return all criterion IDs"


def test_redsm5_loader_local(sample_csv_data, sample_field_map):
    """Test ReDSM5DataLoader with local CSV files."""
    loader = ReDSM5DataLoader(
        field_map=sample_field_map,
        data_source='local',
        data_dir=sample_csv_data
    )
    
    # Test loading posts
    posts = loader.load_posts()
    assert len(posts) == 3, "Should load all posts"
    assert "post_id" in posts.columns, "Should have post_id column"
    assert "text" in posts.columns, "Should have text column"
    
    # Test loading annotations
    annotations = loader.load_annotations()
    assert len(annotations) == 4, "Should load all annotations"
    assert "post_id" in annotations.columns
    assert "criterion_id" in annotations.columns
    assert "status" in annotations.columns
    assert "cases" in annotations.columns


def test_redsm5_loader_validates_required_columns(tmp_path, sample_field_map):
    """Test that loader validates required columns."""
    # Create posts.csv missing required column
    posts = pd.DataFrame({
        "post_id": ["post1"],
        # Missing 'text' column
    })
    posts.to_csv(tmp_path / "posts.csv", index=False)
    
    annotations = pd.DataFrame({
        "post_id": ["post1"],
        "criterion_id": ["A"],
        "status": ["positive"],
        "cases": ["[]"],
    })
    annotations.to_csv(tmp_path / "annotations.csv", index=False)
    
    loader = ReDSM5DataLoader(
        field_map=sample_field_map,
        data_source='local',
        data_dir=tmp_path
    )
    
    with pytest.raises(ValueError, match="Missing required columns"):
        loader.load_posts()


def test_redsm5_loader_dsm_criteria(sample_csv_data, sample_field_map, sample_dsm_criteria):
    """Test loading DSM criteria."""
    loader = ReDSM5DataLoader(
        field_map=sample_field_map,
        data_source='local',
        data_dir=sample_csv_data
    )
    
    criteria = loader.load_dsm_criteria(sample_dsm_criteria)
    assert len(criteria) == 3
    
    valid_ids = loader.get_valid_criterion_ids(sample_dsm_criteria)
    assert valid_ids == {"A", "B", "C"}


def test_group_split_by_post_id():
    """Test post_id-based splitting."""
    df = pd.DataFrame({
        "post_id": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p5"],
        "data": range(8)
    })
    
    train_ids, val_ids, test_ids = group_split_by_post_id(
        df, post_id_col="post_id", random_seed=42
    )
    
    # Check no overlap
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    
    assert len(train_set & val_set) == 0, "No overlap between train and val"
    assert len(train_set & test_set) == 0, "No overlap between train and test"
    assert len(val_set & test_set) == 0, "No overlap between val and test"
    
    # Check all post_ids accounted for
    all_ids = train_set | val_set | test_set
    assert all_ids == set(df["post_id"].unique())


def test_group_split_deterministic():
    """Test that splits are deterministic with same seed."""
    df = pd.DataFrame({
        "post_id": [f"p{i}" for i in range(20)],
        "data": range(20)
    })
    
    # Split twice with same seed
    train1, val1, test1 = group_split_by_post_id(df, random_seed=42)
    train2, val2, test2 = group_split_by_post_id(df, random_seed=42)
    
    assert set(train1) == set(train2)
    assert set(val1) == set(val2)
    assert set(test1) == set(test2)


def test_save_and_load_splits(tmp_path):
    """Test saving and loading splits JSON."""
    train_ids = ['p1', 'p2', 'p3']
    val_ids = ['p4', 'p5']
    test_ids = ['p6', 'p7']
    
    splits_path = tmp_path / 'splits.json'
    
    # Save splits
    save_splits_json(
        train_post_ids=train_ids,
        val_post_ids=val_ids,
        test_post_ids=test_ids,
        output_path=splits_path,
        metadata={'random_seed': 42}
    )
    
    assert splits_path.exists()
    
    # Load splits
    loaded_train, loaded_val, loaded_test = load_splits_json(splits_path)
    
    assert loaded_train == train_ids
    assert loaded_val == val_ids
    assert loaded_test == test_ids


def test_load_splits_detects_leakage(tmp_path):
    """Test that loading splits detects data leakage."""
    # Create splits with overlap (leakage)
    splits_data = {
        'train': ['p1', 'p2', 'p3'],
        'val': ['p3', 'p4'],  # p3 overlaps with train
        'test': ['p5', 'p6'],
        'metadata': {}
    }
    
    splits_path = tmp_path / 'splits.json'
    with open(splits_path, 'w') as f:
        json.dump(splits_data, f)
    
    with pytest.raises(ValueError, match="Data leakage detected"):
        load_splits_json(splits_path)


def test_split_ratios_validation():
    """Test that invalid ratios are rejected."""
    df = pd.DataFrame({
        "post_id": ["p1", "p2", "p3"],
        "data": [1, 2, 3]
    })
    
    # Ratios don't sum to 1.0
    with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
        group_split_by_post_id(
            df,
            train_ratio=0.5,
            val_ratio=0.3,
            test_ratio=0.3  # 0.5 + 0.3 + 0.3 = 1.1
        )
