"""Tests for ground truth generation with STRICT validation.

STRICT RULES:
1. status field -> ONLY for criteria task
2. cases field -> ONLY for evidence task
3. NO cross-contamination allowed
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from psy_agents_aug.data.groundtruth import (
    _assert_field_usage,
    create_criteria_groundtruth,
    create_evidence_groundtruth,
    load_field_map,
    normalize_status_value,
    parse_cases_field,
    validate_strict_separation,
    GroundTruthValidator,
)


@pytest.fixture
def field_map_path(tmp_path):
    """Create temporary field_map.yaml."""
    field_map = {
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
        'status_values': {
            'positive': ['positive', 'present', 'true', '1', 1, True],
            'negative': ['negative', 'absent', 'false', '0', 0, False]
        },
        'cases_structure': {
            'is_list': True,
            'fields': {
                'text': 'text',
                'start_char': 'start_char',
                'end_char': 'end_char',
                'sentence_id': 'sentence_id'
            }
        },
        'validation': {
            'strict_mode': True,
            'allow_cross_contamination': False,
            'fail_on_invalid_criterion_id': True,
            'fail_on_missing_post_id': True,
            'drop_duplicates': True
        }
    }
    
    import yaml
    map_path = tmp_path / 'field_map.yaml'
    with open(map_path, 'w') as f:
        yaml.dump(field_map, f)
    
    return map_path


@pytest.fixture
def sample_posts():
    """Create sample posts DataFrame."""
    return pd.DataFrame({
        'post_id': ['post1', 'post2', 'post3'],
        'text': ['Sample text 1', 'Sample text 2', 'Sample text 3']
    })


@pytest.fixture
def sample_annotations():
    """Create sample annotations DataFrame."""
    return pd.DataFrame({
        'post_id': ['post1', 'post1', 'post2', 'post3'],
        'criterion_id': ['A', 'B', 'A', 'C'],
        'status': ['positive', 'negative', '1', 0],
        'cases': [
            '[{"text": "evidence1", "start_char": 0, "end_char": 9}]',
            '[]',
            '[{"text": "evidence2", "start_char": 5, "end_char": 14}]',
            []
        ]
    })


@pytest.fixture
def valid_criterion_ids():
    """Set of valid criterion IDs."""
    return {'A', 'B', 'C'}


def test_assert_field_usage():
    """Test strict field usage assertion."""
    # Should pass
    _assert_field_usage('status', 'status', 'Test operation')
    
    # Should fail
    with pytest.raises(AssertionError, match='STRICT VALIDATION FAILURE'):
        _assert_field_usage('wrong_field', 'status', 'Test operation')


def test_normalize_status_value(field_map_path):
    """Test status value normalization."""
    field_map = load_field_map(field_map_path)
    status_map = field_map['status_values']
    
    # Test positive values
    assert normalize_status_value('positive', status_map) == 1
    assert normalize_status_value('1', status_map) == 1
    assert normalize_status_value(1, status_map) == 1
    assert normalize_status_value(True, status_map) == 1
    
    # Test negative values
    assert normalize_status_value('negative', status_map) == 0
    assert normalize_status_value('0', status_map) == 0
    assert normalize_status_value(0, status_map) == 0
    assert normalize_status_value(False, status_map) == 0
    
    # Test invalid values
    assert normalize_status_value('invalid', status_map) is None
    assert normalize_status_value(None, status_map) is None


def test_parse_cases_field():
    """Test cases field parsing."""
    # Test empty cases
    assert parse_cases_field(None) == []
    assert parse_cases_field('[]') == []
    assert parse_cases_field([]) == []
    
    # Test list input
    cases_list = [{'text': 'evidence', 'start_char': 0}]
    assert parse_cases_field(cases_list) == cases_list
    
    # Test JSON string
    cases_json = '[{"text": "evidence", "start_char": 0}]'
    result = parse_cases_field(cases_json)
    assert len(result) == 1
    assert result[0]['text'] == 'evidence'


def test_create_criteria_groundtruth(
    sample_posts, sample_annotations, field_map_path, valid_criterion_ids
):
    """Test criteria groundtruth creation."""
    field_map = load_field_map(field_map_path)
    
    criteria_gt = create_criteria_groundtruth(
        annotations=sample_annotations,
        posts=sample_posts,
        field_map=field_map,
        valid_criterion_ids=valid_criterion_ids
    )
    
    # Check structure
    assert set(criteria_gt.columns) == {'post_id', 'criterion_id', 'status', 'label'}
    
    # Check values
    assert len(criteria_gt) == 4  # All annotations should be included
    assert set(criteria_gt['label'].unique()) == {0, 1}
    
    # STRICT: Verify no evidence fields
    assert 'cases' not in criteria_gt.columns
    assert 'evidence_text' not in criteria_gt.columns


def test_create_criteria_groundtruth_enforces_status_field(
    sample_posts, sample_annotations, field_map_path, valid_criterion_ids
):
    """Test that criteria groundtruth enforces using status field."""
    field_map = load_field_map(field_map_path)
    
    # Modify field_map to use wrong field - should raise AssertionError
    field_map['annotations']['status'] = 'wrong_field'
    
    with pytest.raises(AssertionError, match='STRICT VALIDATION FAILURE'):
        create_criteria_groundtruth(
            annotations=sample_annotations,
            posts=sample_posts,
            field_map=field_map,
            valid_criterion_ids=valid_criterion_ids
        )


def test_create_evidence_groundtruth(
    sample_posts, sample_annotations, field_map_path, valid_criterion_ids
):
    """Test evidence groundtruth creation."""
    field_map = load_field_map(field_map_path)
    
    evidence_gt = create_evidence_groundtruth(
        annotations=sample_annotations,
        posts=sample_posts,
        field_map=field_map,
        valid_criterion_ids=valid_criterion_ids
    )
    
    # Check structure
    required_cols = {'post_id', 'criterion_id', 'case_id', 'evidence_text', 
                     'start_char', 'end_char', 'sentence_id'}
    assert required_cols.issubset(set(evidence_gt.columns))
    
    # STRICT: Verify no criteria fields
    assert 'status' not in evidence_gt.columns
    assert 'label' not in evidence_gt.columns
    
    # Check that only posts with cases are included
    assert len(evidence_gt) >= 2  # At least post1 and post2 have evidence


def test_create_evidence_groundtruth_enforces_cases_field(
    sample_posts, sample_annotations, field_map_path, valid_criterion_ids
):
    """Test that evidence groundtruth enforces using cases field."""
    field_map = load_field_map(field_map_path)
    
    # Modify field_map to use wrong field - should raise AssertionError
    field_map['annotations']['cases'] = 'wrong_field'
    
    with pytest.raises(AssertionError, match='STRICT VALIDATION FAILURE'):
        create_evidence_groundtruth(
            annotations=sample_annotations,
            posts=sample_posts,
            field_map=field_map,
            valid_criterion_ids=valid_criterion_ids
        )


def test_validate_strict_separation(
    sample_posts, sample_annotations, field_map_path, valid_criterion_ids
):
    """Test strict separation validation."""
    field_map = load_field_map(field_map_path)
    
    criteria_gt = create_criteria_groundtruth(
        annotations=sample_annotations,
        posts=sample_posts,
        field_map=field_map,
        valid_criterion_ids=valid_criterion_ids
    )
    
    evidence_gt = create_evidence_groundtruth(
        annotations=sample_annotations,
        posts=sample_posts,
        field_map=field_map,
        valid_criterion_ids=valid_criterion_ids
    )
    
    # Should pass
    validate_strict_separation(criteria_gt, evidence_gt, field_map)


def test_validate_strict_separation_detects_contamination(field_map_path):
    """Test that validation detects field contamination."""
    field_map = load_field_map(field_map_path)
    
    # Criteria with evidence field (contaminated)
    criteria_contaminated = pd.DataFrame({
        'post_id': ['post1'],
        'criterion_id': ['A'],
        'status': ['positive'],
        'label': [1],
        'evidence_text': ['contamination']  # Should not be here
    })
    
    evidence_clean = pd.DataFrame({
        'post_id': ['post1'],
        'criterion_id': ['A'],
        'case_id': [0],
        'evidence_text': ['evidence'],
        'start_char': [0],
        'end_char': [8],
        'sentence_id': [0]
    })
    
    with pytest.raises(AssertionError, match='STRICT VIOLATION'):
        validate_strict_separation(criteria_contaminated, evidence_clean, field_map)


def test_groundtruth_validator(field_map_path, valid_criterion_ids):
    """Test GroundTruthValidator class."""
    field_map = load_field_map(field_map_path)
    validator = GroundTruthValidator(field_map, valid_criterion_ids)
    
    # Valid criteria groundtruth
    criteria_df = pd.DataFrame({
        'post_id': ['post1'],
        'criterion_id': ['A'],
        'status': ['positive'],
        'label': [1]
    })
    
    result = validator.validate_criteria_groundtruth(criteria_df)
    assert len(result['errors']) == 0
    
    # Contaminated criteria groundtruth
    criteria_contaminated = pd.DataFrame({
        'post_id': ['post1'],
        'criterion_id': ['A'],
        'status': ['positive'],
        'label': [1],
        'cases': [[]]  # Should not be here
    })
    
    result = validator.validate_criteria_groundtruth(criteria_contaminated)
    assert len(result['errors']) > 0
    assert any('STRICT VIOLATION' in err for err in result['errors'])


def test_invalid_criterion_ids_rejected(
    sample_posts, sample_annotations, field_map_path
):
    """Test that invalid criterion IDs are rejected."""
    field_map = load_field_map(field_map_path)
    
    # Add invalid criterion ID
    bad_annotations = sample_annotations.copy()
    bad_annotations.loc[0, 'criterion_id'] = 'INVALID'
    
    valid_ids = {'A', 'B', 'C'}
    
    with pytest.raises(ValueError, match='Invalid criterion IDs'):
        create_criteria_groundtruth(
            annotations=bad_annotations,
            posts=sample_posts,
            field_map=field_map,
            valid_criterion_ids=valid_ids
        )


def test_missing_post_ids_rejected(
    sample_posts, sample_annotations, field_map_path, valid_criterion_ids
):
    """Test that missing post IDs are rejected."""
    field_map = load_field_map(field_map_path)
    
    # Add annotation for non-existent post
    bad_annotations = sample_annotations.copy()
    bad_annotations.loc[len(bad_annotations)] = {
        'post_id': 'nonexistent',
        'criterion_id': 'A',
        'status': 'positive',
        'cases': '[]'
    }
    
    with pytest.raises(ValueError, match='Post IDs.*not found'):
        create_criteria_groundtruth(
            annotations=bad_annotations,
            posts=sample_posts,
            field_map=field_map,
            valid_criterion_ids=valid_criterion_ids
        )


def test_deterministic_splits(sample_annotations):
    """Test that splits are deterministic with same seed."""
    from psy_agents_aug.data.loaders import group_split_by_post_id
    
    # Generate splits twice with same seed
    train1, val1, test1 = group_split_by_post_id(
        sample_annotations, random_seed=42
    )
    train2, val2, test2 = group_split_by_post_id(
        sample_annotations, random_seed=42
    )
    
    # Should be identical
    assert set(train1) == set(train2)
    assert set(val1) == set(val2)
    assert set(test1) == set(test2)


def test_no_data_leakage_in_splits(sample_annotations):
    """Test that splits have no overlapping post_ids."""
    from psy_agents_aug.data.loaders import group_split_by_post_id
    
    train, val, test = group_split_by_post_id(
        sample_annotations, random_seed=42
    )
    
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)
    
    # No overlap
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0
