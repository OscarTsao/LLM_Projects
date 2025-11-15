"""
Pytest configuration and fixtures
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import json


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_posts_data():
    """Sample posts data for testing"""
    return {
        'post_context': ['Test context 1', 'Test context 2'],
        'source_file': ['test1.csv', 'test2.csv'],
        'translated_post': [
            'I feel very depressed and hopeless about my future.',
            'I have been experiencing anxiety and panic attacks recently.'
        ],
        'eval_len_ratio': [0.8, 0.9],
        'eval_flag_len_short': [False, False],
        'eval_flag_number_mismatch': [False, False],
        'eval_flag_paren_mismatch': [False, False],
        'eval_tgt_alpha_pct': [0.95, 0.92],
        'eval_flag_alpha_high': [False, False]
    }


@pytest.fixture
def sample_criteria_data():
    """Sample criteria data for testing"""
    return [
        {
            "diagnosis": "Major Depressive Disorder",
            "criteria": [
                {
                    "id": "A.1",
                    "text": "Depressed mood most of the day, nearly every day"
                },
                {
                    "id": "A.2", 
                    "text": "Markedly diminished interest or pleasure in activities"
                }
            ]
        },
        {
            "diagnosis": "Generalized Anxiety Disorder",
            "criteria": [
                {
                    "id": "A",
                    "text": "Excessive anxiety and worry occurring more days than not"
                }
            ]
        }
    ]


@pytest.fixture
def sample_posts_file(temp_dir, sample_posts_data):
    """Create a sample posts CSV file"""
    posts_file = temp_dir / "test_posts.csv"
    pd.DataFrame(sample_posts_data).to_csv(posts_file, index=False)
    return posts_file


@pytest.fixture
def sample_criteria_file(temp_dir, sample_criteria_data):
    """Create a sample criteria JSON file"""
    criteria_file = temp_dir / "test_criteria.json"
    with open(criteria_file, 'w') as f:
        json.dump(sample_criteria_data, f)
    return criteria_file


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing"""
    from unittest.mock import Mock
    mock_model = Mock()
    mock_model.encode_single.return_value = [0.1] * 1024
    mock_model.encode_texts.return_value = [[0.1] * 1024] * 2
    mock_model.get_embedding_dimension.return_value = 1024
    return mock_model


@pytest.fixture
def mock_spanbert_model():
    """Mock SpanBERT model for testing"""
    from unittest.mock import Mock
    from src.models.spanbert_model import SpanResult
    
    mock_model = Mock()
    mock_model.filter_criteria_matches.return_value = [
        ("Depressed mood most of the day", 0.8, [SpanResult("depressed", 0, 9, 0.9, "relevant")])
    ]
    mock_model.extract_spans.return_value = [SpanResult("depressed", 0, 9, 0.9, "relevant")]
    return mock_model
