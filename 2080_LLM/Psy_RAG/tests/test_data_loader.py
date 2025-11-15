"""
Tests for data loader functionality
"""
import pytest
import pandas as pd
import json
from pathlib import Path
import tempfile
import os

from src.utils.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class"""
    
    def setup_method(self):
        """Setup test data"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test posts CSV
        self.test_posts_data = {
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
        
        self.test_posts_df = pd.DataFrame(self.test_posts_data)
        self.posts_path = Path(self.temp_dir) / "test_posts.csv"
        self.test_posts_df.to_csv(self.posts_path, index=False)
        
        # Create test criteria JSON
        self.test_criteria_data = [
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
        
        self.criteria_path = Path(self.temp_dir) / "test_criteria.json"
        with open(self.criteria_path, 'w') as f:
            json.dump(self.test_criteria_data, f)
        
        # Initialize DataLoader
        self.data_loader = DataLoader(self.posts_path, self.criteria_path)
    
    def teardown_method(self):
        """Clean up test data"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_posts(self):
        """Test loading posts from CSV"""
        df = self.data_loader.load_posts()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'translated_post' in df.columns
        assert df['translated_post'].iloc[0] == 'I feel very depressed and hopeless about my future.'
    
    def test_load_criteria(self):
        """Test loading criteria from JSON"""
        criteria_data = self.data_loader.load_criteria()
        
        assert isinstance(criteria_data, list)
        assert len(criteria_data) == 2
        assert criteria_data[0]['diagnosis'] == 'Major Depressive Disorder'
        assert len(criteria_data[0]['criteria']) == 2
    
    def test_preprocess_posts(self):
        """Test preprocessing posts"""
        df = self.data_loader.load_posts()
        processed_posts = self.data_loader.preprocess_posts(df)
        
        assert isinstance(processed_posts, list)
        assert len(processed_posts) == 2
        assert 'id' in processed_posts[0]
        assert 'text' in processed_posts[0]
        assert 'source_file' in processed_posts[0]
        assert processed_posts[0]['text'] == 'I feel very depressed and hopeless about my future.'
    
    def test_preprocess_criteria(self):
        """Test preprocessing criteria"""
        criteria_data = self.data_loader.load_criteria()
        processed_criteria = self.data_loader.preprocess_criteria(criteria_data)
        
        assert isinstance(processed_criteria, list)
        assert len(processed_criteria) == 3  # 2 + 1 criteria
        assert 'id' in processed_criteria[0]
        assert 'diagnosis' in processed_criteria[0]
        assert 'criterion_id' in processed_criteria[0]
        assert 'text' in processed_criteria[0]
        assert processed_criteria[0]['diagnosis'] == 'Major Depressive Disorder'
    
    def test_get_texts_for_embedding(self):
        """Test extracting texts for embedding"""
        df = self.data_loader.load_posts()
        processed_posts = self.data_loader.preprocess_posts(df)
        texts = self.data_loader.get_texts_for_embedding(processed_posts)
        
        assert isinstance(texts, list)
        assert len(texts) == 2
        assert texts[0] == 'I feel very depressed and hopeless about my future.'
    
    def test_create_text_id_mapping(self):
        """Test creating text ID mapping"""
        df = self.data_loader.load_posts()
        processed_posts = self.data_loader.preprocess_posts(df)
        mapping = self.data_loader.create_text_id_mapping(processed_posts)
        
        assert isinstance(mapping, dict)
        assert len(mapping) == 2
        assert mapping[0] == 0  # First post has ID 0
        assert mapping[1] == 1  # Second post has ID 1
    
    def test_handle_empty_posts(self):
        """Test handling empty or invalid posts"""
        # Create test data with empty posts
        empty_posts_data = {
            'post_context': ['Test context'],
            'source_file': ['test.csv'],
            'translated_post': [''],  # Empty post
            'eval_len_ratio': [0.8],
            'eval_flag_len_short': [False],
            'eval_flag_number_mismatch': [False],
            'eval_flag_paren_mismatch': [False],
            'eval_tgt_alpha_pct': [0.95],
            'eval_flag_alpha_high': [False]
        }
        
        empty_posts_df = pd.DataFrame(empty_posts_data)
        empty_posts_path = Path(self.temp_dir) / "empty_posts.csv"
        empty_posts_df.to_csv(empty_posts_path, index=False)
        
        empty_data_loader = DataLoader(empty_posts_path, self.criteria_path)
        df = empty_data_loader.load_posts()
        processed_posts = empty_data_loader.preprocess_posts(df)
        
        # Should filter out empty posts
        assert len(processed_posts) == 0
    
    def test_handle_long_texts(self):
        """Test handling long texts (truncation)"""
        # Create test data with long post
        long_text = "This is a very long post. " * 100  # Very long text
        long_posts_data = {
            'post_context': ['Test context'],
            'source_file': ['test.csv'],
            'translated_post': [long_text],
            'eval_len_ratio': [0.8],
            'eval_flag_len_short': [False],
            'eval_flag_number_mismatch': [False],
            'eval_flag_paren_mismatch': [False],
            'eval_tgt_alpha_pct': [0.95],
            'eval_flag_alpha_high': [False]
        }
        
        long_posts_df = pd.DataFrame(long_posts_data)
        long_posts_path = Path(self.temp_dir) / "long_posts.csv"
        long_posts_df.to_csv(long_posts_path, index=False)
        
        long_data_loader = DataLoader(long_posts_path, self.criteria_path)
        df = long_data_loader.load_posts()
        processed_posts = long_data_loader.preprocess_posts(df)
        
        # Should truncate long posts
        assert len(processed_posts) == 1
        assert len(processed_posts[0]['text']) <= 512 + 3  # 512 + "..."
        assert processed_posts[0]['text'].endswith("...")
