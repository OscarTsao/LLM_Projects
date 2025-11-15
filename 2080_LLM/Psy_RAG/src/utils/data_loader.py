"""
Data loading utilities for posts and criteria
"""
# Standard library
import json
import logging
from pathlib import Path
from typing import List, Dict

# Third-party
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of posts and criteria data"""
    
    def __init__(self, posts_path: Path, criteria_path: Path):
        self.posts_path = posts_path
        self.criteria_path = criteria_path
        
    def load_posts(self) -> pd.DataFrame:
        """Load translated posts from CSV"""
        try:
            df = pd.read_csv(self.posts_path)
            logger.info(f"Loaded {len(df)} posts from {self.posts_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading posts: {e}")
            raise
    
    def load_criteria(self) -> List[Dict]:
        """Load DSM-5 criteria from JSON"""
        try:
            with open(self.criteria_path, 'r', encoding='utf-8') as f:
                criteria_data = json.load(f)
            logger.info(f"Loaded {len(criteria_data)} disorders with criteria")
            return criteria_data
        except Exception as e:
            logger.error(f"Error loading criteria: {e}")
            raise
    
    def preprocess_posts(self, df: pd.DataFrame) -> List[Dict]:
        """Preprocess posts for embedding"""
        processed_posts = []

        # Check if expected column exists
        if 'translated_post' not in df.columns:
            # Try alternative column names
            post_column = None
            for col in ['post', 'positive_post', 'post_text', 'text', 'content', 'message']:
                if col in df.columns:
                    post_column = col
                    break

            if post_column is None:
                logger.error(f"No suitable text column found. Available columns: {df.columns.tolist()}")
                return []

            logger.info(f"Using column '{post_column}' as post text")
        else:
            post_column = 'translated_post'
        
        for idx, row in df.iterrows():
            post_text = row[post_column]
            if pd.isna(post_text) or not isinstance(post_text, str):
                continue
                
            # Clean and truncate text
            post_text = post_text.strip()
            if len(post_text) < 5:  # Skip very short posts
                continue
                
            if len(post_text) > 512:  # Truncate long posts
                post_text = post_text[:512] + "..."
            
            processed_posts.append({
                'id': idx,
                'text': post_text,
                'source_file': row.get('source_file', 'unknown'),
                'post_context': row.get('post_context', '')
            })
        
        logger.info(f"Preprocessed {len(processed_posts)} posts")
        return processed_posts
    
    def preprocess_criteria(self, criteria_data: List[Dict]) -> List[Dict]:
        """Preprocess criteria for embedding"""
        processed_criteria = []
        
        for disorder in criteria_data:
            diagnosis = disorder['diagnosis']
            for criterion in disorder['criteria']:
                criterion_text = criterion['text']
                if len(criterion_text) > 256:  # Truncate long criteria
                    criterion_text = criterion_text[:256] + "..."
                
                processed_criteria.append({
                    'id': f"{diagnosis}_{criterion['id']}",
                    'diagnosis': diagnosis,
                    'criterion_id': criterion['id'],
                    'text': criterion_text,
                    'full_text': criterion['text']
                })
        
        logger.info(f"Preprocessed {len(processed_criteria)} criteria")
        return processed_criteria
    
    def get_texts_for_embedding(self, data: List[Dict], text_field: str = 'text') -> List[str]:
        """Extract texts for embedding"""
        return [item[text_field] for item in data]
    
    def create_text_id_mapping(self, data: List[Dict]) -> Dict[int, str]:
        """Create mapping from index to ID"""
        return {idx: item['id'] for idx, item in enumerate(data)}
