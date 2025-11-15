"""
Load social media posts related to mental health from a CSV file.
"""

import pandas as pd

def load_posts(file_path: str) -> list:
    """
    Load posts from a JSON file.
    """
    df = pd.read_csv(file_path)
    posts = df['translated_post'].tolist()
    return posts