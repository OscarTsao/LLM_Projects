"""
Statistics computation for datasets and augmented data.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np


def compute_dataset_statistics(
    df: pd.DataFrame,
    text_field: str = "evidence_sentence",
    label_fields: List[str] = None,
) -> Dict[str, Any]:
    """
    Compute statistics for a dataset.
    
    Args:
        df: Input DataFrame
        text_field: Name of text column
        label_fields: Names of label columns
        
    Returns:
        Dictionary of statistics
    """
    label_fields = label_fields or ["criteria_label", "evidence_label"]
    
    stats = {
        "num_examples": len(df),
        "text_statistics": compute_text_statistics(df[text_field]),
    }
    
    # Label distributions
    for label_field in label_fields:
        if label_field in df.columns:
            stats[f"{label_field}_distribution"] = (
                df[label_field].value_counts().to_dict()
            )
    
    return stats


def compute_text_statistics(texts: pd.Series) -> Dict[str, float]:
    """
    Compute statistics for text lengths.
    
    Args:
        texts: Series of text strings
        
    Returns:
        Dictionary of text statistics
    """
    lengths = texts.str.len()
    word_counts = texts.str.split().str.len()
    
    return {
        "char_count_mean": float(lengths.mean()),
        "char_count_std": float(lengths.std()),
        "char_count_min": int(lengths.min()),
        "char_count_max": int(lengths.max()),
        "char_count_median": float(lengths.median()),
        "word_count_mean": float(word_counts.mean()),
        "word_count_std": float(word_counts.std()),
        "word_count_min": int(word_counts.min()),
        "word_count_max": int(word_counts.max()),
        "word_count_median": float(word_counts.median()),
    }


def compare_original_augmented(
    original_df: pd.DataFrame,
    augmented_df: pd.DataFrame,
    text_field: str = "evidence_sentence",
) -> Dict[str, Any]:
    """
    Compare statistics between original and augmented data.
    
    Args:
        original_df: Original DataFrame
        augmented_df: Augmented DataFrame
        text_field: Name of text column
        
    Returns:
        Dictionary of comparison statistics
    """
    orig_stats = compute_text_statistics(original_df[text_field])
    aug_stats = compute_text_statistics(augmented_df[text_field])
    
    comparison = {
        "original": orig_stats,
        "augmented": aug_stats,
        "relative_changes": {},
    }
    
    # Compute relative changes
    for key in orig_stats:
        if orig_stats[key] != 0:
            relative_change = (aug_stats[key] - orig_stats[key]) / orig_stats[key]
            comparison["relative_changes"][key] = relative_change
    
    return comparison


def compute_diversity_score(texts: List[str]) -> float:
    """
    Compute diversity score based on unique n-grams.
    
    Args:
        texts: List of text strings
        
    Returns:
        Diversity score (0-1)
    """
    from collections import Counter
    
    # Compute unigram and bigram diversity
    unigrams = []
    bigrams = []
    
    for text in texts:
        words = text.lower().split()
        unigrams.extend(words)
        bigrams.extend([f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)])
    
    # Unique ratio
    unigram_diversity = len(set(unigrams)) / len(unigrams) if unigrams else 0
    bigram_diversity = len(set(bigrams)) / len(bigrams) if bigrams else 0
    
    # Average diversity
    return (unigram_diversity + bigram_diversity) / 2


def compute_edit_distance(str1: str, str2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Edit distance
    """
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def compute_average_edit_distance(
    original_texts: List[str],
    augmented_texts: List[str],
) -> float:
    """
    Compute average edit distance between original and augmented texts.
    
    Args:
        original_texts: List of original texts
        augmented_texts: List of augmented texts
        
    Returns:
        Average edit distance
    """
    distances = []
    
    for orig, aug in zip(original_texts, augmented_texts):
        dist = compute_edit_distance(orig, aug)
        distances.append(dist)
    
    return np.mean(distances)
