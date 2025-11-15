#!/usr/bin/env python3

import json
import numpy as np
from scipy import stats
import os
from pathlib import Path

def load_comparison_results(file_path):
    """Load comparison results from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def analyze_similarity_scores():
    """Analyze similarity scores for positive and negative posts"""

    # Find the most recent comparison results file
    results_dir = Path('/media/user/SSD1/YuNing/Psy_RAG/results')
    comparison_files = list(results_dir.glob('comparison_results_*.json'))

    if not comparison_files:
        print("No comparison results files found")
        return

    # Use the most recent file
    latest_file = max(comparison_files, key=lambda x: x.stat().st_mtime)
    print(f"Analyzing file: {latest_file}")

    # Load data
    data = load_comparison_results(latest_file)

    # Extract statistics
    stats_data = data.get('statistics', {})

    # Look for similarity score data
    print("\nAvailable metrics in the data:")
    for metric_name, metric_data in stats_data.items():
        print(f"  - {metric_name}")
        if isinstance(metric_data, dict):
            if 'negative_scores' in metric_data and 'positive_scores' in metric_data:
                analyze_metric_scores(metric_name, metric_data)

def analyze_metric_scores(metric_name, metric_data):
    """Analyze scores for a specific metric"""
    print(f"\n=== {metric_name.upper()} ANALYSIS ===")

    negative_scores = np.array(metric_data.get('negative_scores', []))
    positive_scores = np.array(metric_data.get('positive_scores', []))

    if len(negative_scores) == 0 or len(positive_scores) == 0:
        print(f"No score data available for {metric_name}")
        return

    # Remove any zero scores if they seem to be missing data indicators
    if np.all(negative_scores == 0) and np.all(positive_scores == 0):
        print(f"All scores are zero for {metric_name} - possibly no valid data")
        return

    # Calculate statistics for negative posts
    neg_mean = np.mean(negative_scores)
    neg_std = np.std(negative_scores, ddof=1)
    neg_n = len(negative_scores)

    # Calculate statistics for positive posts
    pos_mean = np.mean(positive_scores)
    pos_std = np.std(positive_scores, ddof=1)
    pos_n = len(positive_scores)

    print(f"\nNEGATIVE POSTS:")
    print(f"  Mean: {neg_mean:.6f}")
    print(f"  Std:  {neg_std:.6f}")
    print(f"  N:    {neg_n}")

    print(f"\nPOSITIVE POSTS:")
    print(f"  Mean: {pos_mean:.6f}")
    print(f"  Std:  {pos_std:.6f}")
    print(f"  N:    {pos_n}")

    # Perform statistical test
    if neg_std > 0 or pos_std > 0:  # Only test if there's variation
        # Use Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(negative_scores, positive_scores, equal_var=False)

        print(f"\nSTATISTICAL TEST (Welch's t-test):")
        print(f"  t-statistic: {t_stat:.6f}")
        print(f"  p-value:     {p_value:.6f}")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((neg_n - 1) * neg_std**2 + (pos_n - 1) * pos_std**2) / (neg_n + pos_n - 2))
        if pooled_std > 0:
            cohens_d = (neg_mean - pos_mean) / pooled_std
            print(f"  Cohen's d:   {cohens_d:.6f}")
    else:
        print(f"\nNo variation in scores - cannot perform statistical test")

if __name__ == "__main__":
    analyze_similarity_scores()