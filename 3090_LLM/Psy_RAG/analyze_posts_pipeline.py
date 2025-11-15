#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy import stats
import json
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# DSM-5 diagnostic criteria for common mental health conditions
DSM_CRITERIA = [
    "Persistent sadness, hopelessness, or empty mood lasting at least two weeks",
    "Markedly diminished interest or pleasure in activities",
    "Significant weight loss or gain, or decrease or increase in appetite",
    "Insomnia or hypersomnia nearly every day",
    "Psychomotor agitation or retardation",
    "Fatigue or loss of energy nearly every day",
    "Feelings of worthlessness or excessive guilt",
    "Diminished ability to think or concentrate, or indecisiveness",
    "Recurrent thoughts of death or suicidal ideation",
    "Excessive anxiety and worry occurring more days than not",
    "Difficulty controlling worry",
    "Restlessness or feeling on edge",
    "Being easily fatigued",
    "Difficulty concentrating",
    "Irritability",
    "Muscle tension",
    "Sleep disturbance",
    "Panic attacks with intense fear or discomfort",
    "Persistent fear of having panic attacks",
    "Avoidance of situations due to fear of panic",
    "Social anxiety in performance or social situations",
    "Fear of being judged or scrutinized by others",
    "Avoidance of social situations",
    "Intrusive memories or flashbacks of traumatic events",
    "Distressing dreams related to trauma",
    "Avoidance of trauma-related stimuli",
    "Negative alterations in mood and cognition",
    "Hypervigilance and exaggerated startle response"
]

def load_posts_data():
    """Load negative and positive posts from CSV files"""
    print("Loading posts data...")

    # Load negative posts
    negative_df = pd.read_csv('/media/user/SSD1/YuNing/Psy_RAG/Data/translated_posts.csv')
    print(f"Loaded {len(negative_df)} negative posts")

    # Load positive posts
    positive_df = pd.read_csv('/media/user/SSD1/YuNing/Psy_RAG/Data/translated_post_positive_random.csv')
    print(f"Loaded {len(positive_df)} positive posts")

    # Extract text from appropriate columns
    negative_posts = negative_df['post_context'].dropna().tolist()
    positive_posts = positive_df['positive_post'].dropna().tolist()

    print(f"Valid negative posts: {len(negative_posts)}")
    print(f"Valid positive posts: {len(positive_posts)}")

    return negative_posts, positive_posts

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', str(text)).strip()

    # Remove very short texts (less than 10 characters)
    if len(text) < 10:
        return ""

    return text

def calculate_similarity_scores(posts, model, dsm_embeddings):
    """Calculate similarity scores between posts and DSM criteria"""
    print(f"Calculating similarity scores for {len(posts)} posts...")

    # Filter out empty posts
    valid_posts = [clean_text(post) for post in posts if clean_text(post)]
    print(f"Valid posts after cleaning: {len(valid_posts)}")

    if not valid_posts:
        return []

    # Get embeddings for posts
    post_embeddings = model.encode(valid_posts, show_progress_bar=True)

    # Calculate cosine similarity with DSM criteria
    similarities = cosine_similarity(post_embeddings, dsm_embeddings)

    # Take maximum similarity score for each post across all DSM criteria
    max_similarities = np.max(similarities, axis=1)

    return max_similarities.tolist()

def process_posts_in_groups(posts, group_size=2000):
    """Process posts in groups of specified size"""
    groups = []
    for i in range(0, len(posts), group_size):
        group = posts[i:i + group_size]
        groups.append(group)
    return groups

def analyze_posts_pipeline():
    """Main pipeline to analyze similarity scores between negative and positive posts"""

    print("=" * 80)
    print("RAG POST SIMILARITY ANALYSIS PIPELINE")
    print("=" * 80)

    # Load sentence transformer model
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Get embeddings for DSM criteria
    print("Encoding DSM-5 criteria...")
    dsm_embeddings = model.encode(DSM_CRITERIA)

    # Load posts data
    negative_posts, positive_posts = load_posts_data()

    # Process negative posts in groups of 2000
    print(f"\nProcessing negative posts in groups of 2000...")
    negative_groups = process_posts_in_groups(negative_posts, 2000)
    print(f"Created {len(negative_groups)} negative groups")

    # Process positive posts in groups of 2000
    print(f"Processing positive posts in groups of 2000...")
    positive_groups = process_posts_in_groups(positive_posts, 2000)
    print(f"Created {len(positive_groups)} positive groups")

    # Calculate similarity scores for each group
    all_negative_scores = []
    all_positive_scores = []

    group_results = {
        'negative_groups': [],
        'positive_groups': [],
        'group_comparisons': []
    }

    # Process negative groups
    for i, group in enumerate(negative_groups):
        print(f"\nProcessing negative group {i+1}/{len(negative_groups)}...")
        scores = calculate_similarity_scores(group, model, dsm_embeddings)

        if scores:
            group_mean = np.mean(scores)
            group_std = np.std(scores, ddof=1) if len(scores) > 1 else 0

            group_results['negative_groups'].append({
                'group_id': i+1,
                'n_posts': len(scores),
                'mean': float(group_mean),
                'std': float(group_std),
                'scores': [float(x) for x in scores]
            })

            all_negative_scores.extend(scores)
            print(f"  Group {i+1}: Mean = {group_mean:.6f}, Std = {group_std:.6f}, N = {len(scores)}")

    # Process positive groups
    for i, group in enumerate(positive_groups):
        print(f"\nProcessing positive group {i+1}/{len(positive_groups)}...")
        scores = calculate_similarity_scores(group, model, dsm_embeddings)

        if scores:
            group_mean = np.mean(scores)
            group_std = np.std(scores, ddof=1) if len(scores) > 1 else 0

            group_results['positive_groups'].append({
                'group_id': i+1,
                'n_posts': len(scores),
                'mean': float(group_mean),
                'std': float(group_std),
                'scores': [float(x) for x in scores]
            })

            all_positive_scores.extend(scores)
            print(f"  Group {i+1}: Mean = {group_mean:.6f}, Std = {group_std:.6f}, N = {len(scores)}")

    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    if all_negative_scores and all_positive_scores:
        # Calculate overall statistics
        neg_mean = np.mean(all_negative_scores)
        neg_std = np.std(all_negative_scores, ddof=1)
        neg_n = len(all_negative_scores)

        pos_mean = np.mean(all_positive_scores)
        pos_std = np.std(all_positive_scores, ddof=1)
        pos_n = len(all_positive_scores)

        print(f"\nNEGATIVE POSTS (ALL GROUPS):")
        print(f"  Mean: {neg_mean:.6f}")
        print(f"  Std:  {neg_std:.6f}")
        print(f"  N:    {neg_n}")

        print(f"\nPOSITIVE POSTS (ALL GROUPS):")
        print(f"  Mean: {pos_mean:.6f}")
        print(f"  Std:  {pos_std:.6f}")
        print(f"  N:    {pos_n}")

        # Statistical test
        print(f"\n" + "-" * 60)
        print("STATISTICAL ANALYSIS")
        print("-" * 60)

        # Welch's t-test
        t_stat, p_value = stats.ttest_ind(all_negative_scores, all_positive_scores, equal_var=False)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((neg_n - 1) * neg_std**2 + (pos_n - 1) * pos_std**2) / (neg_n + pos_n - 2))
        cohens_d = (neg_mean - pos_mean) / pooled_std if pooled_std > 0 else 0

        print(f"\nWelch's t-test:")
        print(f"  t-statistic: {t_stat:.6f}")
        print(f"  p-value:     {p_value:.10f}")
        print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'} (α = 0.05)")
        print(f"\nEffect Size:")
        print(f"  Cohen's d:   {cohens_d:.6f}")

        # Group-wise comparisons
        print(f"\n" + "-" * 60)
        print("GROUP-WISE ANALYSIS")
        print("-" * 60)

        # Compare each negative group with each positive group
        for neg_group in group_results['negative_groups']:
            for pos_group in group_results['positive_groups']:
                if neg_group['scores'] and pos_group['scores']:
                    t_stat_group, p_value_group = stats.ttest_ind(
                        neg_group['scores'],
                        pos_group['scores'],
                        equal_var=False
                    )

                    group_results['group_comparisons'].append({
                        'negative_group': neg_group['group_id'],
                        'positive_group': pos_group['group_id'],
                        't_statistic': float(t_stat_group),
                        'p_value': float(p_value_group),
                        'significant': bool(p_value_group < 0.05)
                    })

                    print(f"Negative Group {neg_group['group_id']} vs Positive Group {pos_group['group_id']}:")
                    print(f"  p-value: {p_value_group:.6f} {'*' if p_value_group < 0.05 else ''}")

        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_statistics': {
                'negative_posts': {
                    'mean': float(neg_mean),
                    'std': float(neg_std),
                    'n': int(neg_n)
                },
                'positive_posts': {
                    'mean': float(pos_mean),
                    'std': float(pos_std),
                    'n': int(pos_n)
                },
                'statistical_test': {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'cohens_d': float(cohens_d),
                    'significant': bool(p_value < 0.05)
                }
            },
            'group_results': group_results
        }

        # Save to results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = results_dir / f"posts_similarity_analysis_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n" + "=" * 80)
        print(f"FINAL RESULTS")
        print("=" * 80)
        print(f"OVERALL P-VALUE: {p_value:.10f}")
        print(f"Results saved to: {output_file}")

        return results

    else:
        print("ERROR: No valid scores calculated")
        return None

if __name__ == "__main__":
    analyze_posts_pipeline()