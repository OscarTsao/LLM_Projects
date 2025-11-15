#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy import stats
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

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', str(text)).strip()
    if len(text) < 10:
        return ""
    return text

def calculate_similarity_scores(posts, model, dsm_embeddings):
    """Calculate similarity scores between posts and DSM criteria"""
    valid_posts = [clean_text(post) for post in posts if clean_text(post)]
    if not valid_posts:
        return []

    post_embeddings = model.encode(valid_posts, show_progress_bar=False)
    similarities = cosine_similarity(post_embeddings, dsm_embeddings)
    max_similarities = np.max(similarities, axis=1)
    return max_similarities.tolist()

def main():
    print("=" * 80)
    print("RAG POST SIMILARITY ANALYSIS - SIMPLIFIED")
    print("=" * 80)

    # Load model
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    dsm_embeddings = model.encode(DSM_CRITERIA)

    # Load data
    print("Loading posts data...")
    negative_df = pd.read_csv('/media/user/SSD1/YuNing/Psy_RAG/Data/translated_posts.csv')
    positive_df = pd.read_csv('/media/user/SSD1/YuNing/Psy_RAG/Data/translated_post_positive_random.csv')

    negative_posts = negative_df['translated_post'].dropna().tolist()
    positive_posts = positive_df['positive_post'].dropna().tolist()

    # Take first 2000 posts from each to match the requirement
    negative_posts_2k = negative_posts[:2000]
    positive_posts_2k = positive_posts[:2000]

    print(f"Analyzing {len(negative_posts_2k)} negative posts and {len(positive_posts_2k)} positive posts")

    # Calculate similarity scores
    print("Calculating similarity scores...")
    negative_scores = calculate_similarity_scores(negative_posts_2k, model, dsm_embeddings)
    positive_scores = calculate_similarity_scores(positive_posts_2k, model, dsm_embeddings)

    # Calculate statistics
    neg_mean = np.mean(negative_scores)
    neg_std = np.std(negative_scores, ddof=1)
    neg_n = len(negative_scores)

    pos_mean = np.mean(positive_scores)
    pos_std = np.std(positive_scores, ddof=1)
    pos_n = len(positive_scores)

    # Statistical test
    t_stat, p_value = stats.ttest_ind(negative_scores, positive_scores, equal_var=False)

    # Effect size
    pooled_std = np.sqrt(((neg_n - 1) * neg_std**2 + (pos_n - 1) * pos_std**2) / (neg_n + pos_n - 2))
    cohens_d = (neg_mean - pos_mean) / pooled_std if pooled_std > 0 else 0

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"\nNEGATIVE POSTS (2000 posts):")
    print(f"  Mean: {neg_mean:.6f}")
    print(f"  Std:  {neg_std:.6f}")
    print(f"  N:    {neg_n}")

    print(f"\nPOSITIVE POSTS (2000 posts):")
    print(f"  Mean: {pos_mean:.6f}")
    print(f"  Std:  {pos_std:.6f}")
    print(f"  N:    {pos_n}")

    print(f"\nSTATISTICAL TEST (Welch's t-test):")
    print(f"  t-statistic: {t_stat:.6f}")
    print(f"  p-value:     {p_value:.2e}")
    print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'} (α = 0.05)")

    print(f"\nEFFECT SIZE:")
    print(f"  Cohen's d:   {cohens_d:.6f}")

    print("\n" + "=" * 80)
    print(f"FINAL P-VALUE: {p_value:.2e}")
    print("=" * 80)

if __name__ == "__main__":
    main()