"""
Helper script to load and process ReDSM5 data for Colab training.

Converts sentence-level annotations to document-level multi-label format.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


# Mapping from ReDSM5 annotation names to our label names
SYMPTOM_MAPPING = {
    'DEPRESSED_MOOD': 'depressed_mood',
    'ANHEDONIA': 'diminished_interest',
    'APPETITE_CHANGE': 'weight_appetite_change',
    'SLEEP_ISSUES': 'sleep_disturbance',
    'PSYCHOMOTOR': 'psychomotor',
    'FATIGUE': 'fatigue',
    'WORTHLESSNESS': 'worthlessness_guilt',
    'COGNITIVE_ISSUES': 'concentration_indecision',
    'SUICIDAL_THOUGHTS': 'suicidality',
}

LABEL_NAMES = list(SYMPTOM_MAPPING.values())


def load_redsm5_data(data_dir: Path) -> pd.DataFrame:
    """
    Load ReDSM5 posts and annotations, convert to document-level labels.

    Args:
        data_dir: Path to directory containing redsm5_posts.csv and redsm5_annotations.csv

    Returns:
        DataFrame with columns: text, depressed_mood, diminished_interest, etc.
    """
    # Load posts
    posts_path = data_dir / 'redsm5_posts.csv'
    posts_df = pd.read_csv(posts_path)
    print(f"✅ Loaded {len(posts_df)} posts from {posts_path.name}")

    # Load annotations
    annotations_path = data_dir / 'redsm5_annotations.csv'
    annotations_df = pd.read_csv(annotations_path)
    print(f"✅ Loaded {len(annotations_df)} annotations from {annotations_path.name}")

    # Filter to only positive annotations (status=1) and exclude SPECIAL_CASE
    positive_annotations = annotations_df[
        (annotations_df['status'] == 1) &
        (annotations_df['DSM5_symptom'] != 'SPECIAL_CASE')
    ].copy()

    print(f"   Positive annotations (status=1, excluding SPECIAL_CASE): {len(positive_annotations)}")

    # Create document-level labels
    # Group by post_id and collect all symptoms
    post_symptoms = positive_annotations.groupby('post_id')['DSM5_symptom'].apply(set).to_dict()

    # Initialize all labels to 0
    for label in LABEL_NAMES:
        posts_df[label] = 0

    # Set labels to 1 where symptoms are present
    for post_id, symptoms in post_symptoms.items():
        if post_id in posts_df['post_id'].values:
            idx = posts_df[posts_df['post_id'] == post_id].index[0]
            for symptom in symptoms:
                if symptom in SYMPTOM_MAPPING:
                    label_name = SYMPTOM_MAPPING[symptom]
                    posts_df.loc[idx, label_name] = 1

    # Drop post_id column (keep text and labels only)
    result_df = posts_df.drop(columns=['post_id'])

    # Print label statistics
    print(f"\n📊 Label Distribution:")
    for label in LABEL_NAMES:
        count = result_df[label].sum()
        pct = 100 * count / len(result_df)
        print(f"   {label:30s}: {count:4d} ({pct:5.1f}%)")

    # Calculate posts with at least one symptom
    posts_with_symptoms = (result_df[LABEL_NAMES].sum(axis=1) > 0).sum()
    print(f"\n   Posts with ≥1 symptom: {posts_with_symptoms} ({100*posts_with_symptoms/len(result_df):.1f}%)")
    print(f"   Posts with no symptoms: {len(result_df) - posts_with_symptoms}")

    return result_df


def split_redsm5_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/dev/test sets with stratification.

    Args:
        df: DataFrame with text and label columns
        train_ratio: Proportion for training
        dev_ratio: Proportion for development
        test_ratio: Proportion for testing
        seed: Random seed

    Returns:
        Tuple of (train_df, dev_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    # Create stratification signature based on label patterns
    df = df.copy()
    df['label_signature'] = df[LABEL_NAMES].apply(
        lambda row: ''.join(str(int(v)) for v in row),
        axis=1
    )

    # First split: train vs (dev+test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=seed,
        stratify=df['label_signature'] if df['label_signature'].nunique() > 1 else None
    )

    # Second split: dev vs test
    dev_test_ratio = dev_ratio / (dev_ratio + test_ratio)
    dev_df, test_df = train_test_split(
        temp_df,
        train_size=dev_test_ratio,
        random_state=seed,
        stratify=temp_df['label_signature'] if temp_df['label_signature'].nunique() > 1 else None
    )

    # Drop stratification signature
    train_df = train_df.drop(columns=['label_signature'])
    dev_df = dev_df.drop(columns=['label_signature'])
    test_df = test_df.drop(columns=['label_signature'])

    print(f"\n📊 Split Statistics:")
    print(f"   Train: {len(train_df)} samples ({100*len(train_df)/len(df):.1f}%)")
    print(f"   Dev:   {len(dev_df)} samples ({100*len(dev_df)/len(df):.1f}%)")
    print(f"   Test:  {len(test_df)} samples ({100*len(test_df)/len(df):.1f}%)")

    return train_df, dev_df, test_df


def save_redsm5_splits(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Save train/dev/test splits to JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_json(output_dir / 'train.jsonl', orient='records', lines=True)
    dev_df.to_json(output_dir / 'dev.jsonl', orient='records', lines=True)
    test_df.to_json(output_dir / 'test.jsonl', orient='records', lines=True)

    print(f"\n✅ Saved splits to {output_dir}")
    print(f"   train.jsonl: {len(train_df)} samples")
    print(f"   dev.jsonl:   {len(dev_df)} samples")
    print(f"   test.jsonl:  {len(test_df)} samples")


if __name__ == '__main__':
    # Example usage
    data_dir = Path('data/redsm5')
    output_dir = Path('/content/redsm5_processed')

    # Load and process
    df = load_redsm5_data(data_dir)

    # Split
    train_df, dev_df, test_df = split_redsm5_data(df, seed=42)

    # Save
    save_redsm5_splits(train_df, dev_df, test_df, output_dir)
