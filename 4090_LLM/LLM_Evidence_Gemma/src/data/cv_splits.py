"""
Cross-validation split utilities for ReDSM5 evidence extraction dataset.

Provides functions for creating stratified K-fold splits and loading fold data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

from .evidence_dataset import (
    EvidenceDataset,
    build_evidence_dataset,
    prepare_evidence_examples,
    SYMPTOM_LABELS,
    NUM_SYMPTOMS,
)


def create_cv_splits(
    posts_path: str,
    annotations_path: str,
    num_folds: int = 5,
    random_seed: int = 42,
    output_dir: Optional[str] = None,
    include_negatives: bool = False,
) -> List[Dict[str, np.ndarray]]:
    """
    Create stratified K-fold cross-validation splits for evidence extraction.

    Args:
        posts_path: Path to redsm5_posts.csv
        annotations_path: Path to redsm5_annotations.csv
        num_folds: Number of folds for cross-validation
        random_seed: Random seed for reproducibility
        output_dir: Optional directory to save fold indices
        include_negatives: Whether to include negative examples (status=0)

    Returns:
        List of dictionaries, each containing 'train' and 'val' indices
    """
    # Load data
    posts_df = pd.read_csv(posts_path)
    annotations_df = pd.read_csv(annotations_path)

    # Filter annotations
    if not include_negatives:
        annotations_df = annotations_df[annotations_df['status'] == 1]

    # Prepare examples
    examples = prepare_evidence_examples(posts_df, annotations_df, include_negatives=include_negatives)

    # Convert to DataFrame for easier manipulation
    examples_df = pd.DataFrame(examples)

    # Create stratified K-fold splitter (stratify by symptom)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

    # Generate splits
    splits = []
    for fold_idx, (train_indices, val_indices) in enumerate(
        skf.split(np.zeros(len(examples_df)), examples_df['symptom_idx'])
    ):
        split = {
            'train': train_indices,
            'val': val_indices,
        }
        splits.append(split)

        # Save fold indices if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save train examples
            train_examples = [examples[i] for i in train_indices]
            train_df = pd.DataFrame(train_examples)
            train_df.to_csv(output_path / f'fold_{fold_idx}_train.csv', index=False)

            # Save val examples
            val_examples = [examples[i] for i in val_indices]
            val_df = pd.DataFrame(val_examples)
            val_df.to_csv(output_path / f'fold_{fold_idx}_val.csv', index=False)

            # Save fold metadata
            train_symptom_dist = train_df['symptom_idx'].apply(lambda x: SYMPTOM_LABELS[x]).value_counts()
            val_symptom_dist = val_df['symptom_idx'].apply(lambda x: SYMPTOM_LABELS[x]).value_counts()

            fold_metadata = {
                'fold': fold_idx,
                'train_size': len(train_indices),
                'val_size': len(val_indices),
                'train_symptom_distribution': train_symptom_dist.to_dict(),
                'val_symptom_distribution': val_symptom_dist.to_dict(),
            }

            with open(output_path / f'fold_{fold_idx}_metadata.json', 'w') as f:
                json.dump(fold_metadata, f, indent=2)

    print(f"Created {num_folds} stratified folds")

    # Save overall split metadata
    if output_dir:
        split_metadata = {
            'num_folds': num_folds,
            'random_seed': random_seed,
            'total_examples': len(examples_df),
            'num_symptoms': NUM_SYMPTOMS,
            'include_negatives': include_negatives,
        }
        with open(Path(output_dir) / 'split_metadata.json', 'w') as f:
            json.dump(split_metadata, f, indent=2)

    return splits


def load_fold_split(
    data_dir: str,
    fold_idx: int,
    tokenizer=None,
    max_length: int = 512,
    use_cached_dataset: bool = False,
    cache_dir: Optional[str] = None,
    overwrite_cache: bool = False,
) -> Tuple[EvidenceDataset, EvidenceDataset]:
    """
    Load a specific fold's train and validation datasets.

    Args:
        data_dir: Directory containing fold CSV files
        fold_idx: Index of fold to load (0-indexed)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        use_cached_dataset: Whether to pre-tokenize & cache dataset tensors
        cache_dir: Optional directory for cached tensors (defaults under data_dir)
        overwrite_cache: Force regeneration of cached tensors

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    data_path = Path(data_dir)

    # Load fold examples
    train_df = pd.read_csv(data_path / f'fold_{fold_idx}_train.csv')
    val_df = pd.read_csv(data_path / f'fold_{fold_idx}_val.csv')

    # Convert to list of dicts
    train_examples = train_df.to_dict('records')
    val_examples = val_df.to_dict('records')

    cache_root = None
    if use_cached_dataset:
        cache_root = Path(cache_dir) if cache_dir else data_path / '_cache'

    cache_dir_str = str(cache_root) if cache_root else None

    # Create datasets
    train_dataset = build_evidence_dataset(
        train_examples,
        tokenizer,
        max_length,
        use_cached_dataset=use_cached_dataset,
        cache_dir=cache_dir_str,
        split_name=f'fold{fold_idx}_train',
        overwrite_cache=overwrite_cache,
    )
    val_dataset = build_evidence_dataset(
        val_examples,
        tokenizer,
        max_length,
        use_cached_dataset=use_cached_dataset,
        cache_dir=cache_dir_str,
        split_name=f'fold{fold_idx}_val',
        overwrite_cache=overwrite_cache,
    )

    return train_dataset, val_dataset


def get_fold_statistics(splits: List[Dict[str, np.ndarray]]) -> pd.DataFrame:
    """
    Compute statistics for all folds.

    Args:
        splits: List of fold dictionaries with 'train' and 'val' indices

    Returns:
        DataFrame with fold statistics
    """
    stats = []
    for fold_idx, split in enumerate(splits):
        stats.append({
            'fold': fold_idx,
            'train_size': len(split['train']),
            'val_size': len(split['val']),
            'total_size': len(split['train']) + len(split['val']),
            'train_ratio': len(split['train']) / (len(split['train']) + len(split['val'])),
            'val_ratio': len(split['val']) / (len(split['train']) + len(split['val'])),
        })

    return pd.DataFrame(stats)


def load_fold_metadata(data_dir: str, fold_idx: int) -> Dict:
    """
    Load metadata for a specific fold.

    Args:
        data_dir: Directory containing fold files
        fold_idx: Index of fold

    Returns:
        Dictionary containing fold metadata
    """
    metadata_path = Path(data_dir) / f'fold_{fold_idx}_metadata.json'

    if not metadata_path.exists():
        return {}

    with open(metadata_path, 'r') as f:
        return json.load(f)


def verify_fold_stratification(data_dir: str, num_folds: int) -> pd.DataFrame:
    """
    Verify that folds are properly stratified.

    Args:
        data_dir: Directory containing fold files
        num_folds: Number of folds to verify

    Returns:
        DataFrame showing symptom distribution across folds
    """
    results = []

    for fold_idx in range(num_folds):
        metadata = load_fold_metadata(data_dir, fold_idx)

        if not metadata:
            continue

        for split_type in ['train', 'val']:
            dist_key = f'{split_type}_symptom_distribution'
            if dist_key in metadata:
                for symptom, count in metadata[dist_key].items():
                    results.append({
                        'fold': fold_idx,
                        'split': split_type,
                        'symptom': symptom,
                        'count': count,
                    })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Pivot for easier viewing
    pivot_df = df.pivot_table(
        index=['fold', 'split'],
        columns='symptom',
        values='count',
        fill_value=0,
    )

    return pivot_df
