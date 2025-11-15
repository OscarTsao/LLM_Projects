"""Dataset module for evidence sentence classification.

This module handles data loading, preprocessing, and dataset creation for
the DeBERTa-v3 NSP-style evidence sentence classification task.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

SYMPTOM_TO_CRITERION_ID = {
    'DEPRESSED_MOOD': 'A.1',
    'ANHEDONIA': 'A.2',
    'APPETITE_CHANGE': 'A.3',
    'SLEEP_ISSUES': 'A.4',
    'PSYCHOMOTOR': 'A.5',
    'FATIGUE': 'A.6',
    'WORTHLESSNESS': 'A.7',
    'COGNITIVE_ISSUES': 'A.8',
    'SUICIDAL_THOUGHTS': 'A.9',
    'SPECIAL_CASE': 'SPECIAL_CASE'
}

SPECIAL_CASE_TEXT = (
    "Clinician-marked special case outside DSM-5 core symptoms; refer to annotation rationale."
)


@dataclass
class EvidenceSample:
    """Represents a single evidence classification sample.

    Attributes:
        post_id: Unique identifier for the post.
        sentence_id: Unique identifier for the sentence within the post.
        criterion_id: Unique identifier for the DSM-5 criterion.
        sentence: The text of the sentence.
        criterion: The text of the DSM-5 criterion.
        label: Binary label (0=not evidence, 1=evidence).
    """
    post_id: str
    sentence_id: str
    criterion_id: str
    sentence: str
    criterion: str
    label: int

    def get_composite_id(self) -> str:
        """Generate composite identity for deduplication.

        Returns:
            Composite ID string in format: post_id|sentence_id|criterion_id
        """
        return f"{self.post_id}|{self.sentence_id}|{self.criterion_id}"


def load_dsm5_criteria(dsm5_dir: str) -> Dict[str, str]:
    """Load DSM-5 criteria from JSON files.

    Args:
        dsm5_dir: Path to directory containing DSM-5 JSON files.

    Returns:
        Dictionary mapping criterion_id to criterion text.

    Raises:
        FileNotFoundError: If DSM-5 directory doesn't exist.
        ValueError: If no valid criteria files found.
    """
    dsm5_path = Path(dsm5_dir)
    if not dsm5_path.exists():
        raise FileNotFoundError(f"DSM-5 directory not found: {dsm5_dir}")

    criteria_dict = {}
    json_files = list(dsm5_path.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {dsm5_dir}")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, dict):
                if 'criteria' in data:
                    criteria_section = data['criteria']
                    if isinstance(criteria_section, dict):
                        # Format: {"criteria": {"id": "text", ...}}
                        for crit_id, crit_text in criteria_section.items():
                            criteria_dict[crit_id] = crit_text
                    elif isinstance(criteria_section, list):
                        # Format: {"criteria": [{"id": "...", "text": "..."}, ...]}
                        for item in criteria_section:
                            crit_id = item.get('id')
                            crit_text = item.get('text')
                            if crit_id and crit_text:
                                criteria_dict[crit_id] = crit_text
                    else:
                        logger.warning(
                            f"Unrecognized criteria format in {json_file.name}: {type(criteria_section)}"
                        )
                else:
                    # Format: {"id": "text", ...}
                    for crit_id, crit_text in data.items():
                        criteria_dict[crit_id] = crit_text
            elif isinstance(data, list):
                # Format: [{"id": "...", "text": "..."}, ...]
                for item in data:
                    if 'id' in item and 'text' in item:
                        criteria_dict[item['id']] = item['text']

            logger.info(f"Loaded {len(data)} criteria from {json_file.name}")
        except Exception as e:
            logger.warning(f"Failed to load {json_file.name}: {e}")
            continue

    if not criteria_dict:
        raise ValueError(f"No criteria loaded from {dsm5_dir}")

    logger.info(f"Total criteria loaded: {len(criteria_dict)}")
    return criteria_dict


def load_redsm5_data(csv_path: str, criteria_dict: Dict[str, str]) -> pd.DataFrame:
    """Load and validate ReDSM5 dataset.

    Args:
        csv_path: Path to posts.csv file.
        criteria_dict: Dictionary of criterion_id to criterion text.

    Returns:
        DataFrame with columns: post_id, sentence_id, sentence,
        criterion_id, criterion, label.

    Raises:
        FileNotFoundError: If CSV file doesn't exist.
        ValueError: If required columns are missing.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    column_mappings = {
        'post_id': ['post_id', 'postId', 'post'],
        'sentence_id': ['sentence_id', 'sentenceId', 'sent_id'],
        'sentence': ['sentence', 'sentence_text', 'text', 'content'],
        'symptom': ['DSM5_symptom', 'symptom', 'criterion'],
        'label': ['status', 'label', 'is_evidence', 'evidence'],
        'criterion_id': ['criterion_id', 'criterionId']
    }

    for standard_name, possible_names in column_mappings.items():
        if standard_name in df.columns:
            continue
        for possible_name in possible_names:
            if possible_name in df.columns:
                df = df.rename(columns={possible_name: standard_name})
                break

    required_base = ['post_id', 'sentence', 'label']
    missing_base = [col for col in required_base if col not in df.columns]
    if missing_base:
        raise ValueError(f"Missing required columns: {missing_base}")

    if 'criterion_id' not in df.columns:
        if 'symptom' not in df.columns:
            raise ValueError("Dataset must contain either 'criterion_id' or 'symptom' column")
        df['criterion_id'] = (
            df['symptom']
            .map(SYMPTOM_TO_CRITERION_ID)
            .fillna(df['symptom'])
            .astype(str)
        )
    else:
        df['criterion_id'] = df['criterion_id'].astype(str)

    # Add sentence_id if not present
    if 'sentence_id' not in df.columns:
        df['sentence_id'] = df.groupby('post_id').cumcount().astype(str)

    # Convert types
    df['post_id'] = df['post_id'].astype(str)
    df['sentence_id'] = df['sentence_id'].astype(str)
    df['criterion_id'] = df['criterion_id'].astype(str)
    df['label'] = df['label'].astype(int)

    # Add criterion text
    df['criterion'] = df['criterion_id'].map(criteria_dict)

    missing_mask = df['criterion'].isna()
    if missing_mask.any():
        logger.warning(
            "Falling back to symptom names for %d rows without matching DSM-5 criteria",
            missing_mask.sum()
        )
        fallback_text = (
            df.loc[missing_mask, 'symptom']
            .fillna(df.loc[missing_mask, 'criterion_id'])
            .str.replace('_', ' ', regex=False)
            .str.title()
        )
        df.loc[missing_mask, 'criterion'] = fallback_text
        special_case_mask = df['criterion_id'] == 'SPECIAL_CASE'
        df.loc[special_case_mask, 'criterion'] = SPECIAL_CASE_TEXT

    logger.info(f"Final dataset size: {len(df)} samples")
    logger.info(f"Class distribution: {df['label'].value_counts().to_dict()}")

    return df


def stratified_negative_sampling(
    df: pd.DataFrame,
    pos_neg_ratio: float = 0.333,
    seed: int = 42
) -> pd.DataFrame:
    """Perform stratified negative sampling to balance dataset.

    Args:
        df: DataFrame with label column.
        pos_neg_ratio: Ratio of positive to negative samples (1:3 = 0.333).
        seed: Random seed for reproducibility.

    Returns:
        Balanced DataFrame with pos:neg ratio maintained.
    """
    np.random.seed(seed)

    positive = df[df['label'] == 1]
    negative = df[df['label'] == 0]

    n_positive = len(positive)
    n_negative_target = int(n_positive / pos_neg_ratio)

    logger.info(f"Positive samples: {n_positive}")
    logger.info(f"Negative samples available: {len(negative)}")
    logger.info(f"Target negative samples: {n_negative_target}")

    if len(negative) > n_negative_target:
        negative_sampled = negative.sample(n=n_negative_target, random_state=seed)
    else:
        logger.warning(
            f"Not enough negative samples. Using all {len(negative)} available."
        )
        negative_sampled = negative

    balanced_df = pd.concat([positive, negative_sampled], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(f"Balanced dataset size: {len(balanced_df)}")
    logger.info(
        f"Balanced distribution: {balanced_df['label'].value_counts().to_dict()}"
    )

    return balanced_df


def create_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create GroupKFold splits grouped by post_id.

    Args:
        df: DataFrame with post_id column.
        n_splits: Number of folds.
        seed: Random seed for reproducibility.

    Returns:
        List of (train_indices, val_indices) tuples for each fold.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df['post_id'].values

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(df, groups=groups), 1
    ):
        logger.info(
            f"Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}"
        )
        folds.append((train_idx, val_idx))

    return folds


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute class weights for weighted cross-entropy loss.

    Args:
        labels: Array of binary labels.

    Returns:
        Tensor of class weights [weight_class_0, weight_class_1].
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    weights = total / (len(unique) * counts)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    logger.info(f"Class weights: {weight_tensor.tolist()}")
    return weight_tensor


class ReDSM5Dataset(Dataset):
    """PyTorch Dataset for evidence sentence classification.

    Implements NSP-style formatting: [CLS] <criterion> [SEP] <sentence> [SEP]
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ):
        """Initialize dataset.

        Args:
            dataframe: DataFrame with sentence, criterion, label columns.
            tokenizer: Hugging Face tokenizer.
            max_length: Maximum sequence length.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(f"Created dataset with {len(self)} samples")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single tokenized sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with input_ids, attention_mask, labels.
        """
        row = self.dataframe.iloc[idx]

        # NSP format: [CLS] criterion [SEP] sentence [SEP]
        encoding = self.tokenizer(
            row['criterion'],
            row['sentence'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(row['label'], dtype=torch.long)
        }
