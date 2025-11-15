"""Dataset for ReDSM5 evidence extraction (SQuAD-style QA)."""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# DSM-5 symptom categories
SYMPTOM_LABELS = [
    'DEPRESSED_MOOD',
    'ANHEDONIA',
    'APPETITE_CHANGE',
    'SLEEP_ISSUES',
    'PSYCHOMOTOR',
    'FATIGUE',
    'WORTHLESSNESS',
    'COGNITIVE_ISSUES',
    'SUICIDAL_THOUGHTS',
    'SPECIAL_CASE'
]

NUM_SYMPTOMS = len(SYMPTOM_LABELS)


class EvidenceDataset(Dataset):
    """
    Dataset for extractive evidence QA from ReDSM5.

    Each example is a (question, context, answer) triplet:
    - Question: "Find evidence for [DSM5_symptom] in this post"
    - Context: Full Reddit post text
    - Answer: Sentence text that provides evidence (with start/end positions)
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer,
        max_length: int = 512,
    ):
        """
        Args:
            examples: List of dicts with keys:
                - post_id
                - question (formatted symptom query)
                - context (post text)
                - answer_text (sentence text)
                - answer_start (char position in context)
                - symptom_idx (0-9)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def _encode_example(self, example: Dict) -> Dict[str, torch.Tensor]:
        # Tokenize question + context
        encoding = self.tokenizer(
            example['question'],
            example['context'],
            max_length=self.max_length,
            padding='max_length',
            truncation='only_second',  # Truncate context if needed
            return_tensors='pt',
            return_offsets_mapping=True,  # For char→token position mapping
        )

        # Get token positions for answer span
        offset_mapping = encoding['offset_mapping'].squeeze(0)
        answer_start_char = example['answer_start']
        answer_end_char = answer_start_char + len(example['answer_text'])

        # Find start and end token positions
        start_position = 0
        end_position = 0
        for i, (start_offset, end_offset) in enumerate(offset_mapping):
            if start_offset <= answer_start_char < end_offset:
                start_position = i
            if start_offset < answer_end_char <= end_offset:
                end_position = i
                break

        # Handle edge case: answer not found (truncated)
        sequence_ids = encoding.sequence_ids(0)
        if sequence_ids is None or start_position == 0:
            # Answer was truncated, set to CLS token position
            start_position = 0
            end_position = 0

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'start_positions': torch.tensor(start_position, dtype=torch.long),
            'end_positions': torch.tensor(end_position, dtype=torch.long),
            'symptom_idx': torch.tensor(example['symptom_idx'], dtype=torch.long),
            'post_id': example['post_id'],
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._encode_example(self.examples[idx])


def _sanitize_for_filename(value: str) -> str:
    """Sanitize arbitrary strings for filesystem use."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', value)


def _hash_examples_metadata(examples: List[Dict]) -> str:
    """Hash minimal example metadata to invalidate cache on data changes."""
    hasher = hashlib.sha256()
    for example in examples:
        meta = (
            f"{example.get('post_id', '')}|"
            f"{example.get('symptom_idx', -1)}|"
            f"{example.get('answer_start', -1)}|"
            f"{len(example.get('context', '') or '')}"
        )
        hasher.update(meta.encode('utf-8'))
    return hasher.hexdigest()


class CachedEvidenceDataset(EvidenceDataset):
    """
    Evidence dataset that precomputes and (optionally) persists tokenized features.

    This avoids repeated tokenization during training/evaluation and dramatically
    improves data loader throughput on repeated epochs or multi-fold runs.
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        split_name: Optional[str] = None,
        overwrite_cache: bool = False,
    ):
        super().__init__(examples, tokenizer, max_length)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split_name = split_name or 'dataset'
        self.overwrite_cache = overwrite_cache
        self._dataset_hash = _hash_examples_metadata(examples)
        self._cache_path = self._build_cache_path()
        self._features = self._load_or_build_features()

    def _build_cache_path(self) -> Optional[Path]:
        if not self.cache_dir:
            return None

        tokenizer_id = getattr(self.tokenizer, 'name_or_path', self.tokenizer.__class__.__name__)
        sanitized_tokenizer = _sanitize_for_filename(tokenizer_id)
        sanitized_split = _sanitize_for_filename(self.split_name)
        filename = (
            f"{sanitized_split}_{sanitized_tokenizer}_len{self.max_length}_"
            f"{self._dataset_hash[:12]}.pt"
        )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / filename

    def _load_or_build_features(self) -> List[Dict[str, torch.Tensor]]:
        if self._cache_path and self._cache_path.exists() and not self.overwrite_cache:
            logger.info(
                "Loaded cached dataset '%s' (%d samples) from %s",
                self.split_name,
                len(self.examples),
                self._cache_path,
            )
            return torch.load(self._cache_path, map_location='cpu')

        logger.info(
            "Caching %d samples for split '%s'%s",
            len(self.examples),
            self.split_name,
            f" at {self._cache_path}" if self._cache_path else "",
        )
        features = [self._encode_example(example) for example in self.examples]

        if self._cache_path:
            tmp_path = self._cache_path.with_suffix('.tmp')
            torch.save(features, tmp_path)
            os.replace(tmp_path, self._cache_path)

        return features

    def __len__(self):
        return len(self._features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._features[idx]


def build_evidence_dataset(
    examples: List[Dict],
    tokenizer,
    max_length: int,
    use_cached_dataset: bool = False,
    cache_dir: Optional[str] = None,
    split_name: Optional[str] = None,
    overwrite_cache: bool = False,
) -> EvidenceDataset:
    """
    Helper to construct either the vanilla EvidenceDataset or its cached variant.
    """
    if use_cached_dataset:
        return CachedEvidenceDataset(
            examples=examples,
            tokenizer=tokenizer,
            max_length=max_length,
            cache_dir=cache_dir,
            split_name=split_name,
            overwrite_cache=overwrite_cache,
        )

    return EvidenceDataset(examples, tokenizer, max_length)


def prepare_evidence_examples(
    posts_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    symptom_filter: Optional[List[str]] = None,
    include_negatives: bool = False,
) -> List[Dict]:
    """
    Prepare QA examples from posts and annotations.

    Args:
        posts_df: DataFrame with columns [post_id, text]
        annotations_df: DataFrame with columns [post_id, sentence_id, sentence_text,
                        DSM5_symptom, status, explanation]
        symptom_filter: Optional list of symptoms to include (default: all)
        include_negatives: Whether to include status=0 examples (default: False)

    Returns:
        List of example dicts ready for EvidenceDataset
    """
    examples = []

    # Create post_id -> text mapping
    post_texts = dict(zip(posts_df['post_id'], posts_df['text']))

    # Filter annotations
    if not include_negatives:
        annotations_df = annotations_df[annotations_df['status'] == 1]

    if symptom_filter:
        annotations_df = annotations_df[annotations_df['DSM5_symptom'].isin(symptom_filter)]

    # Create examples
    for _, row in annotations_df.iterrows():
        post_id = row['post_id']
        if post_id not in post_texts:
            logger.warning(f"Post {post_id} not found in posts DataFrame")
            continue

        context = post_texts[post_id]
        answer_text = row['sentence_text']
        symptom = row['DSM5_symptom']

        # Find answer position in context
        answer_start = context.find(answer_text)
        if answer_start == -1:
            # Try case-insensitive search
            answer_start = context.lower().find(answer_text.lower())
            if answer_start == -1:
                logger.warning(f"Answer not found in context for {row['sentence_id']}")
                continue

        # Format question
        question = f"Find evidence for {symptom.replace('_', ' ').lower()} in this post"

        examples.append({
            'post_id': post_id,
            'question': question,
            'context': context,
            'answer_text': answer_text,
            'answer_start': answer_start,
            'symptom_idx': SYMPTOM_LABELS.index(symptom),
            'status': row['status'],
        })

    logger.info(f"Created {len(examples)} QA examples")
    return examples


def load_redsm5_evidence(
    data_dir: str,
    tokenizer,
    max_length: int = 512,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
    include_negatives: bool = False,
    use_cached_dataset: bool = False,
    cache_dir: Optional[str] = None,
    overwrite_cache: bool = False,
) -> Tuple[EvidenceDataset, EvidenceDataset, EvidenceDataset]:
    """
    Load ReDSM5 dataset and create train/val/test splits for evidence extraction.

    Args:
        data_dir: Path to directory containing redsm5_posts.csv and redsm5_annotations.csv
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_seed: Random seed for reproducibility
        include_negatives: Whether to include negative examples (status=0)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    logger.info(f"Loading ReDSM5 data from {data_dir}")

    # Load data
    posts_path = os.path.join(data_dir, 'redsm5_posts.csv')
    annotations_path = os.path.join(data_dir, 'redsm5_annotations.csv')

    posts_df = pd.read_csv(posts_path)
    annotations_df = pd.read_csv(annotations_path)

    logger.info(f"Loaded {len(posts_df)} posts and {len(annotations_df)} annotations")

    # Prepare examples
    examples = prepare_evidence_examples(
        posts_df,
        annotations_df,
        include_negatives=include_negatives,
    )

    # Stratified split by symptom
    symptom_indices = [ex['symptom_idx'] for ex in examples]

    # Train/temp split
    train_examples, temp_examples = train_test_split(
        examples,
        test_size=(test_size + val_size),
        stratify=symptom_indices,
        random_state=random_seed,
    )

    # Val/test split
    temp_symptom_indices = [ex['symptom_idx'] for ex in temp_examples]
    val_size_adjusted = val_size / (test_size + val_size)

    val_examples, test_examples = train_test_split(
        temp_examples,
        test_size=(1 - val_size_adjusted),
        stratify=temp_symptom_indices,
        random_state=random_seed,
    )

    logger.info(f"Split: {len(train_examples)} train, {len(val_examples)} val, {len(test_examples)} test")

    # Create datasets
    cache_root = None
    if use_cached_dataset:
        cache_root = Path(cache_dir) if cache_dir else Path(data_dir) / 'cache'

    train_dataset = build_evidence_dataset(
        train_examples,
        tokenizer,
        max_length,
        use_cached_dataset=use_cached_dataset,
        cache_dir=str(cache_root) if cache_root else None,
        split_name='train',
        overwrite_cache=overwrite_cache,
    )
    val_dataset = build_evidence_dataset(
        val_examples,
        tokenizer,
        max_length,
        use_cached_dataset=use_cached_dataset,
        cache_dir=str(cache_root) if cache_root else None,
        split_name='val',
        overwrite_cache=overwrite_cache,
    )
    test_dataset = build_evidence_dataset(
        test_examples,
        tokenizer,
        max_length,
        use_cached_dataset=use_cached_dataset,
        cache_dir=str(cache_root) if cache_root else None,
        split_name='test',
        overwrite_cache=overwrite_cache,
    )

    return train_dataset, val_dataset, test_dataset


def get_symptom_labels() -> List[str]:
    """Return list of DSM-5 symptom labels."""
    return SYMPTOM_LABELS.copy()


def get_symptom_distribution(dataset: EvidenceDataset) -> pd.DataFrame:
    """
    Get symptom distribution in dataset.

    Args:
        dataset: EvidenceDataset instance

    Returns:
        DataFrame with columns [symptom, count, percentage]
    """
    symptom_counts = {}
    for example in dataset.examples:
        symptom = SYMPTOM_LABELS[example['symptom_idx']]
        symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1

    total = len(dataset)
    dist_data = [
        {
            'symptom': symptom,
            'count': count,
            'percentage': (count / total) * 100
        }
        for symptom, count in sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)
    ]

    return pd.DataFrame(dist_data)
