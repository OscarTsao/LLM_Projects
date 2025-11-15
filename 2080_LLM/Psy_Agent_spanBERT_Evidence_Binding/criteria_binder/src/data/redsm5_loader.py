# File: src/data/redsm5_loader.py
"""Data loader for RedSM5 dataset format."""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)


class RedSM5DataLoader:
    """Loader for RedSM5 dataset with posts, annotations, and DSM-5 criteria."""

    def __init__(
        self,
        dsm_criteria_path: str,
        posts_path: str,
        annotations_path: str,
    ):
        """Initialize the RedSM5 data loader.

        Args:
            dsm_criteria_path: Path to DSM-5 criteria JSON file
            posts_path: Path to posts CSV file
            annotations_path: Path to annotations CSV file
        """
        self.dsm_criteria_path = Path(dsm_criteria_path)
        self.posts_path = Path(posts_path)
        self.annotations_path = Path(annotations_path)

        # Load data
        self.dsm_criteria = self._load_dsm_criteria()
        self.posts_df = self._load_posts()
        self.annotations_df = self._load_annotations()

        # Create mappings
        self.dsm_symptom_to_criteria = self._create_symptom_mapping()

    def _load_dsm_criteria(self) -> Dict[str, Any]:
        """Load DSM-5 criteria from JSON file."""
        if not self.dsm_criteria_path.exists():
            raise FileNotFoundError(f"DSM criteria file not found: {self.dsm_criteria_path}")

        with open(self.dsm_criteria_path, 'r', encoding='utf-8') as f:
            criteria_data = json.load(f)

        logger.info(f"Loaded DSM-5 criteria from {self.dsm_criteria_path}")
        return criteria_data

    def _load_posts(self) -> pd.DataFrame:
        """Load posts from CSV file."""
        if not self.posts_path.exists():
            raise FileNotFoundError(f"Posts file not found: {self.posts_path}")

        posts_df = pd.read_csv(self.posts_path)
        logger.info(f"Loaded {len(posts_df)} posts from {self.posts_path}")
        return posts_df

    def _load_annotations(self) -> pd.DataFrame:
        """Load annotations from CSV file."""
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")

        annotations_df = pd.read_csv(self.annotations_path)
        logger.info(f"Loaded {len(annotations_df)} annotations from {self.annotations_path}")
        return annotations_df

    def _create_symptom_mapping(self) -> Dict[str, str]:
        """Create mapping from DSM5_symptom codes to criteria text."""
        mapping = {}

        # For Major Depressive Disorder criteria
        if isinstance(self.dsm_criteria, list) and len(self.dsm_criteria) > 0:
            disorder_data = self.dsm_criteria[0]  # Assuming first entry is MDD
            criteria_list = disorder_data.get('criteria', [])

            # Create mapping based on common symptom names to criteria IDs
            symptom_to_criteria_id = {
                'DEPRESSED_MOOD': 'A.1',
                'ANHEDONIA': 'A.2',
                'APPETITE_WEIGHT': 'A.3',
                'APPETITE_CHANGE': 'A.3',  # Alternative naming
                'SLEEP_ISSUES': 'A.4',
                'PSYCHOMOTOR': 'A.5',
                'FATIGUE': 'A.6',
                'WORTHLESSNESS': 'A.7',
                'COGNITIVE_ISSUES': 'A.8',
                'SUICIDE_THOUGHTS': 'A.9',
                'SUICIDAL_THOUGHTS': 'A.9',  # Alternative naming
                'SPECIAL_CASE': 'A.1'  # Default to depressed mood for special cases
            }

            # Map symptom codes to actual criteria text
            for symptom, criteria_id in symptom_to_criteria_id.items():
                for criterion in criteria_list:
                    if criterion.get('id') == criteria_id:
                        mapping[symptom] = criterion.get('text', '')
                        break

        logger.info(f"Created symptom mapping for {len(mapping)} symptoms")
        return mapping

    def get_post_text(self, post_id: str) -> Optional[str]:
        """Get text for a specific post ID."""
        post_row = self.posts_df[self.posts_df['post_id'] == post_id]
        if len(post_row) > 0:
            return post_row.iloc[0]['text']
        return None

    def get_positive_annotations(self) -> pd.DataFrame:
        """Get annotations where status == 1 (criterion is matched)."""
        return self.annotations_df[self.annotations_df['status'] == 1].copy()

    def get_negative_annotations(self) -> pd.DataFrame:
        """Get annotations where status == 0 (criterion is not matched)."""
        return self.annotations_df[self.annotations_df['status'] == 0].copy()

    def create_training_examples(self, include_negatives: bool = True) -> List[Dict[str, Any]]:
        """Create training examples in the format expected by CriteriaBindingDataset.

        Args:
            include_negatives: Whether to include negative examples (status=0)

        Returns:
            List of examples with required fields for training
        """
        examples = []

        # Get relevant annotations
        if include_negatives:
            relevant_annotations = self.annotations_df.copy()
        else:
            relevant_annotations = self.get_positive_annotations()

        for _, annotation in relevant_annotations.iterrows():
            post_id = annotation['post_id']
            sentence_id = annotation['sentence_id']
            sentence_text = annotation['sentence_text']
            dsm5_symptom = annotation['DSM5_symptom']
            status = annotation['status']

            # Get full post text
            post_text = self.get_post_text(post_id)
            if post_text is None:
                logger.warning(f"Post text not found for post_id: {post_id}")
                continue

            # Get criterion text for this symptom
            criterion_text = self.dsm_symptom_to_criteria.get(dsm5_symptom, '')
            if not criterion_text:
                logger.warning(f"Criterion text not found for symptom: {dsm5_symptom}")
                continue

            # Find character spans of evidence sentence in full post
            evidence_spans = []
            if status == 1 and sentence_text.strip():
                # Find the sentence in the post text
                sentence_start = post_text.find(sentence_text.strip())
                if sentence_start != -1:
                    sentence_end = sentence_start + len(sentence_text.strip())
                    evidence_spans = [(sentence_start, sentence_end)]
                else:
                    logger.warning(f"Evidence sentence not found in post for {sentence_id}")

            example = {
                'id': f"{post_id}_{dsm5_symptom}_{sentence_id}",
                'criterion_text': criterion_text,
                'document_text': post_text,
                'label': int(status),  # 1 for matched, 0 for not matched
                'evidence_char_spans': evidence_spans,
                'post_id': post_id,
                'sentence_id': sentence_id,
                'dsm5_symptom': dsm5_symptom,
                'sentence_text': sentence_text,
            }

            examples.append(example)

        logger.info(f"Created {len(examples)} training examples")
        return examples

    def get_unique_posts(self) -> Set[str]:
        """Get set of unique post IDs from annotations."""
        return set(self.annotations_df['post_id'].unique())

    def get_unique_symptoms(self) -> Set[str]:
        """Get set of unique DSM5 symptoms from annotations."""
        return set(self.annotations_df['DSM5_symptom'].unique())

    def split_data(
        self,
        train_ratio: float = 0.7,
        dev_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split_by_post: bool = True,
        random_seed: int = 42
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Split data into train/dev/test sets.

        Args:
            train_ratio: Fraction for training set
            dev_ratio: Fraction for dev set
            test_ratio: Fraction for test set
            split_by_post: If True, split by post_id to avoid data leakage
            random_seed: Random seed for reproducible splits

        Returns:
            Dictionary with 'train', 'dev', 'test' keys containing examples
        """
        import random
        random.seed(random_seed)

        # Validate ratios
        if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, dev, and test ratios must sum to 1.0")

        # Get all examples
        all_examples = self.create_training_examples(include_negatives=True)

        if split_by_post:
            # Split by unique posts to avoid data leakage
            unique_posts = list(self.get_unique_posts())
            random.shuffle(unique_posts)

            n_posts = len(unique_posts)
            n_train = int(n_posts * train_ratio)
            n_dev = int(n_posts * dev_ratio)

            train_posts = set(unique_posts[:n_train])
            dev_posts = set(unique_posts[n_train:n_train + n_dev])
            test_posts = set(unique_posts[n_train + n_dev:])

            # Assign examples to splits based on post_id
            train_examples = [ex for ex in all_examples if ex['post_id'] in train_posts]
            dev_examples = [ex for ex in all_examples if ex['post_id'] in dev_posts]
            test_examples = [ex for ex in all_examples if ex['post_id'] in test_posts]

        else:
            # Random split of examples
            random.shuffle(all_examples)
            n_examples = len(all_examples)
            n_train = int(n_examples * train_ratio)
            n_dev = int(n_examples * dev_ratio)

            train_examples = all_examples[:n_train]
            dev_examples = all_examples[n_train:n_train + n_dev]
            test_examples = all_examples[n_train + n_dev:]

        logger.info(f"Split data: {len(train_examples)} train, {len(dev_examples)} dev, {len(test_examples)} test")

        return {
            'train': train_examples,
            'dev': dev_examples,
            'test': test_examples
        }

    def save_split_to_jsonl(
        self,
        split_data: Dict[str, List[Dict[str, Any]]],
        output_dir: str
    ) -> None:
        """Save split data to JSONL files.

        Args:
            split_data: Dictionary with train/dev/test splits
            output_dir: Directory to save JSONL files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for split_name, examples in split_data.items():
            output_file = output_path / f"{split_name}.jsonl"

            with open(output_file, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')

            logger.info(f"Saved {len(examples)} examples to {output_file}")