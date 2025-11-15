"""Dataset loading from Hugging Face Datasets with validation.

This module provides dataset loading functionality that strictly uses Hugging Face
Datasets with explicit splits, no local CSV fallbacks.

Implements FR-019: Dataset loading from Hugging Face
Implements FR-026: Dataset identifier validation and revision pinning
Implements FR-029: Dataset failure handling
"""

import hashlib
import logging
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset

logger = logging.getLogger(__name__)


class DatasetLoadError(Exception):
    """Exception raised when dataset loading fails."""

    pass


class DatasetValidator:
    """Validator for dataset configuration and splits."""

    @staticmethod
    def validate_dataset_id(dataset_id: str) -> bool:
        """Validate dataset identifier format.

        Args:
            dataset_id: Dataset identifier (e.g., "irlab-udc/redsm5")

        Returns:
            True if valid format

        Raises:
            DatasetLoadError: If format is invalid
        """
        if not dataset_id or not isinstance(dataset_id, str):
            raise DatasetLoadError(
                f"Invalid dataset identifier: {dataset_id}. "
                "Expected format: 'organization/dataset-name'"
            )

        # Basic format validation: should contain '/'
        if "/" not in dataset_id:
            raise DatasetLoadError(
                f"Invalid dataset identifier format: {dataset_id}. "
                "Expected format: 'organization/dataset-name' (e.g., 'irlab-udc/redsm5')"
            )

        return True

    @staticmethod
    def validate_splits(
        dataset: DatasetDict, required_splits: tuple[str, ...] = ("train", "validation", "test")
    ) -> bool:
        """Validate that required splits exist.

        Args:
            dataset: Loaded dataset
            required_splits: Tuple of required split names

        Returns:
            True if all required splits exist

        Raises:
            DatasetLoadError: If required splits are missing
        """
        available_splits = set(dataset.keys())
        required_set = set(required_splits)
        missing_splits = required_set - available_splits

        if missing_splits:
            raise DatasetLoadError(
                f"Missing required splits: {sorted(missing_splits)}. "
                f"Required: {sorted(required_splits)}. "
                f"Available: {sorted(available_splits)}. "
                f"Remediation: Verify dataset identifier and ensure all splits are present."
            )

        return True

    @staticmethod
    def validate_split_disjointness(dataset: DatasetDict, id_column: str = "post_id") -> bool:
        """Validate that splits are disjoint (no overlapping IDs).

        Args:
            dataset: Loaded dataset
            id_column: Column name for unique identifiers

        Returns:
            True if splits are disjoint

        Raises:
            DatasetLoadError: If splits overlap
        """
        splits_to_check = ["train", "validation", "test"]
        available_splits = [s for s in splits_to_check if s in dataset]

        if len(available_splits) < 2:
            return True  # Nothing to check

        # Collect IDs from each split
        split_ids = {}
        for split_name in available_splits:
            if id_column in dataset[split_name].column_names:
                split_ids[split_name] = set(dataset[split_name][id_column])
            else:
                logger.warning(
                    f"ID column '{id_column}' not found in split '{split_name}'. "
                    "Skipping disjointness check for this split."
                )

        # Check for overlaps
        overlaps = []
        for i, split1 in enumerate(available_splits):
            if split1 not in split_ids:
                continue
            for split2 in available_splits[i + 1 :]:
                if split2 not in split_ids:
                    continue
                overlap = split_ids[split1] & split_ids[split2]
                if overlap:
                    overlaps.append((split1, split2, len(overlap)))

        if overlaps:
            overlap_msg = "; ".join(
                f"{s1}-{s2}: {count} overlapping IDs" for s1, s2, count in overlaps
            )
            raise DatasetLoadError(
                f"Splits are not disjoint! {overlap_msg}. "
                "This violates data integrity requirements."
            )

        return True


def load_hf_dataset(
    dataset_id: str = "irlab-udc/redsm5",
    revision: str | None = "main",
    cache_dir: str | None = None,
    required_splits: tuple[str, ...] = ("train", "validation", "test"),
    validate_disjoint: bool = True,
) -> tuple[DatasetDict, dict[str, str]]:
    """Load dataset from Hugging Face Datasets with validation.

    Implements FR-019, FR-026, FR-029.

    Args:
        dataset_id: Hugging Face dataset identifier (default: irlab-udc/redsm5)
        revision: Dataset revision/tag/commit (default: main)
        cache_dir: Cache directory for downloaded datasets
        required_splits: Required split names
        validate_disjoint: Whether to validate split disjointness

    Returns:
        Tuple of (dataset, metadata) where metadata contains:
            - dataset_id: Dataset identifier
            - revision: Requested revision
            - resolved_hash: Resolved dataset hash/commit

    Raises:
        DatasetLoadError: If loading or validation fails

    Example:
        dataset, metadata = load_hf_dataset("irlab-udc/redsm5", revision="main")
        train_data = dataset["train"]
    """
    # Validate dataset ID format (FR-026)
    DatasetValidator.validate_dataset_id(dataset_id)

    logger.info(
        f"Loading dataset: {dataset_id} " f"(revision: {revision if revision else 'latest'})"
    )

    try:
        # Load dataset from Hugging Face (FR-019)
        # Special handling for irlab-udc/redsm5 which has multiple CSV files with different schemas
        if dataset_id == "irlab-udc/redsm5":
            # Load both posts and annotations
            posts_dataset = load_dataset(
                dataset_id,
                data_files="redsm5_posts.csv",
                revision=revision,
                cache_dir=cache_dir,
                split="train",
            )
            annotations_dataset = load_dataset(
                dataset_id,
                data_files="redsm5_annotations.csv",
                revision=revision,
                cache_dir=cache_dir,
                split="train",
            )

            # Group annotations by post_id to create multi-label dataset
            from collections import defaultdict

            post_annotations = defaultdict(list)

            # Map DSM5 symptoms to indices (based on actual dataset values)
            symptom_to_idx = {
                "SLEEP_ISSUES": 0,
                "ANHEDONIA": 1,
                "APPETITE_CHANGE": 2,
                "FATIGUE": 3,
                "WORTHLESSNESS": 4,
                "COGNITIVE_ISSUES": 5,
                "PSYCHOMOTOR": 6,
                "SUICIDAL_THOUGHTS": 7,
                "DEPRESSED_MOOD": 8
                # Note: SPECIAL_CASE is excluded as it's not a standard DSM-5 criterion
            }

            # Collect annotations per post
            for item in annotations_dataset:
                if item["status"] == 1:  # Only positive annotations
                    post_id = item["post_id"]
                    symptom = item["DSM5_symptom"]
                    if symptom in symptom_to_idx:
                        post_annotations[post_id].append(symptom_to_idx[symptom])

            # Create post-level dataset with multi-label targets
            posts_data = []
            for post in posts_dataset:
                post_id = post["post_id"]
                # Create multi-label vector (9 labels)
                labels = [0] * 9
                if post_id in post_annotations:
                    for idx in post_annotations[post_id]:
                        labels[idx] = 1

                posts_data.append(
                    {"post_id": post_id, "post_text": post["text"], "criteria_labels": labels}
                )

            # Convert to HF Dataset
            from datasets import Dataset as HFDataset

            full_dataset = HFDataset.from_list(posts_data)

            # Get unique post IDs and split them to ensure no overlap
            post_ids = list(full_dataset["post_id"])

            # Shuffle and split post IDs (80/10/10)
            import random

            random.seed(42)
            random.shuffle(post_ids)

            n_train = int(0.8 * len(post_ids))
            n_val = int(0.1 * len(post_ids))

            train_ids = set(post_ids[:n_train])
            val_ids = set(post_ids[n_train : n_train + n_val])
            test_ids = set(post_ids[n_train + n_val :])

            # Filter dataset by post IDs
            train_dataset = full_dataset.filter(lambda x: x["post_id"] in train_ids)
            val_dataset = full_dataset.filter(lambda x: x["post_id"] in val_ids)
            test_dataset = full_dataset.filter(lambda x: x["post_id"] in test_ids)

            dataset = DatasetDict(
                {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
            )
        else:
            dataset = load_dataset(dataset_id, revision=revision, cache_dir=cache_dir)

        if not isinstance(dataset, DatasetDict):
            raise DatasetLoadError(
                f"Expected DatasetDict with splits, got {type(dataset)}. "
                "The dataset must have explicit train/validation/test splits."
            )

    except Exception as e:
        # FR-029: Actionable error on failure
        raise DatasetLoadError(
            f"Failed to load dataset '{dataset_id}' (revision: {revision}). "
            f"Error: {str(e)}. "
            f"Remediation: "
            f"1. Verify dataset identifier is correct. "
            f"2. Check internet connection. "
            f"3. Verify Hugging Face authentication if dataset is private. "
            f"4. Try without revision parameter to use latest version."
        ) from e

    # Validate required splits exist (FR-026)
    DatasetValidator.validate_splits(dataset, required_splits)

    # Validate split disjointness (FR-026)
    if validate_disjoint:
        DatasetValidator.validate_split_disjointness(dataset)

    # Calculate dataset hash for reproducibility (FR-026)
    # Use dataset info to create a deterministic hash
    dataset_info = str(dataset)
    resolved_hash = hashlib.sha256(dataset_info.encode()).hexdigest()[:16]

    metadata = {
        "dataset_id": dataset_id,
        "revision": revision if revision else "latest",
        "resolved_hash": resolved_hash,
        "splits": list(dataset.keys()),
        "num_examples": {split: len(dataset[split]) for split in dataset.keys()},
    }

    logger.info(
        f"Dataset loaded successfully: {dataset_id} "
        f"(hash: {resolved_hash}, splits: {list(dataset.keys())})"
    )

    return dataset, metadata


class RedSM5Dataset(TorchDataset):
    """PyTorch Dataset for RedSM5 mental health criteria detection.

    Supports both binary_pairs and multi_label input formats.
    """

    # DSM-5 criterion descriptions for next sentence prediction
    CRITERION_TEXTS = [
        "Sleep issues or insomnia",  # 0: SLEEP_ISSUES
        "Loss of interest or anhedonia",  # 1: ANHEDONIA
        "Appetite or weight change",  # 2: APPETITE_CHANGE
        "Fatigue or loss of energy",  # 3: FATIGUE
        "Worthlessness or guilt",  # 4: WORTHLESSNESS
        "Cognitive or concentration issues",  # 5: COGNITIVE_ISSUES
        "Psychomotor agitation or retardation",  # 6: PSYCHOMOTOR
        "Suicidal thoughts or ideation",  # 7: SUICIDAL_THOUGHTS
        "Depressed mood",  # 8: DEPRESSED_MOOD
    ]

    def __init__(
        self,
        hf_dataset: Dataset,
        tokenizer: Any,
        input_format: str = "binary_pairs",
        max_length: int = 512,
        augmentation_prob: float = 0.0,
        augmentation_methods: list[str] = None,
    ):
        """Initialize dataset.

        Args:
            hf_dataset: Hugging Face dataset
            tokenizer: Model tokenizer
            input_format: "binary_pairs" or "multi_label"
            max_length: Maximum sequence length
            augmentation_prob: Probability of applying augmentation
            augmentation_methods: List of augmentation methods to use
        """
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.input_format = input_format
        self.max_length = max_length
        self.augmentation_prob = augmentation_prob
        self.augmentation_methods = augmentation_methods or []

        # Validate input format
        if input_format not in ["binary_pairs", "multi_label"]:
            raise ValueError(f"Invalid input_format: {input_format}")

        # Prepare examples based on format
        self.examples = self._prepare_examples()

    def _prepare_examples(self) -> list[dict[str, Any]]:
        """Prepare examples based on input format."""
        examples = []

        for item in self.hf_dataset:
            if self.input_format == "binary_pairs":
                # Create binary pairs for each criterion
                post_text = item.get("post_text", "")
                criteria_labels = item.get("criteria_labels", [])
                evidence_spans = item.get("evidence_spans", [])

                # Assume 9 criteria for mental health detection
                for criterion_idx in range(9):
                    criterion_label = (
                        1
                        if criterion_idx < len(criteria_labels) and criteria_labels[criterion_idx]
                        else 0
                    )

                    # Find evidence for this criterion
                    criterion_evidence = []
                    for span in evidence_spans:
                        if span.get("criterion_id") == criterion_idx:
                            criterion_evidence.append(span)

                    # Create start/end positions
                    start_pos = 0
                    end_pos = 0
                    if criterion_evidence:
                        # Use first evidence span
                        start_pos = criterion_evidence[0].get("start", 0)
                        end_pos = criterion_evidence[0].get("end", 0)

                    examples.append(
                        {
                            "text": post_text,
                            "criterion_id": criterion_idx,
                            "criterion_label": criterion_label,
                            "start_position": start_pos,
                            "end_position": end_pos,
                        }
                    )

            else:  # multi_label
                # Multi-label format: one example per post
                post_text = item.get("post_text", "")
                criteria_labels = item.get("criteria_labels", [0] * 9)
                evidence_spans = item.get("evidence_spans", [])

                # Pad or truncate criteria labels to 9
                if len(criteria_labels) < 9:
                    criteria_labels.extend([0] * (9 - len(criteria_labels)))
                elif len(criteria_labels) > 9:
                    criteria_labels = criteria_labels[:9]

                # For multi-label, use first evidence span or default to 0
                start_pos = 0
                end_pos = 0
                if evidence_spans:
                    start_pos = evidence_spans[0].get("start", 0)
                    end_pos = evidence_spans[0].get("end", 0)

                examples.append(
                    {
                        "text": post_text,
                        "criteria_labels": criteria_labels,
                        "start_position": start_pos,
                        "end_position": end_pos,
                    }
                )

        return examples

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single example."""
        example = self.examples[idx]
        text = example["text"]

        # Apply augmentation if specified
        if self.augmentation_prob > 0 and torch.rand(1).item() < self.augmentation_prob:
            text = self._apply_augmentation(text)

        # Create next sentence prediction format
        if self.input_format == "binary_pairs":
            # Binary pairs: [CLS] post [SEP] criterion [SEP]
            criterion_id = example.get("criterion_id", 0)
            criterion_text = self.CRITERION_TEXTS[criterion_id] if criterion_id < len(self.CRITERION_TEXTS) else ""
            
            # Tokenize as sentence pair
            encoding = self.tokenizer(
                text,
                criterion_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            # Multi-label: [CLS] post [SEP] criterion1 [SEP] criterion2 ... [SEP] criterion9 [SEP]
            # Concatenate all criteria with SEP tokens
            all_criteria = " ".join(self.CRITERION_TEXTS)
            
            # Tokenize as sentence pair
            encoding = self.tokenizer(
                text,
                all_criteria,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

        # Prepare output
        output = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        
        # Add token_type_ids for NSP format (0 for post, 1 for criteria)
        if "token_type_ids" in encoding:
            output["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self.input_format == "binary_pairs":
            # Binary classification for single criterion - convert to multi-label format
            # Create a 9-dimensional vector with the single criterion label
            labels = [0.0] * 9  # Initialize all criteria as 0
            criterion_id = example.get("criterion_id", 0)
            if 0 <= criterion_id < 9:
                labels[criterion_id] = float(example["criterion_label"])
            output["criteria_labels"] = torch.tensor(labels, dtype=torch.float)
        else:
            # Multi-label classification
            output["criteria_labels"] = torch.tensor(example["criteria_labels"], dtype=torch.float)

        # Add evidence positions
        output["start_positions"] = torch.tensor(example["start_position"], dtype=torch.long)
        output["end_positions"] = torch.tensor(example["end_position"], dtype=torch.long)

        return output

    def _apply_augmentation(self, text: str) -> str:
        """Apply text augmentation (placeholder implementation)."""
        # Simple augmentation - in practice, would use TextAttack
        if "synonym" in self.augmentation_methods:
            # Placeholder: just return original text
            pass
        return text


def create_pytorch_dataset(
    hf_dataset: Dataset,
    tokenizer: Any,
    input_format: str = "binary_pairs",
    max_length: int = 512,
    augmentation_prob: float = 0.0,
    augmentation_methods: list[str] = None,
) -> RedSM5Dataset:
    """Create a PyTorch dataset from Hugging Face dataset.

    Args:
        hf_dataset: Hugging Face dataset
        tokenizer: Model tokenizer
        input_format: "binary_pairs" or "multi_label"
        max_length: Maximum sequence length
        augmentation_prob: Probability of applying augmentation
        augmentation_methods: List of augmentation methods

    Returns:
        PyTorch dataset

    Example:
        dataset = create_pytorch_dataset(
            hf_dataset["train"],
            tokenizer=tokenizer,
            input_format="binary_pairs"
        )
    """
    return RedSM5Dataset(
        hf_dataset=hf_dataset,
        tokenizer=tokenizer,
        input_format=input_format,
        max_length=max_length,
        augmentation_prob=augmentation_prob,
        augmentation_methods=augmentation_methods or [],
    )
