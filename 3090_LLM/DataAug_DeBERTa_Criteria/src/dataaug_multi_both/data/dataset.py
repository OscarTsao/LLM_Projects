"""Dataset loading from local CSV files with validation.

This module provides dataset loading functionality that uses local CSV files
from Data/redsm5 directory with explicit train/validation/test splits.

Implements FR-019: Dataset loading from local files
Implements FR-026: Dataset identifier validation
Implements FR-029: Dataset failure handling
"""

import csv
import hashlib
import logging
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset

from dataaug_multi_both.data.augmentation import AugmentationConfig, TextAugmenter

logger = logging.getLogger(__name__)


class Dataset:
    """Simple dataset class to replace Hugging Face Dataset."""

    def __init__(self, data: list[dict[str, Any]]):
        """Initialize dataset with list of dictionaries.

        Args:
            data: List of dictionaries where each dict is one example
        """
        self.data = data
        self._column_names = list(data[0].keys()) if data else []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, str):
            # Column access: return list of values for that column
            return [item[key] for item in self.data]
        elif isinstance(key, slice):
            return Dataset(self.data[key])
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    @property
    def column_names(self) -> list[str]:
        return self._column_names

    def filter(self, func):
        """Filter dataset based on a function."""
        filtered_data = [item for item in self.data if func(item)]
        return Dataset(filtered_data)

    @classmethod
    def from_list(cls, data: list[dict[str, Any]]) -> "Dataset":
        """Create dataset from list of dictionaries."""
        return cls(data)


class DatasetDict(dict):
    """Simple DatasetDict class to replace Hugging Face DatasetDict."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if not isinstance(value, Dataset):
            raise TypeError(f"Value must be a Dataset instance, got {type(value)}")
        super().__setitem__(key, value)


class DatasetLoadError(Exception):
    """Exception raised when dataset loading fails."""

    pass


class DatasetValidator:
    """Validator for dataset configuration and splits."""

    @staticmethod
    def validate_dataset_id(dataset_id: str) -> bool:
        """Validate dataset identifier format.

        Args:
            dataset_id: Dataset identifier (e.g., "redsm5" or "irlab-udc/redsm5")

        Returns:
            True if valid format

        Raises:
            DatasetLoadError: If format is invalid
        """
        if not dataset_id or not isinstance(dataset_id, str):
            raise DatasetLoadError(
                f"Invalid dataset identifier: {dataset_id}. "
                "Expected a non-empty string"
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
    dataset_id: str = "redsm5",
    revision: str | None = None,
    cache_dir: str | None = None,
    required_splits: tuple[str, ...] = ("train", "validation", "test"),
    validate_disjoint: bool = True,
) -> tuple[DatasetDict, dict[str, str]]:
    """Load dataset from local CSV files with validation.

    Implements FR-019, FR-026, FR-029.

    Args:
        dataset_id: Dataset identifier (default: redsm5) - accepts "redsm5" or "irlab-udc/redsm5"
        revision: Ignored (kept for API compatibility)
        cache_dir: Ignored (kept for API compatibility)
        required_splits: Required split names
        validate_disjoint: Whether to validate split disjointness

    Returns:
        Tuple of (dataset, metadata) where metadata contains:
            - dataset_id: Dataset identifier
            - revision: Always "local"
            - resolved_hash: Resolved dataset hash

    Raises:
        DatasetLoadError: If loading or validation fails

    Example:
        dataset, metadata = load_hf_dataset("redsm5")
        train_data = dataset["train"]
    """
    # Validate dataset ID format (FR-026)
    DatasetValidator.validate_dataset_id(dataset_id)

    # Normalize dataset_id to handle both "redsm5" and "irlab-udc/redsm5"
    if dataset_id in ("redsm5", "irlab-udc/redsm5"):
        dataset_id = "redsm5"
    else:
        raise DatasetLoadError(
            f"Unknown dataset: {dataset_id}. Only 'redsm5' is supported."
        )

    logger.info(f"Loading dataset from local CSV files: {dataset_id}")

    # Load from local directory
    local_dir = Path("Data/redsm5")
    try:
        dataset = _build_redsm5_from_local(local_dir)
    except DatasetLoadError as exc:
        raise DatasetLoadError(
            f"Failed to load dataset '{dataset_id}' from {local_dir}. "
            f"Reason: {exc}. "
            f"Remediation: ensure CSV files exist in {local_dir}."
        ) from exc

    # Validate required splits exist (FR-026)
    DatasetValidator.validate_splits(dataset, required_splits)

    # Validate split disjointness (FR-026)
    if validate_disjoint:
        DatasetValidator.validate_split_disjointness(dataset)

    # Calculate dataset hash for reproducibility (FR-026)
    dataset_info = str(sorted(dataset.keys())) + str({k: len(dataset[k]) for k in dataset.keys()})
    resolved_hash = hashlib.sha256(dataset_info.encode()).hexdigest()[:16]

    metadata = {
        "dataset_id": dataset_id,
        "revision": "local",
        "resolved_hash": resolved_hash,
        "splits": list(dataset.keys()),
        "num_examples": {split: len(dataset[split]) for split in dataset.keys()},
    }

    logger.info(
        f"Dataset loaded successfully: {dataset_id} "
        f"(hash: {resolved_hash}, splits: {list(dataset.keys())})"
    )

    return dataset, metadata


def _build_redsm5_from_local(local_dir: Path | None) -> DatasetDict:
    """Build dataset from local CSV files.

    Args:
        local_dir: Path to directory containing CSV files

    Returns:
        DatasetDict with train/validation/test splits

    Raises:
        DatasetLoadError: If CSV files are missing or invalid
    """
    if local_dir is None:
        raise DatasetLoadError("Local RedSM5 directory not provided.")

    posts_path = local_dir / "redsm5_posts.csv"
    ann_path = local_dir / "redsm5_annotations.csv"
    if not posts_path.exists() or not ann_path.exists():
        raise DatasetLoadError(
            f"Local RedSM5 CSVs not found at {local_dir}. "
            "Expected files: redsm5_posts.csv and redsm5_annotations.csv"
        )

    # Load posts CSV
    with posts_path.open("r", encoding="utf-8") as f:
        posts_reader = csv.DictReader(f)
        posts_rows = list(posts_reader)

    # Load annotations CSV
    with ann_path.open("r", encoding="utf-8") as f:
        ann_reader = csv.DictReader(f)
        annotations_rows = list(ann_reader)

    # Validate CSV structure
    if not posts_rows or "post_id" not in posts_rows[0] or "text" not in posts_rows[0]:
        raise DatasetLoadError("Local posts CSV missing required columns 'post_id' and 'text'.")
    if annotations_rows and ("post_id" not in annotations_rows[0] or "DSM5_symptom" not in annotations_rows[0]):
        raise DatasetLoadError("Local annotations CSV missing required columns 'post_id' and 'DSM5_symptom'.")

    # DSM-5 symptom mapping
    symptom_to_idx = {
        "SLEEP_ISSUES": 0,
        "ANHEDONIA": 1,
        "APPETITE_CHANGE": 2,
        "FATIGUE": 3,
        "WORTHLESSNESS": 4,
        "COGNITIVE_ISSUES": 5,
        "PSYCHOMOTOR": 6,
        "SUICIDAL_THOUGHTS": 7,
        "DEPRESSED_MOOD": 8,
    }

    # Build post annotations mapping
    post_annotations: dict[str, list[int]] = defaultdict(list)
    for item in annotations_rows:
        if int(item.get("status", 0)) != 1:
            continue
        post_id = item["post_id"]
        symptom = item.get("DSM5_symptom")
        if symptom in symptom_to_idx:
            post_annotations[post_id].append(symptom_to_idx[symptom])

    # Create dataset with labels
    posts_data = []
    for post in posts_rows:
        post_id = post["post_id"]
        labels = [0] * 9
        for idx in post_annotations.get(post_id, []):
            labels[idx] = 1
        posts_data.append(
            {"post_id": post_id, "post_text": post["text"], "criteria_labels": labels}
        )

    # Create full dataset
    full_dataset = Dataset.from_list(posts_data)

    # Split dataset into train/val/test (80/10/10)
    post_ids = [item["post_id"] for item in posts_data]
    random.seed(42)
    random.shuffle(post_ids)

    n_train = int(0.8 * len(post_ids))
    n_val = int(0.1 * len(post_ids))
    train_ids = set(post_ids[:n_train])
    val_ids = set(post_ids[n_train : n_train + n_val])
    test_ids = set(post_ids[n_train + n_val :])

    # Filter datasets by split
    train_dataset = full_dataset.filter(lambda x: x["post_id"] in train_ids)
    val_dataset = full_dataset.filter(lambda x: x["post_id"] in val_ids)
    test_dataset = full_dataset.filter(lambda x: x["post_id"] in test_ids)

    logger.info(
        f"Created splits: train={len(train_dataset)}, validation={len(val_dataset)}, test={len(test_dataset)}"
    )

    return DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})


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

        # Load persistent augmentation cache (if available)
        self.persistent_cache = self._load_persistent_cache()
        self.use_persistent_cache = self.persistent_cache is not None

        if self.use_persistent_cache:
            logger.info(f"Loaded persistent augmentation cache with {len(self.persistent_cache)} entries")
            logger.info(f"Using cached augmentations for methods: {self.augmentation_methods}")
        else:
            logger.warning(
                "Persistent cache not found at experiments/augmentation_cache.pkl. "
                "Run scripts/precompute_augmentations.py first for optimal performance."
            )

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

                # Assume 9 criteria for mental health detection
                for criterion_idx in range(9):
                    criterion_label = (
                        1
                        if criterion_idx < len(criteria_labels) and criteria_labels[criterion_idx]
                        else 0
                    )

                    examples.append(
                        {
                            "text": post_text,
                            "criterion_id": criterion_idx,
                            "criterion_label": criterion_label,
                        }
                    )

            else:  # multi_label
                # Multi-label format: one example per post
                post_text = item.get("post_text", "")
                criteria_labels = item.get("criteria_labels", [0] * 9)

                # Pad or truncate criteria labels to 9
                if len(criteria_labels) < 9:
                    criteria_labels.extend([0] * (9 - len(criteria_labels)))
                elif len(criteria_labels) > 9:
                    criteria_labels = criteria_labels[:9]

                examples.append(
                    {
                        "text": post_text,
                        "criteria_labels": criteria_labels,
                    }
                )

        return examples

    def _load_persistent_cache(self) -> dict | None:
        """Load persistent augmentation cache from disk.

        Returns:
            Dictionary mapping (text_idx, method) to augmented text, or None if not found
        """
        cache_path = Path("experiments/augmentation_cache.pkl")
        if not cache_path.exists():
            return None

        try:
            with cache_path.open("rb") as f:
                cache = pickle.load(f)
            logger.info(f"Loaded persistent cache from {cache_path}")
            return cache
        except Exception as e:
            logger.error(f"Failed to load persistent cache: {e}")
            return None

    def _get_text_index(self, idx: int) -> int:
        """Get the original dataset text index for a given example index.

        For multi_label format: idx corresponds directly to dataset index
        For binary_pairs format: idx // 9 gives the dataset index
        """
        if self.input_format == "multi_label":
            return idx
        else:  # binary_pairs
            return idx // 9

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single example."""
        example = self.examples[idx]
        text = example["text"]

        # Apply augmentation using persistent cache (if available)
        if (
            self.augmentation_prob > 0
            and self.augmentation_methods
            and torch.rand(1).item() < self.augmentation_prob
        ):
            if self.use_persistent_cache:
                # Randomly select one method from the trial's selected methods
                method = random.choice(self.augmentation_methods)

                # Get the original text index
                text_idx = self._get_text_index(idx)

                # Look up cached augmentation (instant!)
                cache_key = (text_idx, method)
                text = self.persistent_cache.get(cache_key, text)

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

        # Single-task classification: no evidence span positions
        return output

    def _apply_augmentation(self, text: str) -> str:
        """Apply text augmentation using the configured augmenter."""
        if self.augmenter is None or not text:
            return text

        try:
            # Use augment_evidence which handles probability internally
            augmented_text, metadata = self.augmenter.augment_evidence(text)

            # Log if augmentation was applied (at debug level to avoid spam)
            if metadata.get("augmentation_applied", False):
                logger.debug(f"Applied augmentation using method: {metadata.get('method_used')}")

            return augmented_text
        except Exception as e:
            logger.warning(f"Augmentation failed, using original text: {e}")
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
