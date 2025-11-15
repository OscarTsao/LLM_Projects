"""Dataset loading from Hugging Face Datasets with validation.

This module provides dataset loading functionality that strictly uses Hugging Face
Datasets with explicit splits, no local CSV fallbacks.

Implements FR-019: Dataset loading from Hugging Face
Implements FR-026: Dataset identifier validation and revision pinning
Implements FR-029: Dataset failure handling
"""

import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
from datasets import load_dataset, DatasetDict
import hashlib

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
        if '/' not in dataset_id:
            raise DatasetLoadError(
                f"Invalid dataset identifier format: {dataset_id}. "
                "Expected format: 'organization/dataset-name' (e.g., 'irlab-udc/redsm5')"
            )
        
        return True
    
    @staticmethod
    def validate_splits(
        dataset: DatasetDict,
        required_splits: Tuple[str, ...] = ("train", "validation", "test")
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
    def validate_split_disjointness(
        dataset: DatasetDict,
        id_column: str = "post_id"
    ) -> bool:
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
            for split2 in available_splits[i+1:]:
                if split2 not in split_ids:
                    continue
                overlap = split_ids[split1] & split_ids[split2]
                if overlap:
                    overlaps.append((split1, split2, len(overlap)))
        
        if overlaps:
            overlap_msg = "; ".join(
                f"{s1}-{s2}: {count} overlapping IDs"
                for s1, s2, count in overlaps
            )
            raise DatasetLoadError(
                f"Splits are not disjoint! {overlap_msg}. "
                "This violates data integrity requirements."
            )
        
        return True


def load_hf_dataset(
    dataset_id: str = "irlab-udc/redsm5",
    revision: Optional[str] = "main",
    cache_dir: Optional[str] = None,
    required_splits: Tuple[str, ...] = ("train", "validation", "test"),
    validate_disjoint: bool = True
) -> Tuple[DatasetDict, Dict[str, str]]:
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
        f"Loading dataset: {dataset_id} "
        f"(revision: {revision if revision else 'latest'})"
    )
    
    try:
        # Load dataset from Hugging Face (FR-019)
        dataset = load_dataset(
            dataset_id,
            revision=revision,
            cache_dir=cache_dir
        )
        
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
        "num_examples": {split: len(dataset[split]) for split in dataset.keys()}
    }
    
    logger.info(
        f"Dataset loaded successfully: {dataset_id} "
        f"(hash: {resolved_hash}, splits: {list(dataset.keys())})"
    )
    
    return dataset, metadata

