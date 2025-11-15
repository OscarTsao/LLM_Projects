"""Data loaders for ReDSM-5 dataset with STRICT validation.

Supports loading from:
- HuggingFace: 'irlab-udc/redsm5'
- Local CSVs: data/raw/redsm5/posts.csv and annotations.csv
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_field_map(config_path: str | Path) -> dict[str, Any]:
    """Load field mapping configuration.

    Args:
        config_path: Path to field_map.yaml

    Returns:
        Dictionary with field mappings
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


class ReDSM5DataLoader:
    """Loader for ReDSM-5 dataset with strict field validation.

    Enforces STRICT mapping rules:
    - status field -> criteria task ONLY
    - cases field -> evidence task ONLY

    Can load from:
    - HuggingFace dataset: 'irlab-udc/redsm5'
    - Local CSV files: posts.csv and annotations.csv
    """

    def __init__(
        self,
        field_map: dict[str, Any],
        data_source: str = "local",
        data_dir: str | Path | None = None,
        hf_dataset_name: str | None = None,
        hf_cache_dir: str | Path | None = None,
        posts_file: str = "posts.csv",
        annotations_file: str = "annotations.csv",
    ):
        """Initialize loader.

        Args:
            field_map: Field mapping configuration (from field_map.yaml)
            data_source: 'local' or 'huggingface'
            data_dir: Directory containing posts.csv and annotations.csv (for local)
            hf_dataset_name: HuggingFace dataset name (e.g., 'irlab-udc/redsm5')
        """
        self.field_map = field_map
        self.data_source = data_source.lower()
        self.data_dir = Path(data_dir) if data_dir else None
        self.hf_dataset_name = hf_dataset_name
        self.hf_cache_dir = Path(hf_cache_dir) if hf_cache_dir else None
        self.posts_file = posts_file
        self.annotations_file = annotations_file

        self._validate_config()

    def _validate_config(self):
        """Validate configuration."""
        if self.data_source == "local":
            if self.data_dir is None:
                raise ValueError("data_dir required for local data source")
            if not self.data_dir.exists():
                raise ValueError(f"Data directory not found: {self.data_dir}")
        elif self.data_source == "huggingface":
            if self.hf_dataset_name is None:
                raise ValueError("hf_dataset_name required for huggingface data source")
        else:
            raise ValueError(
                f"Invalid data_source: {self.data_source}. Must be 'local' or 'huggingface'"
            )

    def load_posts(self) -> pd.DataFrame:
        """Load posts data.

        Returns:
            DataFrame with posts
        """
        if self.data_source == "local":
            posts_path = self.data_dir / self.posts_file
            if not posts_path.exists():
                raise FileNotFoundError(f"Posts file not found: {posts_path}")
            df = pd.read_csv(posts_path)
        else:
            # Load from HuggingFace
            from datasets import load_dataset

            dataset_kwargs = {}
            if self.hf_cache_dir:
                dataset_kwargs["cache_dir"] = str(self.hf_cache_dir)
            dataset = load_dataset(self.hf_dataset_name, "posts", **dataset_kwargs)
            df = dataset["train"].to_pandas()

        # Validate required columns
        post_id_field = self.field_map["posts"]["post_id"]
        text_field = self.field_map["posts"]["text"]

        required = {post_id_field, text_field}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in posts: {missing}")

        return df

    def load_annotations(self) -> pd.DataFrame:
        """Load annotations data.

        Returns:
            DataFrame with annotations
        """
        if self.data_source == "local":
            annot_path = self.data_dir / self.annotations_file
            if not annot_path.exists():
                raise FileNotFoundError(f"Annotations file not found: {annot_path}")
            df = pd.read_csv(annot_path)
        else:
            # Load from HuggingFace
            from datasets import load_dataset

            dataset_kwargs = {}
            if self.hf_cache_dir:
                dataset_kwargs["cache_dir"] = str(self.hf_cache_dir)
            dataset = load_dataset(
                self.hf_dataset_name, "annotations", **dataset_kwargs
            )
            df = dataset["train"].to_pandas()

        # Validate required columns
        post_id_field = self.field_map["annotations"]["post_id"]
        criterion_id_field = self.field_map["annotations"]["criterion_id"]
        status_field = self.field_map["annotations"]["status"]
        cases_field = self.field_map["annotations"]["cases"]

        required = {post_id_field, criterion_id_field, status_field, cases_field}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in annotations: {missing}")

        return df

    def load_dsm_criteria(self, dsm_path: str | Path) -> list[dict[str, str]]:
        """Load DSM criteria from JSON.

        Args:
            dsm_path: Path to dsm_criteria.json

        Returns:
            List of criterion dictionaries with 'id' and 'text'
        """
        with open(dsm_path, encoding="utf-8") as f:
            criteria = json.load(f)

        # Validate structure
        for criterion in criteria:
            if "id" not in criterion or "text" not in criterion:
                raise ValueError(
                    f"Invalid criterion structure: {criterion}. "
                    "Must have 'id' and 'text' keys"
                )

        return criteria

    def get_valid_criterion_ids(self, dsm_path: str | Path) -> set[str]:
        """Get set of valid criterion IDs from DSM criteria.

        Args:
            dsm_path: Path to dsm_criteria.json

        Returns:
            Set of criterion IDs
        """
        criteria = self.load_dsm_criteria(dsm_path)
        return {c["id"] for c in criteria}


def group_split_by_post_id(
    df: pd.DataFrame,
    post_id_col: str = "post_id",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Split data by post_id to prevent data leakage.

    All annotations for a given post_id will be in the same split.

    Args:
        df: DataFrame with post_id column
        post_id_col: Name of post_id column
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for test
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_post_ids, val_post_ids, test_post_ids)

    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    # Get unique post IDs
    unique_post_ids = df[post_id_col].unique()

    # First split: train vs (val + test)
    train_ids, temp_ids = train_test_split(
        unique_post_ids, test_size=(val_ratio + test_ratio), random_state=random_seed
    )

    # Second split: val vs test
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=val_test_ratio, random_state=random_seed
    )

    return train_ids.tolist(), val_ids.tolist(), test_ids.tolist()


def save_splits_json(
    train_post_ids: list[str],
    val_post_ids: list[str],
    test_post_ids: list[str],
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
):
    """Save splits to JSON for reproducibility.

    Args:
        train_post_ids: List of training post IDs
        val_post_ids: List of validation post IDs
        test_post_ids: List of test post IDs
        output_path: Path to save JSON
        metadata: Optional metadata to include
    """
    splits_data = {
        "train": train_post_ids,
        "val": val_post_ids,
        "test": test_post_ids,
        "metadata": metadata or {},
    }

    splits_data["metadata"].update(
        {
            "train_count": len(train_post_ids),
            "val_count": len(val_post_ids),
            "test_count": len(test_post_ids),
            "total_count": len(train_post_ids) + len(val_post_ids) + len(test_post_ids),
        }
    )

    with open(output_path, "w") as f:
        json.dump(splits_data, f, indent=2)

    print(f"Saved splits to {output_path}")
    print(f"  Train: {len(train_post_ids)} posts")
    print(f"  Val: {len(val_post_ids)} posts")
    print(f"  Test: {len(test_post_ids)} posts")


def load_splits_json(splits_path: str | Path) -> tuple[list[str], list[str], list[str]]:
    """Load splits from JSON.

    Args:
        splits_path: Path to splits JSON

    Returns:
        Tuple of (train_post_ids, val_post_ids, test_post_ids)
    """
    with open(splits_path) as f:
        splits_data = json.load(f)

    train_ids = splits_data["train"]
    val_ids = splits_data["val"]
    test_ids = splits_data["test"]

    # Validate no overlap
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            f"Data leakage detected in splits! "
            f"Train-Val overlap: {len(overlap_train_val)}, "
            f"Train-Test overlap: {len(overlap_train_test)}, "
            f"Val-Test overlap: {len(overlap_val_test)}"
        )

    print(f"Loaded splits from {splits_path}")
    print(f"  Train: {len(train_ids)} posts")
    print(f"  Val: {len(val_ids)} posts")
    print(f"  Test: {len(test_ids)} posts")

    return train_ids, val_ids, test_ids


class DSMCriteriaLoader:
    """Loader for DSM-5 criteria knowledge base."""

    def __init__(self, dsm_path: str | Path):
        """Initialize DSM criteria loader.

        Args:
            dsm_path: Path to DSM criteria JSON file
        """
        self.dsm_path = Path(dsm_path)
        if not self.dsm_path.exists():
            raise FileNotFoundError(f"DSM criteria file not found: {self.dsm_path}")

    def load_criteria(self) -> list[dict[str, str]]:
        """Load DSM-5 criteria.

        Returns:
            List of criteria dictionaries with 'id' and 'text' keys
        """
        with open(self.dsm_path, encoding="utf-8") as f:
            criteria = json.load(f)

        # Validate structure
        for criterion in criteria:
            if "id" not in criterion or "text" not in criterion:
                raise ValueError(
                    f"Invalid criterion structure: {criterion}. "
                    "Must have 'id' and 'text' keys"
                )

        return criteria

    def get_criterion_by_id(self, criterion_id: str) -> dict[str, str] | None:
        """Get a specific criterion by its ID.

        Args:
            criterion_id: Criterion ID (e.g., 'A', 'B', 'C')

        Returns:
            Criterion dictionary or None if not found
        """
        criteria = self.load_criteria()
        for criterion in criteria:
            if criterion["id"] == criterion_id:
                return criterion
        return None

    def get_all_criterion_ids(self) -> list[str]:
        """Get list of all criterion IDs.

        Returns:
            List of criterion IDs
        """
        criteria = self.load_criteria()
        return [c["id"] for c in criteria]
