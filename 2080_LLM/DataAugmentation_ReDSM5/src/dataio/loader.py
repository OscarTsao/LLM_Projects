"""
REDSM5 data loader with support for train/val/test splits.

This module provides utilities for loading the REDSM5 dataset from various sources
and preparing it for augmentation and training.
"""

from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import logging
import json

# Setup module logger
logger = logging.getLogger(__name__)


class REDSM5Loader:
    """
    Loader for REDSM5 dataset.
    
    Supports loading from:
    - HuggingFace datasets hub
    - Local Parquet files
    - Local CSV files
    
    Attributes:
        base_path: Path to the base data directory
        text_field: Name of the text field (default: "evidence_sentence")
        label_fields: Names of label fields
    """
    
    def __init__(
        self,
        base_path: Union[str, Path] = "data/redsm5/base",
        text_field: str = "evidence_sentence",
        label_fields: Optional[List[str]] = None,
        source_posts_path: Union[str, Path] = "Data/ReDSM5/redsm5_posts.csv",
        source_annotations_path: Union[str, Path] = "Data/ReDSM5/redsm5_annotations.csv",
        source_ground_truth_path: Union[str, Path] = "Data/GroundTruth/Final_Ground_Truth.json",
        random_seed: int = 42,
    ):
        """
        Initialize REDSM5Loader.

        Args:
            base_path: Path to base data directory
            text_field: Name of text field to augment
            label_fields: Names of label fields (default: ["criteria_label", "evidence_label"])
            source_posts_path: Path to source posts CSV file
            source_annotations_path: Path to source annotations CSV file
            source_ground_truth_path: Path to source ground truth JSON file
            random_seed: Random seed for reproducible splits
        """
        self.base_path = Path(base_path)
        self.text_field = text_field
        self.label_fields = label_fields or ["criteria_label", "evidence_label"]
        self.source_posts_path = Path(source_posts_path)
        self.source_annotations_path = Path(source_annotations_path)
        self.source_ground_truth_path = Path(source_ground_truth_path)
        self.random_seed = random_seed

    def load_local_redsm5(self) -> pd.DataFrame:
        """
        Load REDSM5 dataset from local CSV and JSON files.

        This method loads the posts, annotations, and ground truth files,
        then merges them into a unified dataset with evidence sentences and labels.

        Returns:
            DataFrame with columns: id, post_id, criterion, evidence_sentence,
            evidence_label, criteria_label, post_text

        Raises:
            FileNotFoundError: If source files are not found
            ValueError: If required columns are missing
        """
        logger.info("Loading REDSM5 dataset from local files...")

        # Check if files exist
        for path, name in [
            (self.source_posts_path, "posts"),
            (self.source_annotations_path, "annotations"),
            (self.source_ground_truth_path, "ground truth"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{name} file not found: {path}")

        # Load posts
        logger.info(f"Loading posts from {self.source_posts_path}")
        posts_df = pd.read_csv(self.source_posts_path)
        if "post_id" not in posts_df.columns or "text" not in posts_df.columns:
            raise ValueError("posts csv must contain post_id and text columns")
        posts_df = posts_df.set_index("post_id")

        # Load annotations
        logger.info(f"Loading annotations from {self.source_annotations_path}")
        annotations_df = pd.read_csv(self.source_annotations_path)
        expected_cols = {"post_id", "sentence_id", "sentence_text", "DSM5_symptom", "status"}
        missing = expected_cols - set(annotations_df.columns)
        if missing:
            raise ValueError(f"annotations csv missing columns: {missing}")

        # Fix typo in DSM5_symptom if it exists
        if "LEEP_ISSUES" in annotations_df["DSM5_symptom"].values:
            annotations_df["DSM5_symptom"] = annotations_df["DSM5_symptom"].replace(
                {"LEEP_ISSUES": "SLEEP_ISSUES"}
            )

        # Load ground truth
        logger.info(f"Loading ground truth from {self.source_ground_truth_path}")
        with open(self.source_ground_truth_path, "r") as f:
            ground_truth = json.load(f)

        if not isinstance(ground_truth, list):
            raise ValueError("ground truth json must be a list")

        # Convert ground truth to DataFrame with criteria labels
        gt_rows = []
        for example in ground_truth:
            post_id = str(example["post_id"])
            post_text = str(example["post"])
            criteria = example.get("criteria", {})

            for criterion, payload in criteria.items():
                label = int(payload.get("groundtruth", 0))
                row = {
                    "post_id": post_id,
                    "criterion": criterion,
                    "post_text": post_text,
                    "criteria_label": label,
                }
                gt_rows.append(row)

        gt_df = pd.DataFrame(gt_rows)
        logger.info(f"Loaded {len(gt_df)} criterion-level labels from ground truth")

        # Filter to positive evidence sentences (status=1)
        positive_evidence = annotations_df[annotations_df["status"] == 1].copy()
        logger.info(f"Found {len(positive_evidence)} positive evidence sentences")

        # Merge annotations with ground truth to get both evidence and criteria labels
        # For each post_id + criterion pair, get the first evidence sentence
        evidence_with_labels = []

        for _, gt_row in gt_df.iterrows():
            post_id = gt_row["post_id"]
            criterion = gt_row["criterion"]
            criteria_label = gt_row["criteria_label"]
            post_text = gt_row["post_text"]

            # Find matching evidence sentences for this post+criterion
            matching_evidence = positive_evidence[
                (positive_evidence["post_id"] == post_id) &
                (positive_evidence["DSM5_symptom"] == criterion)
            ]

            if len(matching_evidence) > 0:
                # Take the first evidence sentence
                evidence_row = matching_evidence.iloc[0]
                evidence_with_labels.append({
                    "id": f"{post_id}_{criterion}",
                    "post_id": post_id,
                    "criterion": criterion,
                    "evidence_sentence": evidence_row["sentence_text"],
                    "evidence_label": 1,  # Positive evidence
                    "criteria_label": criteria_label,
                    "post_text": post_text,
                })
            else:
                # No evidence sentence found, but we still have the criteria label
                # Use empty string or the full post as fallback
                evidence_with_labels.append({
                    "id": f"{post_id}_{criterion}",
                    "post_id": post_id,
                    "criterion": criterion,
                    "evidence_sentence": "",  # No evidence
                    "evidence_label": 0,  # No evidence
                    "criteria_label": criteria_label,
                    "post_text": post_text,
                })

        result_df = pd.DataFrame(evidence_with_labels)
        logger.info(f"Created dataset with {len(result_df)} examples")
        logger.info(f"  - Examples with evidence: {(result_df['evidence_label'] == 1).sum()}")
        logger.info(f"  - Examples without evidence: {(result_df['evidence_label'] == 0).sum()}")
        logger.info(f"  - Positive criteria labels: {(result_df['criteria_label'] == 1).sum()}")
        logger.info(f"  - Negative criteria labels: {(result_df['criteria_label'] == 0).sum()}")

        return result_df

    def create_train_val_test_splits(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        stratify_column: str = "criteria_label",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train/validation/test sets with stratification.

        Args:
            df: Input DataFrame
            train_size: Proportion for training set
            val_size: Proportion for validation set
            test_size: Proportion for test set
            stratify_column: Column to use for stratification

        Returns:
            Tuple of (train_df, val_df, test_df)

        Raises:
            ValueError: If split sizes don't sum to 1.0
        """
        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError(
                f"Split sizes must sum to 1.0, got {train_size + val_size + test_size}"
            )

        logger.info("Creating train/val/test splits...")
        logger.info(f"  Split ratios: train={train_size}, val={val_size}, test={test_size}")
        logger.info(f"  Stratifying by: {stratify_column}")

        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=df[stratify_column] if stratify_column in df.columns else None,
        )

        # Second split: separate train and validation
        val_ratio = val_size / (train_size + val_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=self.random_seed,
            stratify=train_val_df[stratify_column] if stratify_column in train_val_df.columns else None,
        )

        logger.info(f"Split sizes:")
        logger.info(f"  Train: {len(train_df)} examples ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"  Val:   {len(val_df)} examples ({len(val_df)/len(df)*100:.1f}%)")
        logger.info(f"  Test:  {len(test_df)} examples ({len(test_df)/len(df)*100:.1f}%)")

        return train_df, val_df, test_df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing or null values in the dataset.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values...")

        # Count missing values before
        missing_before = df.isnull().sum()
        if missing_before.sum() > 0:
            logger.warning(f"Found missing values:\n{missing_before[missing_before > 0]}")

        df = df.copy()

        # Handle missing evidence sentences
        if "evidence_sentence" in df.columns:
            df["evidence_sentence"] = df["evidence_sentence"].fillna("")

        # Handle missing labels (fill with 0)
        for label_field in self.label_fields:
            if label_field in df.columns:
                df[label_field] = df[label_field].fillna(0).astype(int)

        # Handle missing post_text
        if "post_text" in df.columns:
            df["post_text"] = df["post_text"].fillna("")

        # Handle missing criterion
        if "criterion" in df.columns:
            df["criterion"] = df["criterion"].fillna("UNKNOWN")

        # Count missing values after
        missing_after = df.isnull().sum()
        if missing_after.sum() > 0:
            logger.warning(f"Remaining missing values:\n{missing_after[missing_after > 0]}")
        else:
            logger.info("All missing values handled successfully")

        return df

    def prepare_base_dataset(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load local REDSM5 data and prepare train/val/test splits.

        This is the main method to prepare the base dataset from local files.

        Args:
            train_size: Proportion for training set
            val_size: Proportion for validation set
            test_size: Proportion for test set

        Returns:
            Dictionary mapping split names to DataFrames
        """
        # Load the data
        df = self.load_local_redsm5()

        # Handle missing values
        df = self.handle_missing_values(df)

        # Create splits
        train_df, val_df, test_df = self.create_train_val_test_splits(
            df, train_size, val_size, test_size
        )

        return {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }

    def load_from_hub(
        self, 
        dataset_name: str = "redsm5",
        splits: Optional[List[str]] = None,
    ) -> DatasetDict:
        """
        Load REDSM5 dataset from HuggingFace hub.
        
        Args:
            dataset_name: Name of dataset on hub
            splits: List of splits to load (default: ["train", "validation", "test"])
            
        Returns:
            DatasetDict with requested splits
        """
        splits = splits or ["train", "validation", "test"]
        
        # Load dataset
        dataset = load_dataset(dataset_name)
        
        # Filter to requested splits
        split_mapping = {
            "val": "validation",
            "dev": "validation",
        }
        
        filtered = {}
        for split in splits:
            mapped_split = split_mapping.get(split, split)
            if mapped_split in dataset:
                filtered[split] = dataset[mapped_split]
            else:
                raise ValueError(f"Split '{split}' not found in dataset")
                
        return DatasetDict(filtered)
    
    def load_from_parquet(
        self,
        splits: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load REDSM5 data from local Parquet files.
        
        Expects files named: {split}.parquet in base_path
        
        Args:
            splits: List of splits to load (default: ["train", "val", "test"])
            
        Returns:
            Dictionary mapping split names to DataFrames
        """
        splits = splits or ["train", "val", "test"]
        
        data = {}
        for split in splits:
            parquet_path = self.base_path / f"{split}.parquet"
            if not parquet_path.exists():
                raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
            
            data[split] = pd.read_parquet(parquet_path)
            
        return data
    
    def load_from_csv(
        self,
        splits: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load REDSM5 data from local CSV files.
        
        Expects files named: {split}.csv in base_path
        
        Args:
            splits: List of splits to load (default: ["train", "val", "test"])
            
        Returns:
            Dictionary mapping split names to DataFrames
        """
        splits = splits or ["train", "val", "test"]
        
        data = {}
        for split in splits:
            csv_path = self.base_path / f"{split}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            data[split] = pd.read_csv(csv_path)
            
        return data
    
    def save_to_parquet(
        self,
        data: Dict[str, Union[pd.DataFrame, Dataset]],
        compression: str = "zstd",
        compression_level: int = 3,
    ) -> None:
        """
        Save data splits to Parquet files.
        
        Args:
            data: Dictionary mapping split names to DataFrames or Datasets
            compression: Compression algorithm (default: "zstd")
            compression_level: Compression level (default: 3)
        """
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        for split, df in data.items():
            # Convert Dataset to DataFrame if needed
            if isinstance(df, Dataset):
                df = df.to_pandas()
            
            output_path = self.base_path / f"{split}.parquet"
            df.to_parquet(
                output_path,
                compression=compression,
                compression_level=compression_level,
                index=False,
            )
            print(f"Saved {split} split to {output_path} ({len(df)} examples)")
    
    def validate_data(self, data: Union[pd.DataFrame, Dataset]) -> bool:
        """
        Validate that data contains required fields.
        
        Args:
            data: DataFrame or Dataset to validate
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        if isinstance(data, Dataset):
            columns = data.column_names
        else:
            columns = data.columns.tolist()
        
        # Check text field
        if self.text_field not in columns:
            raise ValueError(f"Text field '{self.text_field}' not found in data")
        
        # Check label fields
        for label_field in self.label_fields:
            if label_field not in columns:
                raise ValueError(f"Label field '{label_field}' not found in data")
        
        return True
    
    def get_statistics(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Compute statistics for each split.
        
        Args:
            data: Dictionary mapping split names to DataFrames
            
        Returns:
            Dictionary of statistics per split
        """
        stats = {}
        
        for split, df in data.items():
            split_stats = {
                "num_examples": len(df),
                "text_field": self.text_field,
                "avg_text_length": df[self.text_field].str.len().mean(),
                "max_text_length": df[self.text_field].str.len().max(),
                "min_text_length": df[self.text_field].str.len().min(),
            }
            
            # Label distribution
            for label_field in self.label_fields:
                if label_field in df.columns:
                    split_stats[f"{label_field}_distribution"] = (
                        df[label_field].value_counts().to_dict()
                    )
            
            stats[split] = split_stats
        
        return stats
