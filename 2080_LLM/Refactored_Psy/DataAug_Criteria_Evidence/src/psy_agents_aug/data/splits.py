"""Data splitting utilities with reproducibility guarantees."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    """
    Handles train/val/test splitting with reproducibility.
    
    Supports:
    - Stratified splitting by label
    - Seed-based reproducibility
    - Split persistence to JSON
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize splitter.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets.
        
        Args:
            df: Input DataFrame
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            stratify_column: Column name for stratified splitting
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Validate ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
            )
        
        # First split: train vs (val + test)
        stratify_values = df[stratify_column] if stratify_column else None
        
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            random_state=self.random_seed,
            stratify=stratify_values,
        )
        
        # Second split: val vs test
        val_test_ratio = test_ratio / (val_ratio + test_ratio)
        stratify_temp = temp_df[stratify_column] if stratify_column else None
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=val_test_ratio,
            random_state=self.random_seed,
            stratify=stratify_temp,
        )
        
        return train_df, val_df, test_df
    
    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path,
    ):
        """
        Save splits to CSV files and create index JSON.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            output_dir: Output directory for splits
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSVs
        train_df.to_csv(output_dir / "train.csv", index=False)
        val_df.to_csv(output_dir / "val.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)
        
        # Save split metadata
        split_info = {
            "random_seed": self.random_seed,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "train_indices": train_df.index.tolist(),
            "val_indices": val_df.index.tolist(),
            "test_indices": test_df.index.tolist(),
        }
        
        with open(output_dir / "splits.json", "w") as f:
            json.dump(split_info, f, indent=2)
    
    def load_splits(
        self, input_dir: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load existing splits from directory.
        
        Args:
            input_dir: Directory containing split CSV files
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        input_dir = Path(input_dir)
        
        train_df = pd.read_csv(input_dir / "train.csv")
        val_df = pd.read_csv(input_dir / "val.csv")
        test_df = pd.read_csv(input_dir / "test.csv")
        
        # Validate split metadata if exists
        split_json = input_dir / "splits.json"
        if split_json.exists():
            with open(split_json, "r") as f:
                split_info = json.load(f)
            
            # Verify sizes match
            assert len(train_df) == split_info["train_size"]
            assert len(val_df) == split_info["val_size"]
            assert len(test_df) == split_info["test_size"]
        
        return train_df, val_df, test_df


def create_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_seed: int = 42,
    stratify_column: Optional[str] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create K-fold cross-validation splits.
    
    Args:
        df: Input DataFrame
        n_splits: Number of folds
        random_seed: Random seed for reproducibility
        stratify_column: Column name for stratified splitting
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    from sklearn.model_selection import StratifiedKFold, KFold
    
    if stratify_column:
        kfold = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_seed
        )
        splits = list(kfold.split(df, df[stratify_column]))
    else:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        splits = list(kfold.split(df))
    
    return splits
