#!/usr/bin/env python3
"""Test script for data loading and preprocessing."""

import logging
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from Project.SubProject.data.dataset import (
    create_folds,
    load_dsm5_criteria,
    load_redsm5_data,
    ReDSM5Dataset,
    stratified_negative_sampling,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run data loading tests."""
    logger.info("Starting data loading tests...")
    
    logger.info("\nTest 1: Loading DSM-5 criteria...")
    dsm5_dir = "data/DSM5"
    criteria_dict = load_dsm5_criteria(dsm5_dir)
    logger.info(f"Loaded {len(criteria_dict)} criteria")
    assert len(criteria_dict) > 0, "No criteria loaded!"
    
    logger.info("\nTest 2: Loading ReDSM5 dataset...")
    csv_path = "data/redsm5/posts.csv"
    df = load_redsm5_data(csv_path, criteria_dict)
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Class distribution:\n{df['label'].value_counts()}")
    assert len(df) > 0, "No data loaded!"
    
    logger.info("\nTest 3: Testing stratified negative sampling...")
    df_balanced = stratified_negative_sampling(df, pos_neg_ratio=0.333, seed=42)
    logger.info(f"Balanced dataset size: {len(df_balanced)}")
    logger.info(f"Class distribution:\n{df_balanced['label'].value_counts()}")
    
    logger.info("\nTest 4: Creating cross-validation folds...")
    folds = create_folds(df_balanced, n_splits=5, seed=42)
    logger.info(f"Created {len(folds)} folds")
    for i, (train_idx, val_idx) in enumerate(folds, 1):
        logger.info(f"Fold {i}: train={len(train_idx)}, val={len(val_idx)}")
    
    logger.info("\nTest 5: Testing dataset creation...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    train_idx, val_idx = folds[0]
    train_df = df_balanced.iloc[train_idx]
    
    dataset = ReDSM5Dataset(train_df, tokenizer, max_length=512)
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    sample = dataset[0]
    logger.info(f"Sample keys: {sample.keys()}")
    logger.info(f"Input IDs shape: {sample['input_ids'].shape}")
    logger.info(f"Attention mask shape: {sample['attention_mask'].shape}")
    logger.info(f"Label: {sample['labels'].item()}")
    
    logger.info("\n" + "=" * 60)
    logger.info("All tests passed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
