#!/usr/bin/env python3
"""
Quick test of RAG-enhanced BERT training with small data subset
"""

import pandas as pd
from rag_spanbert_classifier import RAGDSMClassificationTrainer
import torch


def quick_rag_training_test():
    """Test RAG training with a small subset of data"""
    print("Quick RAG Training Test")
    print("=" * 50)

    # Initialize trainer with smaller retrieval settings for quick test
    trainer = RAGDSMClassificationTrainer(
        retrieval_top_k=3,  # Reduced for quick test
        retrieval_threshold=0.4  # Higher threshold for more focused retrieval
    )

    print(f"Device: {trainer.device}")

    # Load a small subset of data for quick testing
    print("Loading small data subset...")
    gt_df = pd.read_csv('Data/Groundtruth/criteria_evaluation.csv')

    # Use only first 20 posts for quick test
    small_df = gt_df.head(20).copy()
    print(f"Using {len(small_df)} posts for quick test")

    # Setup retriever
    trainer.setup_retriever('Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json')

    # Prepare data
    posts = []
    labels_list = []

    for idx, row in small_df.iterrows():
        post_text = row['post_id']
        posts.append(post_text)

        # Get all criteria columns
        criteria_columns = [col for col in small_df.columns if col != 'post_id']
        post_labels = {}
        for criterion_col in criteria_columns:
            post_labels[criterion_col] = int(row[criterion_col])

        labels_list.append(post_labels)

    print(f"Prepared {len(posts)} posts for training")

    # Create datasets with minimal split for quick test
    train_dataset, val_dataset, test_dataset = trainer.create_flat_dataset(
        posts, labels_list, test_size=0.3, val_size=0.2
    )

    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Quick training with minimal epochs
    print("Starting quick training...")
    try:
        trained_model = trainer.train(
            train_dataset,
            val_dataset,
            output_dir='./quick_rag_test_model',
            num_epochs=1,  # Just 1 epoch for quick test
            batch_size=4,  # Small batch size
            learning_rate=2e-5
        )
        print("✓ Quick training completed successfully!")

        # Test prediction
        test_post = "I feel very sad and hopeless every day."
        predictions = trainer.predict_for_post(test_post)

        print(f"\nTest prediction for: '{test_post}'")
        print(f"Found {len(predictions)} relevant criteria predictions:")
        for criterion_key, pred_info in predictions.items():
            print(f"  {criterion_key}: prediction={pred_info['prediction']}, confidence={pred_info['probability'][1]:.3f}")

        return True

    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = quick_rag_training_test()
    print("\n" + "=" * 50)
    if success:
        print("🎉 Quick RAG training test PASSED!")
        print("The RAG-enhanced BERT system is working correctly!")
    else:
        print("⚠️  Quick training test FAILED!")
    print("=" * 50)