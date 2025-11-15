#!/usr/bin/env python3
"""
Test training script with small subset of data
"""

import torch
import sys
import os
from spanbert_classifier import DSMClassificationTrainer

def test_training_pipeline():
    """Test the complete training pipeline with a small subset"""

    print("=== Testing SpanBERT Training Pipeline ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Initialize trainer
    trainer = DSMClassificationTrainer(max_length=256)  # Reduced for testing

    try:
        # Load and prepare data
        print("\nLoading data...")
        df = trainer.load_and_prepare_data(
            'Data/translated_posts.csv',
            'Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json',
            'Data/Groundtruth/criteria_evaluation.csv'
        )

        # Use only first 100 examples for testing
        df_small = df.head(100).copy()
        print(f"Using {len(df_small)} examples for testing")
        print(f"Label distribution: {df_small['label'].value_counts().to_dict()}")

        # Create datasets
        print("Creating datasets...")
        train_dataset, val_dataset, test_dataset = trainer.create_datasets(df_small)

        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        # Test training with minimal epochs and small batch size
        print("\nTesting training (1 epoch)...")
        trained_model = trainer.train(
            train_dataset,
            val_dataset,
            output_dir='./test_spanbert_model',
            num_epochs=1,  # Only 1 epoch for testing
            batch_size=4,   # Small batch size
            learning_rate=2e-5
        )

        # Test evaluation
        print("Testing evaluation...")
        results = trainer.evaluate(test_dataset, './test_spanbert_model')

        print(f"\n✅ Test completed successfully!")
        print(f"Final Test Accuracy: {results['accuracy']:.4f}")
        print(f"Final Test F1 Score: {results['f1_score']:.4f}")

        # Clean up test model
        import shutil
        if os.path.exists('./test_spanbert_model'):
            shutil.rmtree('./test_spanbert_model')
            print("Cleaned up test model directory")

        return True

    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_pipeline()
    sys.exit(0 if success else 1)