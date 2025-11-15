#!/usr/bin/env python3
"""
Test the simplified training pipeline
"""

import torch
import sys
from spanbert_simple import DSMClassificationTrainer

def test_simple_training():
    """Test the simplified training pipeline"""

    print("=== Testing Simple SpanBERT Training Pipeline ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Initialize trainer
    trainer = DSMClassificationTrainer(max_length=256)

    try:
        # Load and prepare data
        print("\nLoading data...")
        df = trainer.load_and_prepare_data(
            'Data/translated_posts.csv',
            'Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json',
            'Data/Groundtruth/criteria_evaluation.csv'
        )

        # Use only first 50 examples for quick testing
        df_small = df.head(50).copy()
        print(f"Using {len(df_small)} examples for testing")

        # Create datasets
        train_dataset, val_dataset, test_dataset = trainer.create_datasets(df_small)

        # Test training with minimal configuration
        print("\nTesting training (1 epoch)...")
        trainer.train(
            train_dataset,
            val_dataset,
            output_dir='./test_simple_model',
            num_epochs=1,
            batch_size=4,
            learning_rate=2e-5
        )

        # Test evaluation
        print("Testing evaluation...")
        results = trainer.evaluate(test_dataset, './test_simple_model')

        print(f"\n✅ Simple test completed successfully!")
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        print(f"Test F1 Score: {results['f1_score']:.4f}")

        return True

    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_training()
    sys.exit(0 if success else 1)