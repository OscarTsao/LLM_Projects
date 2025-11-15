#!/usr/bin/env python3
"""
Quick test of the basic classifier
"""

# Standard library imports
import os
import sys

# Third-party imports
import torch

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.basic_classifier import BasicTrainer

def quick_test():
    print("=== Quick Test of Basic Classifier ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Initialize trainer
    trainer = BasicTrainer(max_features=1000)  # Small vocab for testing

    # Load data
    df = trainer.load_and_prepare_data(
        'Data/translated_posts.csv',
        'Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json',
        'Data/Groundtruth/criteria_evaluation.csv'
    )

    # Use small subset
    df_small = df.head(100)
    print(f"Using {len(df_small)} examples for testing")

    # Create datasets
    train_dataset, val_dataset, test_dataset = trainer.create_datasets(df_small)

    # Quick training (2 epochs)
    print("Starting quick training...")
    trainer.train(
        train_dataset,
        val_dataset,
        output_dir='./quick_test_model',
        num_epochs=3,
        batch_size=16,
        learning_rate=1e-3
    )

    # Evaluate
    results = trainer.evaluate(test_dataset, './quick_test_model')

    print(f"\n✅ Quick test completed!")
    print(f"Final Accuracy: {results['accuracy']:.4f}")
    print(f"Final F1 Score: {results['f1_score']:.4f}")

    return results

if __name__ == "__main__":
    results = quick_test()