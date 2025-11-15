#!/usr/bin/env python3
"""
Final optimized training script for DSM-5 criteria classification
"""

# Standard library imports
import json
import os
import sys

# Third-party imports
import torch

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.basic_classifier import BasicTrainer

def main():
    """Run the complete training pipeline"""

    print("=== DSM-5 Criteria Classification - Final Training ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {memory_gb:.1f} GB")

        # Optimize for RTX 3090
        if 'RTX 3090' in gpu_name:
            batch_size = 128
            max_features = 8000
            num_epochs = 20
        else:
            batch_size = 64
            max_features = 5000
            num_epochs = 15
    else:
        batch_size = 32
        max_features = 3000
        num_epochs = 10

    print(f"\nOptimized training parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max features: {max_features}")
    print(f"  Epochs: {num_epochs}")

    # Initialize trainer
    trainer = BasicTrainer(max_features=max_features)

    # Load data
    print("\nLoading dataset...")
    df = trainer.load_and_prepare_data(
        'Data/translated_posts.csv',
        'Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json',
        'Data/Groundtruth/criteria_evaluation.csv'
    )

    print(f"\nDataset statistics:")
    print(f"Total examples: {len(df):,}")
    print(f"Positive rate: {df['label'].mean():.3f}")

    # Create datasets
    train_dataset, val_dataset, test_dataset = trainer.create_datasets(df)

    # Train model
    print(f"\nStarting training...")
    trainer.train(
        train_dataset,
        val_dataset,
        output_dir='./final_dsm_model',
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=5e-4
    )

    # Evaluate
    print("\nEvaluating model...")
    results = trainer.evaluate(test_dataset, './final_dsm_model')

    # Save summary
    summary = {
        'total_examples': len(df),
        'train_size': len(train_dataset),
        'test_accuracy': results['accuracy'],
        'test_f1_score': results['f1_score'],
        'gpu_used': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'training_params': {
            'batch_size': batch_size,
            'max_features': max_features,
            'num_epochs': num_epochs
        }
    }

    with open('./final_dsm_model/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n🎉 Training completed successfully!")
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"Final Test F1 Score: {results['f1_score']:.4f}")
    print(f"Model saved to: ./final_dsm_model/")

    return results

if __name__ == "__main__":
    results = main()