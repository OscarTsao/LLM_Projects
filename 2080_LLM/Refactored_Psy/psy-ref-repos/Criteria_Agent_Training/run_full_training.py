#!/usr/bin/env python3
"""
Full training script optimized for RTX 3090
"""

import torch
import os
import json
from basic_classifier import BasicTrainer

def run_full_training():
    """Run full training on the complete dataset"""

    print("=== DSM-5 Criteria Classification - Full Training ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {memory_gb:.1f} GB")

        # Optimize parameters for RTX 3090
        if 'RTX 3090' in gpu_name:
            batch_size = 128  # Larger batch size for 3090
            max_features = 8000  # More features for better performance
            hidden_dim = 512
        else:
            batch_size = 64
            max_features = 5000
            hidden_dim = 256
    else:
        print("GPU not available, using CPU with reduced parameters")
        batch_size = 32
        max_features = 3000
        hidden_dim = 128

    print(f"Training parameters:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Max features: {max_features}")
    print(f"  - Hidden dim: {hidden_dim}")

    # Initialize trainer
    trainer = BasicTrainer(max_features=max_features)

    # Load and prepare data
    print("\nLoading full dataset...")
    df = trainer.load_and_prepare_data(
        'Data/translated_posts.csv',
        'Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json',
        'Data/Groundtruth/criteria_evaluation.csv'
    )

    print(f"\nDataset statistics:")
    print(f"Total examples: {len(df)}")
    print(f"Positive examples: {sum(df['label'])}")
    print(f"Negative examples: {len(df) - sum(df['label'])}")
    print(f"Balance ratio: {sum(df['label']) / len(df):.3f}")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset, val_dataset, test_dataset = trainer.create_datasets(df)

    # Modify model for better performance
    class OptimizedBasicClassifier(trainer.model.__class__ if trainer.model else torch.nn.Module):
        def __init__(self, input_dim, hidden_dim=hidden_dim, dropout_rate=0.3, num_labels=2):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(hidden_dim, hidden_dim // 2),
                torch.nn.BatchNorm1d(hidden_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(hidden_dim // 2, hidden_dim // 4),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate // 2),
                torch.nn.Linear(hidden_dim // 4, num_labels)
            )

        def forward(self, features):
            return self.network(features)

    # Replace the model class
    from basic_classifier import BasicClassifier
    BasicClassifier.__init__ = OptimizedBasicClassifier.__init__
    BasicClassifier.forward = OptimizedBasicClassifier.forward

    # Train model
    print(f"\nStarting full training...")
    trainer.train(
        train_dataset,
        val_dataset,
        output_dir='./dsm_criteria_model',
        num_epochs=20,  # More epochs for full training
        batch_size=batch_size,
        learning_rate=5e-4  # Slightly lower learning rate
    )

    # Evaluate model
    print("\nEvaluating final model...")
    results = trainer.evaluate(test_dataset, './dsm_criteria_model')

    # Save training summary
    summary = {
        'dataset_size': len(df),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'final_accuracy': results['accuracy'],
        'final_f1_score': results['f1_score'],
        'training_params': {
            'batch_size': batch_size,
            'max_features': max_features,
            'hidden_dim': hidden_dim,
            'num_epochs': 20,
            'learning_rate': 5e-4
        },
        'gpu_used': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    }

    with open('./dsm_criteria_model/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n🎉 Full training completed successfully!")
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"Final Test F1 Score: {results['f1_score']:.4f}")
    print(f"Model saved to: ./dsm_criteria_model/")

    return trainer, results

if __name__ == "__main__":
    trainer, results = run_full_training()