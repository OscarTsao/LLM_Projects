#!/usr/bin/env python3
"""
Main training script for DSM-5 Criteria Classification
"""

import argparse
import torch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from basic_classifier import BasicTrainer
from spanbert_classifier import DSMClassificationTrainer
from rag_spanbert_classifier import RAGDSMClassificationTrainer


def main():
    parser = argparse.ArgumentParser(description='Train DSM-5 criteria classifier')
    parser.add_argument('--model', choices=['basic', 'spanbert', 'rag'], default='basic',
                       help='Model type to train (default: basic)')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (auto-detected if not set)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output_dir', default='./models/trained', help='Output directory')
    parser.add_argument('--max_features', type=int, default=None, help='Max features for TF-IDF (basic model only)')

    args = parser.parse_args()

    print("=== DSM-5 Criteria Classification Training ===")
    print(f"Model: {args.model}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {memory_gb:.1f} GB")

        # Auto-optimize batch size if not specified
        if args.batch_size is None:
            if 'RTX 3090' in gpu_name:
                args.batch_size = 128 if args.model == 'basic' else 16
            else:
                args.batch_size = 64 if args.model == 'basic' else 8
    else:
        args.batch_size = args.batch_size or 32

    # Initialize trainer based on model type
    if args.model == 'basic':
        max_features = args.max_features or (8000 if torch.cuda.is_available() else 5000)
        trainer = BasicTrainer(max_features=max_features)
    elif args.model == 'spanbert':
        trainer = DSMClassificationTrainer()
    elif args.model == 'rag':
        trainer = RAGDSMClassificationTrainer()

    # Load data
    print("\nLoading and preparing data...")
    df = trainer.load_and_prepare_data(
        'Data/translated_posts.csv',
        'Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json',
        'Data/Groundtruth/criteria_evaluation.csv'
    )

    print(f"Loaded {len(df)} examples")

    # Create datasets
    train_dataset, val_dataset, test_dataset = trainer.create_datasets(df)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Train model
    print(f"\nTraining {args.model} model...")
    trainer.train(
        train_dataset,
        val_dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    # Evaluate
    print("Evaluating model...")
    results = trainer.evaluate(test_dataset, args.output_dir)

    print(f"\n✅ Training completed!")
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"Final Test F1 Score: {results['f1_score']:.4f}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()