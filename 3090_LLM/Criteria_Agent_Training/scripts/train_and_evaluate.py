#!/usr/bin/env python3
"""
Main script to train and evaluate SpanBERT model for DSM-5 criteria classification
"""

# Standard library imports
import argparse
import os
import sys

# Third-party imports
import torch

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.spanbert_classifier import DSMClassificationTrainer

def check_gpu():
    """Check GPU availability and print system info"""
    print("=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    print("=" * 40)

def main():
    parser = argparse.ArgumentParser(description='Train SpanBERT for DSM-5 criteria classification')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--model_name', type=str, default='SpanBERT/spanbert-base-cased',
                       help='Model name from HuggingFace')
    parser.add_argument('--output_dir', type=str, default='./spanbert_dsm_model',
                       help='Output directory for saved model')
    parser.add_argument('--test_only', action='store_true', help='Only run evaluation on existing model')

    args = parser.parse_args()

    # Check system information
    check_gpu()

    # Verify data files exist
    data_files = [
        'Data/translated_posts.csv',
        'Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json',
        'Data/Groundtruth/criteria_evaluation.csv'
    ]

    for file_path in data_files:
        if not os.path.exists(file_path):
            print(f"ERROR: Required data file not found: {file_path}")
            sys.exit(1)

    print("✓ All required data files found")

    # Optimize batch size for RTX 3090 (24GB VRAM)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        if 'rtx 3090' in gpu_name or 'rtx 4090' in gpu_name:
            # Increase batch size for high-memory GPUs
            if args.batch_size == 16:  # Only if using default
                args.batch_size = 24
                print(f"Optimized batch size for {gpu_name}: {args.batch_size}")

    # Initialize trainer
    print(f"Initializing trainer with model: {args.model_name}")
    trainer = DSMClassificationTrainer(
        model_name=args.model_name,
        max_length=args.max_length
    )

    if not args.test_only:
        print("Loading and preparing data...")
        df = trainer.load_and_prepare_data(
            'Data/translated_posts.csv',
            'Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json',
            'Data/Groundtruth/criteria_evaluation.csv'
        )

        print("Creating train/val/test datasets...")
        train_dataset, val_dataset, test_dataset = trainer.create_datasets(df)

        print(f"Starting training with parameters:")
        print(f"  - Epochs: {args.epochs}")
        print(f"  - Batch size: {args.batch_size}")
        print(f"  - Learning rate: {args.learning_rate}")
        print(f"  - Max length: {args.max_length}")

        # Train model
        trained_model = trainer.train(
            train_dataset,
            val_dataset,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )

        print("Training completed! Evaluating on test set...")
        results = trainer.evaluate(test_dataset, args.output_dir)

    else:
        print("Running evaluation only...")
        if not os.path.exists(args.output_dir):
            print(f"ERROR: Model directory not found: {args.output_dir}")
            sys.exit(1)

        df = trainer.load_and_prepare_data(
            'Data/translated_posts.csv',
            'Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json',
            'Data/Groundtruth/criteria_evaluation.csv'
        )
        _, _, test_dataset = trainer.create_datasets(df)
        results = trainer.evaluate(test_dataset, args.output_dir)

    print(f"\n🎉 Process completed successfully!")
    print(f"Final Accuracy: {results['accuracy']:.4f}")
    print(f"Final F1 Score: {results['f1_score']:.4f}")

if __name__ == "__main__":
    main()