#!/usr/bin/env python3
"""
Hydra-enabled training script for DSM-5 criteria classification
"""

import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import json
from basic_classifier import BasicTrainer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration"""

    print("=== DSM-5 Criteria Classification with Hydra ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Print configuration
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Auto-optimize parameters based on GPU if enabled
    if cfg.gpu.auto_optimize and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {memory_gb:.1f} GB")

        if 'RTX 3090' in gpu_name:
            print("Detected RTX 3090 - using optimized settings")
            if hasattr(cfg.training, 'rtx_3090'):
                cfg.training.update(cfg.training.rtx_3090)
            if hasattr(cfg.model, 'rtx_3090'):
                cfg.model.update(cfg.model.rtx_3090)
        else:
            print("Using standard GPU settings")
            if hasattr(cfg.training, 'standard_gpu'):
                cfg.training.update(cfg.training.standard_gpu)
            if hasattr(cfg.model, 'standard_gpu'):
                cfg.model.update(cfg.model.standard_gpu)
    elif not torch.cuda.is_available():
        print("CUDA not available - using CPU settings")
        if hasattr(cfg.training, 'cpu'):
            cfg.training.update(cfg.training.cpu)
        if hasattr(cfg.model, 'cpu'):
            cfg.model.update(cfg.model.cpu)

    print(f"\nFinal training parameters:")
    print(f"  Batch size: {cfg.training.batch_size}")
    print(f"  Max features: {cfg.model.max_features}")
    print(f"  Epochs: {cfg.training.num_epochs}")
    print(f"  Learning rate: {cfg.training.learning_rate}")

    # Initialize trainer with Hydra config
    trainer = BasicTrainer(max_features=cfg.model.max_features)

    # Load data
    print("\nLoading dataset...")
    df = trainer.load_and_prepare_data(
        cfg.data.translated_posts_path,
        cfg.data.criteria_path,
        cfg.data.groundtruth_path
    )

    print(f"\nDataset statistics:")
    print(f"Total examples: {len(df):,}")
    print(f"Positive rate: {df['label'].mean():.3f}")

    # Create datasets
    train_dataset, val_dataset, test_dataset = trainer.create_datasets(
        df,
        test_size=cfg.training.test_size,
        val_size=cfg.training.val_size
    )

    # Train model
    print(f"\nStarting training...")
    trainer.train(
        train_dataset,
        val_dataset,
        output_dir=cfg.output.model_dir,
        num_epochs=cfg.training.num_epochs,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate
    )

    # Evaluate
    print("\nEvaluating model...")
    results = trainer.evaluate(test_dataset, cfg.output.model_dir)

    # Save detailed results and configuration
    detailed_results = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'dataset_stats': {
            'total_examples': len(df),
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
            'positive_rate': float(df['label'].mean())
        },
        'results': {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'weighted_precision': float(results['weighted_precision']),
            'weighted_recall': float(results['weighted_recall']),
            'f1_score': float(results['f1_score']),
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'per_class_metrics': {
                k: {
                    'precision': float(v['precision']),
                    'recall': float(v['recall']),
                    'f1': float(v['f1'])
                } for k, v in results['per_class_metrics'].items()
            }
        },
        'system_info': {
            'gpu_used': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
    }

    # Ensure output directory exists
    os.makedirs(cfg.output.model_dir, exist_ok=True)

    # Save results
    with open(os.path.join(cfg.output.model_dir, 'detailed_results.json'), 'w') as f:
        json.dump(detailed_results, f, indent=2)

    # Save the config that was actually used
    with open(os.path.join(cfg.output.model_dir, 'config_used.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)

    print(f"\n🎉 Training completed successfully!")
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"Final Test Precision: {results['precision']:.4f}")
    print(f"Final Test Recall: {results['recall']:.4f}")
    print(f"Final Test F1 Score: {results['f1_score']:.4f}")
    print(f"Model and results saved to: {cfg.output.model_dir}")

    return results


if __name__ == "__main__":
    main()