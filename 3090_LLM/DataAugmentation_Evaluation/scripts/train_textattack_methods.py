#!/usr/bin/env python
"""Train models with different TextAttack augmentation methods.

This script generates augmented datasets using different TextAttack augmentation
techniques and trains models using the best config for each method.

Available TextAttack methods:
- EmbeddingAugmenter: Word replacement using word embeddings (default in pipeline)
- WordNetAugmenter: Synonym replacement using WordNet
- EasyDataAugmenter (EDA): Combination of synonym replacement, random insertion, swap, deletion
- CharSwapAugmenter: Character-level perturbations
- CheckListAugmenter: Contrast sets with systematic perturbations
- CLAREAugmenter: Contextualized augmentation using masked language model
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Available TextAttack augmentation methods
TEXTATTACK_METHODS = {
    "embedding": {
        "name": "EmbeddingAugmenter",
        "description": "Word embedding-based replacement",
        "config": {"transformations_per_example": 3},
    },
    "wordnet": {
        "name": "WordNetAugmenter",
        "description": "WordNet synonym replacement",
        "config": {"pct_words_to_swap": 0.1},
    },
    "eda": {
        "name": "EasyDataAugmenter",
        "description": "EDA: synonym replacement, insertion, swap, deletion",
        "config": {"pct_words_to_swap": 0.1, "transformations_per_example": 4},
    },
    "charswap": {
        "name": "CharSwapAugmenter",
        "description": "Character-level perturbations",
        "config": {"pct_words_to_swap": 0.1, "transformations_per_example": 3},
    },
    "checklist": {
        "name": "CheckListAugmenter",
        "description": "Systematic contrast sets",
        "config": {"transformations_per_example": 3},
    },
    "clare": {
        "name": "CLAREAugmenter",
        "description": "Contextualized MLM-based augmentation",
        "config": {"pct_words_to_swap": 0.1},
    },
}

# Base encoders to test
ENCODERS = {
    "deberta": "microsoft/deberta-base",
    "roberta": "FacebookAI/roberta-base",
    "bert": "google-bert/bert-base-uncased",
}


def generate_augmented_dataset(method: str, output_path: Path) -> bool:
    """Generate augmented dataset using specified TextAttack method.

    Args:
        method: TextAttack augmentation method key
        output_path: Path to save augmented dataset

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Generating augmented dataset: {method}")
    print(f"Method: {TEXTATTACK_METHODS[method]['name']}")
    print(f"Description: {TEXTATTACK_METHODS[method]['description']}")
    print(f"{'='*60}\n")

    try:
        from textattack.augmentation import Augmenter
        import textattack.augmentation as aug
        import pandas as pd

        # Get augmenter class
        augmenter_class = getattr(aug, TEXTATTACK_METHODS[method]["name"])
        config = TEXTATTACK_METHODS[method]["config"]
        augmenter = augmenter_class(**config)

        # Load original data
        posts_path = Path("Data/ReDSM5/redsm5_posts.csv")
        if not posts_path.exists():
            print(f"Error: Posts file not found at {posts_path}")
            return False

        posts_df = pd.read_csv(posts_path)

        # Generate augmented data (simplified - you may need to adapt to your data structure)
        augmented_data = []
        for idx, row in posts_df.iterrows():
            original_text = row['post_text']  # Adjust column name as needed

            # Generate augmentations
            try:
                augmented_texts = augmenter.augment(original_text)
                for aug_text in augmented_texts[:3]:  # Limit to 3 per example
                    augmented_data.append({
                        **row.to_dict(),
                        'post_text': aug_text,
                        'augmentation_method': method,
                        'original_post_id': row.get('post_id', idx),
                    })
            except Exception as e:
                print(f"Warning: Failed to augment example {idx}: {e}")
                continue

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(posts_df)} examples...")

        # Save augmented dataset
        output_path.parent.mkdir(parents=True, exist_ok=True)
        augmented_df = pd.DataFrame(augmented_data)
        augmented_df.to_csv(output_path, index=False)

        print(f"\n✓ Generated {len(augmented_data)} augmented examples")
        print(f"✓ Saved to: {output_path}")
        return True

    except ImportError as e:
        print(f"Error: TextAttack not installed or import failed: {e}")
        print("Install with: pip install textattack")
        return False
    except Exception as e:
        print(f"Error generating dataset: {e}")
        return False


def train_with_method(method: str, encoder: str = "deberta", use_best_config: bool = True) -> bool:
    """Train model with specific TextAttack method.

    Args:
        method: TextAttack augmentation method key
        encoder: Encoder model key (deberta, roberta, bert)
        use_best_config: Whether to use best_config as base

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Training with {method} augmentation + {encoder} encoder")
    print(f"{'='*60}\n")

    # Build training command
    if use_best_config:
        cmd = [
            "python", "-m", "src.training.train",
            "--config-name=best_config",
            f"model.pretrained_model_name={ENCODERS[encoder]}",
            f"dataset.name=textattack_{method}",
            f"notes=TextAttack {method} with {encoder}",
        ]
    else:
        cmd = [
            "python", "-m", "src.training.train",
            f"dataset=original_textattack",
            f"model.pretrained_model_name={ENCODERS[encoder]}",
            f"notes=TextAttack {method} with {encoder}",
        ]

    try:
        # Run training
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ Training completed successfully for {method} + {encoder}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed for {method} + {encoder}")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train models with different TextAttack augmentation methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(TEXTATTACK_METHODS.keys()) + ["all"],
        default=["all"],
        help="TextAttack methods to use (default: all)",
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        choices=list(ENCODERS.keys()) + ["all"],
        default=["deberta"],
        help="Encoder models to use (default: deberta)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip dataset generation (use existing datasets)",
    )
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="Only generate datasets, skip training",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="List available TextAttack methods and exit",
    )

    args = parser.parse_args()

    # List methods and exit
    if args.list_methods:
        print("\nAvailable TextAttack Augmentation Methods:")
        print("=" * 60)
        for key, info in TEXTATTACK_METHODS.items():
            print(f"\n{key}:")
            print(f"  Class: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Config: {info['config']}")
        print("\n" + "=" * 60)
        return 0

    # Expand "all" selections
    methods = list(TEXTATTACK_METHODS.keys()) if "all" in args.methods else args.methods
    encoders = list(ENCODERS.keys()) if "all" in args.encoders else args.encoders

    print("\n" + "=" * 60)
    print("TextAttack Augmentation Training Pipeline")
    print("=" * 60)
    print(f"Methods: {', '.join(methods)}")
    print(f"Encoders: {', '.join(encoders)}")
    print(f"Skip generation: {args.skip_generation}")
    print(f"Generation only: {args.generation_only}")
    print("=" * 60)

    # Generate datasets
    if not args.skip_generation:
        print("\n[Step 1] Generating augmented datasets...")
        for method in methods:
            output_path = Path(f"Data/Augmentation/textattack_{method}_dataset.csv")
            success = generate_augmented_dataset(method, output_path)
            if not success:
                print(f"Warning: Failed to generate dataset for {method}")
                print("Continuing with other methods...")

    if args.generation_only:
        print("\n✓ Dataset generation completed")
        return 0

    # Train models
    print("\n[Step 2] Training models...")
    results = {}
    total = len(methods) * len(encoders)
    completed = 0

    for method in methods:
        for encoder in encoders:
            completed += 1
            print(f"\n[{completed}/{total}] Training: {method} + {encoder}")
            success = train_with_method(method, encoder)
            results[f"{method}_{encoder}"] = success

    # Summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    successful = sum(1 for v in results.values() if v)
    print(f"Total runs: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")

    if successful < total:
        print("\nFailed runs:")
        for name, success in results.items():
            if not success:
                print(f"  - {name}")

    print("\n✓ All training runs completed")
    print("\nView results:")
    print("  - File system: ls -lt outputs/$(date +%Y-%m-%d)/")
    print("  - MLflow UI: make mlflow-ui (http://localhost:5000)")

    return 0 if successful == total else 1


if __name__ == "__main__":
    sys.exit(main())
