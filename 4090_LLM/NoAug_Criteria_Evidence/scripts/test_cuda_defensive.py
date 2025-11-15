#!/usr/bin/env python3
"""
Test script to validate defensive CUDA error handling and label validation.

This script tests:
1. Full dataset iteration without errors
2. Label validation in training loop
3. CUDA error handling (if GPU available)
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from Project.Criteria.data.dataset import CriteriaDataset
from Project.Criteria.models.model import Model as CriteriaModel


def test_dataset_iteration():
    """Test that we can iterate through the entire dataset without errors."""
    print("=" * 80)
    print("TEST 1: Full Dataset Iteration")
    print("=" * 80)

    csv_path = Path(__file__).parent.parent / "data" / "processed" / "redsm5_matched_criteria.csv"

    if not csv_path.exists():
        print(f"❌ Dataset not found at {csv_path}")
        return False

    try:
        dataset = CriteriaDataset(csv_path=csv_path)
        print(f"✓ Loaded dataset with {len(dataset)} examples")

        # Check all labels
        all_labels = []
        for i in range(len(dataset)):
            item = dataset[i]
            label = item["labels"].item()
            all_labels.append(label)

            if label < 0 or label > 1:
                print(f"❌ Invalid label at index {i}: {label}")
                return False

        unique_labels = sorted(set(all_labels))
        print(f"✓ All {len(all_labels)} labels are valid")
        print(f"  Unique labels: {unique_labels}")
        print(f"  Label 0 count: {all_labels.count(0)}")
        print(f"  Label 1 count: {all_labels.count(1)}")
        print()
        return True

    except Exception as e:
        print(f"❌ Error during dataset iteration: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_iteration():
    """Test DataLoader batch iteration."""
    print("=" * 80)
    print("TEST 2: DataLoader Batch Iteration")
    print("=" * 80)

    csv_path = Path(__file__).parent.parent / "data" / "processed" / "redsm5_matched_criteria.csv"

    try:
        dataset = CriteriaDataset(csv_path=csv_path)
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

        print(f"✓ Created DataLoader with batch_size=16")

        batch_count = 0
        total_samples = 0

        for batch in loader:
            batch_count += 1
            labels = batch["labels"]

            # Validate labels
            if labels.min() < 0 or labels.max() > 1:
                print(f"❌ Invalid labels in batch {batch_count}")
                print(f"  Min: {labels.min().item()}, Max: {labels.max().item()}")
                print(f"  Labels: {labels.tolist()}")
                return False

            total_samples += len(labels)

        print(f"✓ Processed {batch_count} batches ({total_samples} samples)")
        print()
        return True

    except Exception as e:
        print(f"❌ Error during batch iteration: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward():
    """Test model forward pass with validation."""
    print("=" * 80)
    print("TEST 3: Model Forward Pass")
    print("=" * 80)

    csv_path = Path(__file__).parent.parent / "data" / "processed" / "redsm5_matched_criteria.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Create model
        model = CriteriaModel(
            model_name="bert-base-uncased",
            num_labels=2,
            classifier_dropout=0.1,
        )
        model = model.to(device)
        print(f"✓ Created model")

        # Create dataset and loader
        dataset = CriteriaDataset(csv_path=csv_path)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        # Test first batch
        batch = next(iter(loader))
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        print(f"✓ Loaded batch")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels.cpu().tolist()}")

        # Validate labels before forward pass
        num_labels = 2
        if labels.min() < 0 or labels.max() >= num_labels:
            print(f"❌ Invalid labels detected!")
            print(f"  Expected range [0, {num_labels-1}], got [{labels.min().item()}, {labels.max().item()}]")
            return False

        print(f"✓ Labels are valid (range: [{labels.min().item()}, {labels.max().item()}])")

        # Forward pass
        model.eval()
        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        print(f"✓ Forward pass successful")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Expected shape: ({labels.size(0)}, {num_labels})")

        # Validate output shape
        expected_shape = (labels.size(0), num_labels)
        if logits.shape != expected_shape:
            print(f"❌ Output shape mismatch!")
            return False

        print(f"✓ Output shape is correct")

        # Test loss calculation
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        print(f"✓ Loss calculation successful: {loss.item():.4f}")
        print()

        # Cleanup
        del model, logits, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ Error during model forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("CUDA DEFENSIVE VALIDATION TEST SUITE")
    print("=" * 80 + "\n")

    results = []

    # Test 1: Dataset iteration
    results.append(("Dataset Iteration", test_dataset_iteration()))

    # Test 2: Batch iteration
    results.append(("Batch Iteration", test_batch_iteration()))

    # Test 3: Model forward pass
    results.append(("Model Forward Pass", test_model_forward()))

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
        if not passed:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
