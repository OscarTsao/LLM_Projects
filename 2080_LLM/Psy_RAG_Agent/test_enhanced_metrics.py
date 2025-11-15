#!/usr/bin/env python3
"""
Test script for enhanced metrics functionality
"""

import numpy as np
from basic_classifier import calculate_metrics

def test_metrics():
    """Test the enhanced metrics calculation"""
    print("Testing enhanced metrics...")

    # Create sample predictions and labels for testing
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0])

    print(f"True labels: {y_true}")
    print(f"Predictions: {y_pred}")

    # Calculate metrics
    result = calculate_metrics(y_true, y_pred)

    print("\n=== Metrics Results ===")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"Weighted Precision: {result['weighted_precision']:.4f}")
    print(f"Weighted Recall: {result['weighted_recall']:.4f}")
    print(f"Weighted F1: {result['weighted_f1']:.4f}")

    print(f"\nConfusion Matrix:")
    cm = result['confusion_matrix']
    print(f"              Predicted")
    print(f"              0    1")
    print(f"Actual 0   {cm[0, 0]:4d} {cm[0, 1]:4d}")
    print(f"Actual 1   {cm[1, 0]:4d} {cm[1, 1]:4d}")

    print("\nPer-class metrics:")
    for class_name, metrics in result['per_class_metrics'].items():
        print(f"{class_name}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

    print("\n✅ Metrics test completed successfully!")
    return True

if __name__ == "__main__":
    test_metrics()