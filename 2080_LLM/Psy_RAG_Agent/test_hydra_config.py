#!/usr/bin/env python3
"""
Test script for Hydra configuration
"""

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def test_config(cfg: DictConfig) -> None:
    """Test Hydra configuration loading"""
    print("=== Testing Hydra Configuration ===")
    print("Configuration loaded successfully!")
    print("\nFull configuration:")
    print(OmegaConf.to_yaml(cfg))

    print(f"\nData paths:")
    print(f"  Posts: {cfg.data.translated_posts_path}")
    print(f"  Criteria: {cfg.data.criteria_path}")
    print(f"  Ground truth: {cfg.data.groundtruth_path}")

    print(f"\nModel parameters:")
    print(f"  Max features: {cfg.model.max_features}")
    print(f"  Hidden dim: {cfg.model.hidden_dim}")
    print(f"  Dropout rate: {cfg.model.dropout_rate}")

    print(f"\nTraining parameters:")
    print(f"  Epochs: {cfg.training.num_epochs}")
    print(f"  Batch size: {cfg.training.batch_size}")
    print(f"  Learning rate: {cfg.training.learning_rate}")

    print("\n✅ Hydra configuration test completed successfully!")

if __name__ == "__main__":
    test_config()