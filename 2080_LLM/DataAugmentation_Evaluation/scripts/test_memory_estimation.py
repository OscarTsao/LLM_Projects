#!/usr/bin/env python3
"""Test script for memory estimation and configuration validation."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.memory_utils import (
    get_gpu_memory_info,
    estimate_model_memory,
    is_configuration_memory_safe,
    log_memory_usage
)


def test_configurations():
    """Test various model configurations for memory safety."""
    print("=== GPU Memory Information ===")
    memory_info = get_gpu_memory_info()
    for key, value in memory_info.items():
        print(f"{key}: {value:.2f}")
    
    print("\n=== Testing Model Configurations ===")
    
    # Test configurations that caused OOM
    test_configs = [
        # The problematic configuration from the error
        {
            "name": "DeBERTa OOM Config",
            "model": "deberta_base",
            "batch_size": 128,
            "max_seq_length": 384,
            "classifier_layers": 3,
            "gradient_accumulation": 8
        },
        # Safe configurations
        {
            "name": "DeBERTa Safe Config",
            "model": "deberta_base",
            "batch_size": 16,
            "max_seq_length": 256,
            "classifier_layers": 1,
            "gradient_accumulation": 2
        },
        {
            "name": "BERT Large Config",
            "model": "bert_base",
            "batch_size": 64,
            "max_seq_length": 512,
            "classifier_layers": 2,
            "gradient_accumulation": 4
        },
        {
            "name": "RoBERTa Medium Config",
            "model": "roberta_base",
            "batch_size": 32,
            "max_seq_length": 384,
            "classifier_layers": 2,
            "gradient_accumulation": 2
        }
    ]
    
    for config in test_configs:
        print(f"\n--- {config['name']} ---")
        
        estimated_memory = estimate_model_memory(
            config["model"],
            config["batch_size"],
            config["max_seq_length"],
            config["classifier_layers"],
            config["gradient_accumulation"]
        )
        
        is_safe, est_mem, avail_mem = is_configuration_memory_safe(
            config["model"],
            config["batch_size"],
            config["max_seq_length"],
            config["classifier_layers"],
            config["gradient_accumulation"]
        )
        
        status = "✅ SAFE" if is_safe else "❌ UNSAFE"
        print(f"Model: {config['model']}")
        print(f"Batch Size: {config['batch_size']}")
        print(f"Sequence Length: {config['max_seq_length']}")
        print(f"Classifier Layers: {config['classifier_layers']}")
        print(f"Gradient Accumulation: {config['gradient_accumulation']}")
        print(f"Estimated Memory: {estimated_memory:.2f} GB")
        print(f"Available Memory: {avail_mem:.2f} GB")
        print(f"Status: {status}")


def suggest_safe_configs():
    """Suggest safe configurations for different models."""
    print("\n=== Suggested Safe Configurations ===")
    
    models = ["bert_base", "roberta_base", "deberta_base"]
    
    for model in models:
        print(f"\n--- {model.upper()} ---")
        
        # Find maximum safe batch size for different sequence lengths
        seq_lengths = [128, 256, 384, 512]
        
        for seq_len in seq_lengths:
            max_safe_batch = 1
            for batch_size in [8, 16, 32, 64, 128, 256]:
                is_safe, _, _ = is_configuration_memory_safe(
                    model, batch_size, seq_len, 1, 2  # 1 classifier layer, 2 grad accum
                )
                if is_safe:
                    max_safe_batch = batch_size
                else:
                    break
            
            print(f"Seq Length {seq_len}: Max safe batch size = {max_safe_batch}")


if __name__ == "__main__":
    test_configurations()
    suggest_safe_configs()
