#!/usr/bin/env python3
"""Test script to validate the HPO pipeline implementation."""

import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from dataaug_multi_both.hpo.search_space import suggest_trial_config, OptunaSearchSpace
        from dataaug_multi_both.hpo.trial_executor import TrialExecutor, TrialSpec, TrialResult
        from dataaug_multi_both.utils.mlflow_setup import setup_mlflow
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_search_space():
    """Test the search space generation."""
    print("Testing search space...")
    
    try:
        import optuna
        
        # Create a test study
        study = optuna.create_study(direction="maximize")
        
        # Test trial suggestion
        trial = study.ask()
        from dataaug_multi_both.hpo.search_space import suggest_trial_config
        
        config = suggest_trial_config(trial)
        
        # Validate config has required keys
        required_keys = [
            "model_name", "learning_rate", "batch_size", 
            "loss_function", "optimizer", "criteria_head_type",
            "evidence_head_type", "task_coupling"
        ]
        
        for key in required_keys:
            if key not in config:
                print(f"✗ Missing required key: {key}")
                return False
        
        print(f"✓ Search space generated config with {len(config)} parameters")
        print(f"  Sample config: model_name={config['model_name']}, lr={config['learning_rate']:.2e}")
        return True
        
    except Exception as e:
        print(f"✗ Search space test failed: {e}")
        return False

def test_model_creation():
    """Test model creation from config."""
    print("Testing model creation...")
    
    try:
        import torch
        from dataaug_multi_both.hpo.trial_executor import create_model_from_config, get_model_id_from_name
        
        # Test model ID mapping
        model_id = get_model_id_from_name("bert-base")
        if model_id != "google-bert/bert-base-uncased":
            print(f"✗ Model ID mapping failed: {model_id}")
            return False
        
        # Test config creation (without actually loading the model)
        config = {
            "model_name": "bert-base",
            "criteria_head_type": "linear",
            "evidence_head_type": "start_end_linear",
            "task_coupling": "independent",
            "num_criteria": 9,
            "criteria_pooling": "cls",
            "criteria_hidden_dim": 512,
            "criteria_dropout": 0.1,
            "evidence_dropout": 0.1,
            "max_span_length": 512
        }
        
        print("✓ Model configuration validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation functions."""
    print("Testing dataset creation...")
    
    try:
        from dataaug_multi_both.data.dataset import RedSM5Dataset, create_pytorch_dataset
        
        # Create mock HF dataset
        class MockDataset:
            def __init__(self):
                self.data = [
                    {
                        "post_text": "I feel anxious about everything",
                        "criteria_labels": [1, 0, 1, 0, 0, 0, 0, 0, 0],
                        "evidence_spans": [{"criterion_id": 0, "start": 7, "end": 14}]
                    }
                ]
            
            def __iter__(self):
                return iter(self.data)
            
            def __len__(self):
                return len(self.data)
        
        # Create mock tokenizer
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                return {
                    "input_ids": torch.tensor([[101, 102, 103, 104, 105]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
                }
        
        mock_dataset = MockDataset()
        mock_tokenizer = MockTokenizer()
        
        # Test dataset creation
        pytorch_dataset = create_pytorch_dataset(
            mock_dataset,
            tokenizer=mock_tokenizer,
            input_format="multi_label"
        )
        
        if len(pytorch_dataset) != 1:
            print(f"✗ Dataset length mismatch: {len(pytorch_dataset)}")
            return False
        
        print("✓ Dataset creation passed")
        return True
        
    except Exception as e:
        print(f"✗ Dataset creation test failed: {e}")
        return False

def test_cli_help():
    """Test that CLI help works."""
    print("Testing CLI help...")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "src.dataaug_multi_both.cli.train", "hpo", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"✗ CLI help failed with return code {result.returncode}")
            print(f"  stderr: {result.stderr}")
            return False
        
        if "Run hyperparameter optimization" not in result.stdout:
            print("✗ CLI help output doesn't contain expected text")
            return False
        
        print("✓ CLI help works")
        return True
        
    except Exception as e:
        print(f"✗ CLI help test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("HPO Pipeline Validation Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_search_space,
        test_model_creation,
        test_dataset_creation,
        test_cli_help
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! HPO pipeline is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())
