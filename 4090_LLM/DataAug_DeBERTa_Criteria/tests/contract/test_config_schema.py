"""Contract tests for configuration schema validation.

These tests verify that configuration files conform to expected schemas
and that invalid configurations are properly rejected.
"""

import pytest
from pathlib import Path
import yaml


class TestConfigSchema:
    """Test suite for configuration schema validation."""
    
    def test_train_config_exists(self):
        """Test that main training config exists."""
        config_path = Path("configs/train.yaml")
        assert config_path.exists(), f"Training config not found: {config_path}"
    
    def test_train_config_valid_yaml(self):
        """Test that training config is valid YAML."""
        config_path = Path("configs/train.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert config is not None
        assert isinstance(config, dict)
    
    def test_train_config_has_required_fields(self):
        """Test that training config has required fields."""
        config_path = Path("configs/train.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Required top-level fields
        assert "trainer" in config
        assert "seed" in config
        assert "mlflow" in config
        
        # Trainer fields
        assert "max_epochs" in config["trainer"]
        assert "checkpoint_interval" in config["trainer"]
        assert "optimization_metric" in config["trainer"]
        
        # MLflow fields
        assert "experiment_name" in config["mlflow"]
        assert "tracking_uri" in config["mlflow"]
    
    def test_model_config_exists(self):
        """Test that model config exists."""
        config_path = Path("configs/model/mental_bert.yaml")
        assert config_path.exists(), f"Model config not found: {config_path}"
    
    def test_model_config_valid_yaml(self):
        """Test that model config is valid YAML."""
        config_path = Path("configs/model/mental_bert.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert config is not None
        assert isinstance(config, dict)
    
    def test_model_config_has_required_fields(self):
        """Test that model config has required fields."""
        config_path = Path("configs/model/mental_bert.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert "name" in config
        assert "type" in config
        assert "encoder" in config
        assert "heads" in config
        assert "learning_rate" in config
    
    def test_data_config_exists(self):
        """Test that data config exists."""
        config_path = Path("configs/data/redsm5.yaml")
        assert config_path.exists(), f"Data config not found: {config_path}"
    
    def test_data_config_valid_yaml(self):
        """Test that data config is valid YAML."""
        config_path = Path("configs/data/redsm5.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert config is not None
        assert isinstance(config, dict)
    
    def test_data_config_has_required_fields(self):
        """Test that data config has required fields."""
        config_path = Path("configs/data/redsm5.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert "dataset_id" in config
        assert "splits" in config
        assert "batch_size" in config
        assert "max_length" in config
        
        # Verify dataset ID is correct
        assert config["dataset_id"] == "irlab-udc/redsm5"
        
        # Verify splits
        assert "train" in config["splits"]
        assert "validation" in config["splits"]
        assert "test" in config["splits"]
    
    def test_retention_policy_config_exists(self):
        """Test that retention policy config exists."""
        config_path = Path("configs/retention_policy/default.yaml")
        assert config_path.exists(), f"Retention policy config not found: {config_path}"
    
    def test_retention_policy_config_valid_yaml(self):
        """Test that retention policy config is valid YAML."""
        config_path = Path("configs/retention_policy/default.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert config is not None
        assert isinstance(config, dict)
    
    def test_retention_policy_config_has_required_fields(self):
        """Test that retention policy config has required fields."""
        config_path = Path("configs/retention_policy/default.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert "keep_last_n" in config
        assert "keep_best_k" in config
        assert "keep_best_k_max" in config
        assert "max_total_size" in config
        assert "disk_threshold_percent" in config
        
        # Verify values are reasonable
        assert config["keep_last_n"] >= 0
        assert config["keep_best_k"] >= 0
        assert config["keep_best_k_max"] >= config["keep_best_k"]
        assert config["disk_threshold_percent"] > 0
        assert config["disk_threshold_percent"] <= 100
    
    def test_retention_policy_defaults_match_spec(self):
        """Test that retention policy defaults match specification."""
        config_path = Path("configs/retention_policy/default.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # From spec: default policy is keep_last_n=1, keep_best_k=1, keep_best_k_max=2
        assert config["keep_last_n"] == 1
        assert config["keep_best_k"] == 1
        assert config["keep_best_k_max"] == 2
        
        # From spec: disk threshold is 10%
        assert config["disk_threshold_percent"] == 10
        
        # From spec: max_total_size is 10GB
        assert config["max_total_size"] == 10737418240  # 10GB in bytes

