"""Unit tests for Optuna search space."""

import pytest
from unittest.mock import Mock, MagicMock
from src.dataaug_multi_both.hpo.search_space import (
    SearchSpaceConfig,
    OptunaSearchSpace,
    create_search_space,
    suggest_trial_config,
    OPTUNA_AVAILABLE
)


class TestSearchSpaceConfig:
    """Test suite for SearchSpaceConfig."""
    
    def test_config_defaults(self):
        """Test that config has correct defaults."""
        config = SearchSpaceConfig()
        assert config.model_name == "microsoft/deberta-v3-base"
        assert config.learning_rate == "log_uniform"
        assert config.batch_size == "categorical"


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestOptunaSearchSpace:
    """Test suite for OptunaSearchSpace."""
    
    def create_mock_trial(self):
        """Create a mock Optuna trial."""
        trial = Mock()
        
        # Mock suggest methods
        trial.suggest_categorical = Mock(side_effect=lambda name, choices: choices[0])
        trial.suggest_float = Mock(side_effect=lambda name, low, high, **kwargs: (low + high) / 2)
        trial.suggest_int = Mock(side_effect=lambda name, low, high: (low + high) // 2)
        
        return trial
    
    def test_search_space_initialization(self):
        """Test that search space can be initialized."""
        search_space = OptunaSearchSpace()
        assert search_space.config is not None
    
    def test_suggest_hyperparameters(self):
        """Test suggesting hyperparameters."""
        search_space = OptunaSearchSpace()
        trial = self.create_mock_trial()
        
        params = search_space.suggest_hyperparameters(trial)
        assert params["model_name"] == "microsoft/deberta-v3-base"
        assert "learning_rate" in params
        assert "batch_size" in params
        assert "loss_type" in params
        assert "optimizer" in params
    
    def test_suggest_hyperparameters_with_focal_loss(self):
        """Test that focal_gamma is suggested when loss_type=focal."""
        search_space = OptunaSearchSpace()
        trial = self.create_mock_trial()

        # Force loss_type to be focal
        trial.suggest_categorical = Mock(side_effect=lambda name, choices: "focal" if name == "loss_type" else choices[0])

        params = search_space.suggest_hyperparameters(trial)

        assert params["loss_type"] == "focal"
        assert "focal_gamma" in params
    
    def test_suggest_hyperparameters_without_focal_loss(self):
        """Test that focal_gamma is not suggested when loss_type!=focal."""
        search_space = OptunaSearchSpace()
        trial = self.create_mock_trial()

        # Force loss_type to be bce
        trial.suggest_categorical = Mock(side_effect=lambda name, choices: "bce" if name == "loss_type" else choices[0])

        params = search_space.suggest_hyperparameters(trial)

        assert params["loss_type"] == "bce"
        assert "focal_gamma" not in params

    def test_suggest_hyperparameters_adaptive_focal(self):
        """Test that adaptive focal variants do not request focal_gamma."""
        search_space = OptunaSearchSpace()
        trial = self.create_mock_trial()

        trial.suggest_categorical = Mock(
            side_effect=lambda name, choices: "adaptive_focal" if name == "loss_type" else choices[0]
        )

        params = search_space.suggest_hyperparameters(trial)

        assert params["loss_type"] == "adaptive_focal"
        assert "focal_gamma" not in params
    
    def test_validate_params_valid(self):
        """Test validating valid parameters."""
        search_space = OptunaSearchSpace()
        
        params = {
            "model_name": "microsoft/deberta-v3-base",
            "learning_rate": 1e-4,
            "batch_size": 16,
            "loss_type": "bce",
            "optimizer": "adam",
            "weight_decay": 1e-4
        }
        
        assert search_space.validate_params(params) is True
    
    def test_validate_params_missing_required(self):
        """Test that validation fails for missing required params."""
        search_space = OptunaSearchSpace()
        
        params = {
            "model_name": "microsoft/deberta-v3-base",
            # Missing learning_rate
            "batch_size": 16
        }
        
        assert search_space.validate_params(params) is False
    
    def test_validate_params_invalid_learning_rate(self):
        """Test that validation fails for invalid learning rate."""
        search_space = OptunaSearchSpace()
        
        params = {
            "model_name": "microsoft/deberta-v3-base",
            "learning_rate": 1.0,  # Too high
            "batch_size": 16,
            "loss_type": "bce",
            "optimizer": "adam"
        }
        
        assert search_space.validate_params(params) is False
    
    def test_validate_params_invalid_batch_size(self):
        """Test that validation fails for invalid batch size."""
        search_space = OptunaSearchSpace()
        
        params = {
            "model_name": "microsoft/deberta-v3-base",
            "learning_rate": 1e-4,
            "batch_size": 7,  # Not in valid choices
            "loss_type": "bce",
            "optimizer": "adam"
        }
        
        assert search_space.validate_params(params) is False
    
    def test_validate_params_focal_missing_gamma(self):
        """Test that validation fails when focal loss missing gamma."""
        search_space = OptunaSearchSpace()
        
        params = {
            "model_name": "microsoft/deberta-v3-base",
            "learning_rate": 1e-4,
            "batch_size": 16,
            "loss_type": "focal",
            "optimizer": "adam"
            # Missing focal_gamma
        }
        
        assert search_space.validate_params(params) is False
    
    def test_validate_params_focal_invalid_gamma(self):
        """Test that validation fails for invalid focal gamma."""
        search_space = OptunaSearchSpace()
        
        params = {
            "model_name": "microsoft/deberta-v3-base",
            "learning_rate": 1e-4,
            "batch_size": 16,
            "loss_type": "focal",
            "optimizer": "adam",
            "focal_gamma": 10.0  # Too high
        }
        
        assert search_space.validate_params(params) is False


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestCreateSearchSpace:
    """Test suite for factory function."""
    
    def test_create_search_space_defaults(self):
        """Test creating search space with defaults."""
        search_space = create_search_space()
        assert isinstance(search_space, OptunaSearchSpace)
        assert search_space.config is not None
    
    def test_create_search_space_custom_config(self):
        """Test creating search space with custom config."""
        config = SearchSpaceConfig()
        search_space = create_search_space(config)
        assert search_space.config == config


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestSuggestTrialConfig:
    """Test suite for suggest_trial_config function."""
    
    def create_mock_trial(self):
        """Create a mock Optuna trial."""
        trial = Mock()
        trial.suggest_categorical = Mock(side_effect=lambda name, choices: choices[0])
        trial.suggest_float = Mock(side_effect=lambda name, low, high, **kwargs: (low + high) / 2)
        return trial
    
    def test_suggest_trial_config(self):
        """Test suggesting trial config."""
        trial = self.create_mock_trial()
        config = suggest_trial_config(trial)
        
        assert "model_name" in config
        assert "learning_rate" in config
        assert "batch_size" in config


class TestOptunaNotAvailable:
    """Test suite for when Optuna is not available."""
    
    @pytest.mark.skipif(OPTUNA_AVAILABLE, reason="Optuna is installed")
    def test_search_space_raises_import_error(self):
        """Test that ImportError is raised when Optuna not available."""
        with pytest.raises(ImportError, match="Optuna is required"):
            OptunaSearchSpace()
