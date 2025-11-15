"""Optuna search space definition for hyperparameter optimization.

This module defines the search space for HPO, mapping hyperparameters
to Optuna suggestions with support for conditional parameters.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import Optuna (optional dependency)
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")


@dataclass
class SearchSpaceConfig:
    """Configuration for search space definition."""
    
    # Model hyperparameters
    model_name: str = "categorical"  # categorical choice
    learning_rate: str = "log_uniform"  # log scale
    batch_size: str = "categorical"  # categorical choice
    max_epochs: int = 10  # fixed
    
    # Loss hyperparameters
    loss_type: str = "categorical"
    focal_gamma: str = "uniform"  # only if loss_type=focal
    label_smoothing: str = "uniform"
    
    # Optimizer hyperparameters
    optimizer: str = "categorical"
    weight_decay: str = "log_uniform"
    
    # Data augmentation
    augmentation_prob: str = "uniform"
    augmentation_method: str = "categorical"


class OptunaSearchSpace:
    """Optuna search space for hyperparameter optimization."""

    def __init__(self, config: Optional[SearchSpaceConfig] = None):
        """Initialize search space.

        Args:
            config: Search space configuration
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for search space definition. "
                "Install with: pip install optuna"
            )

        self.config = config or SearchSpaceConfig()

        logger.info("Initialized OptunaSearchSpace")
    
    def suggest_hyperparameters(self, trial: 'optuna.Trial') -> Dict[str, Any]:
        """Suggest hyperparameters for a trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {}

        # Model selection (DeBERTa-v3-base by default)
        params["model_name"] = trial.suggest_categorical(
            "model_name",
            ["deberta-v3-base"]  # Using only DeBERTa-v3-base as specified
        )

        # Encoder mode
        params["encoder_mode"] = trial.suggest_categorical(
            "encoder_mode",
            ["shared", "separate"]
        )

        # Combination method
        params["combination_method"] = trial.suggest_categorical(
            "combination_method",
            ["concat", "add"]
        )

        # Learning rate (log scale)
        params["learning_rate"] = trial.suggest_float(
            "learning_rate",
            1e-5, 1e-3,
            log=True
        )

        # Batch size
        params["batch_size"] = trial.suggest_categorical(
            "batch_size",
            [8, 16, 32]
        )
        
        # Loss type
        params["loss_type"] = trial.suggest_categorical(
            "loss_type",
            ["bce", "focal", "weighted_bce", "hybrid"]
        )
        
        # Conditional: focal gamma (only if loss_type=focal or hybrid)
        if params["loss_type"] in ["focal", "hybrid"]:
            params["focal_gamma"] = trial.suggest_float(
                "focal_gamma",
                1.0, 5.0
            )
        
        # Label smoothing
        params["label_smoothing"] = trial.suggest_float(
            "label_smoothing",
            0.0, 0.2
        )
        
        # Optimizer
        params["optimizer"] = trial.suggest_categorical(
            "optimizer",
            ["adam", "adamw", "sgd"]
        )
        
        # Weight decay (log scale)
        params["weight_decay"] = trial.suggest_float(
            "weight_decay",
            1e-6, 1e-2,
            log=True
        )
        
        # Data augmentation probability
        params["augmentation_prob"] = trial.suggest_float(
            "augmentation_prob",
            0.0, 0.5
        )
        
        # Augmentation method
        params["augmentation_method"] = trial.suggest_categorical(
            "augmentation_method",
            ["synonym", "insert", "swap", "char_perturb", "eda", "none"]
        )
        
        # Dropout
        params["dropout"] = trial.suggest_float(
            "dropout",
            0.0, 0.5
        )
        
        # Pooling strategy
        params["pooling_strategy"] = trial.suggest_categorical(
            "pooling_strategy",
            ["cls", "mean", "max"]
        )
        
        # Gradient accumulation steps
        params["gradient_accumulation_steps"] = trial.suggest_categorical(
            "gradient_accumulation_steps",
            [1, 2, 4]
        )
        
        return params
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate suggested parameters.
        
        Args:
            params: Dictionary of parameters
            
        Returns:
            True if valid, False otherwise
        """
        # Check required parameters
        required = [
            "model_name", "learning_rate", "batch_size",
            "loss_type", "optimizer"
        ]
        
        for key in required:
            if key not in params:
                logger.error(f"Missing required parameter: {key}")
                return False
        
        # Validate ranges
        if not 1e-6 <= params["learning_rate"] <= 1e-2:
            logger.error(f"Invalid learning_rate: {params['learning_rate']}")
            return False
        
        if params["batch_size"] not in [8, 16, 32, 64]:
            logger.error(f"Invalid batch_size: {params['batch_size']}")
            return False
        
        # Validate conditional parameters
        if params["loss_type"] in ["focal", "hybrid"]:
            if "focal_gamma" not in params:
                logger.error("focal_gamma required for focal/hybrid loss")
                return False
            
            if not 1.0 <= params["focal_gamma"] <= 5.0:
                logger.error(f"Invalid focal_gamma: {params['focal_gamma']}")
                return False
        
        return True


def create_search_space(config: Optional[SearchSpaceConfig] = None) -> 'OptunaSearchSpace':
    """Factory function to create search space.

    Args:
        config: Search space configuration

    Returns:
        Initialized search space

    Example:
        search_space = create_search_space()
        params = search_space.suggest_hyperparameters(trial)
    """
    return OptunaSearchSpace(config)


def suggest_trial_config(trial: 'optuna.Trial') -> Dict[str, Any]:
    """Suggest a complete trial configuration.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Dictionary of trial configuration
        
    Example:
        def objective(trial):
            config = suggest_trial_config(trial)
            # Train model with config
            return validation_metric
    """
    search_space = create_search_space()
    params = search_space.suggest_hyperparameters(trial)
    
    # Validate
    if not search_space.validate_params(params):
        raise ValueError("Invalid trial configuration")
    
    return params


# Predefined search space configurations
SEARCH_SPACE_SMALL = SearchSpaceConfig(
    # Smaller search space for quick experiments
)

SEARCH_SPACE_FULL = SearchSpaceConfig(
    # Full search space for comprehensive HPO
)

