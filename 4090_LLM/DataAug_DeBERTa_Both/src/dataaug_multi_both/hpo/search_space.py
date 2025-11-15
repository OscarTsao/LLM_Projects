"""Optuna search space definition for hyperparameter optimization.

This module defines the search space for HPO, mapping hyperparameters
to Optuna suggestions with support for conditional parameters.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import Optuna (optional dependency)
try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")

try:  # pragma: no cover - optional dependency
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class SearchSpaceConfig:
    """Configuration for search space definition."""

    # Model hyperparameters
    model_name: str = "microsoft/deberta-v3-base"  # fixed model identifier
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

        self._focal_loss_variants = {
            "focal",
            "hybrid_bce_focal",
            "hybrid_weighted_bce_focal",
        }
        self._adaptive_focal_variants = {
            "adaptive_focal",
            "hybrid_bce_adaptive_focal",
            "hybrid_weighted_bce_adaptive_focal",
        }
        self._batch_size_choices = self._build_batch_size_choices()

        logger.info("Initialized OptunaSearchSpace")

    def _build_batch_size_choices(self) -> List[int]:
        """Determine permissible batch sizes based on available GPU memory."""
        base_choices = [8, 16, 32]
        max_batch_size = 32

        if torch is not None and torch.cuda.is_available():  # pragma: no cover - depends on runtime GPU
            try:
                device_name = torch.cuda.get_device_name(0).lower()
                total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                if "5090" in device_name:
                    max_batch_size = 128
                elif total_memory_gb >= 24:
                    max_batch_size = 96
                elif total_memory_gb >= 16:
                    max_batch_size = 64
                else:
                    max_batch_size = 48

                logger.info(
                    "Detected GPU '%s' with %.1f GB; setting batch size upper bound to %s",
                    device_name,
                    total_memory_gb,
                    max_batch_size,
                )
            except Exception as exc:  # pragma: no cover - best effort hardware detection
                logger.warning("Failed to auto-detect GPU batch size capability: %s", exc)
        else:
            logger.info("No CUDA device detected; using conservative batch size upper bound.")

        dynamic_choices = [size for size in [48, 64, 96, 128, 160, 192, 256] if size <= max_batch_size]
        full_choices = sorted(set(base_choices + dynamic_choices))

        if not full_choices:
            full_choices = base_choices

        return full_choices

    def suggest_hyperparameters(self, trial: 'optuna.Trial') -> Dict[str, Any]:
        """Suggest hyperparameters for a trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {}

        # Model selection (fixed)
        params["model_name"] = self.config.model_name

        # Learning rate (log scale)
        params["learning_rate"] = trial.suggest_float(
            "learning_rate",
            1e-5,
            1e-3,
            log=True
        )

        # Batch size
        params["batch_size"] = trial.suggest_categorical(
            "batch_size",
            self._batch_size_choices
        )

        # Loss type
        params["loss_type"] = trial.suggest_categorical(
            "loss_type",
            [
                "bce",
                "focal",
                "adaptive_focal",
                "weighted_bce",
                "hybrid_bce_focal",
                "hybrid_bce_adaptive_focal",
                "hybrid_weighted_bce_focal",
                "hybrid_weighted_bce_adaptive_focal",
            ]
        )

        # Conditional: focal gamma (only if the loss uses fixed focal)
        if params["loss_type"] in self._focal_loss_variants:
            params["focal_gamma"] = trial.suggest_float(
                "focal_gamma",
                1.0,
                5.0,
            )

        # Label smoothing
        params["label_smoothing"] = trial.suggest_float(
            "label_smoothing",
            0.0,
            0.2,
        )

        # Optimizer
        params["optimizer"] = trial.suggest_categorical(
            "optimizer",
            ["adam", "adamw", "sgd"]
        )

        # Weight decay (log scale)
        params["weight_decay"] = trial.suggest_float(
            "weight_decay",
            1e-6,
            1e-2,
            log=True
        )

        # Data augmentation probability
        params["augmentation_prob"] = trial.suggest_float(
            "augmentation_prob",
            0.0,
            0.5,
        )

        # Augmentation method
        params["augmentation_method"] = trial.suggest_categorical(
            "augmentation_method",
            ["synonym", "insert", "swap", "char_perturb", "eda", "none"]
        )

        # Dropout
        params["dropout"] = trial.suggest_float(
            "dropout",
            0.0,
            0.5,
        )

        # Pooling strategy
        params["pooling_strategy"] = trial.suggest_categorical(
            "pooling_strategy",
            ["cls", "mean", "max"]
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

        if params["model_name"] != self.config.model_name:
            logger.error(
                "Invalid model_name: %s (expected %s)",
                params["model_name"],
                self.config.model_name,
            )
            return False

        if params["batch_size"] not in self._batch_size_choices:
            logger.error(f"Invalid batch_size: {params['batch_size']}")
            return False

        # Validate conditional parameters
        if params["loss_type"] in self._focal_loss_variants:
            if "focal_gamma" not in params:
                logger.error("focal_gamma required for selected focal loss variant")
                return False

            if not 1.0 <= params["focal_gamma"] <= 5.0:
                logger.error(f"Invalid focal_gamma: {params['focal_gamma']}")
                return False

        adaptive_allowed = self._focal_loss_variants | self._adaptive_focal_variants | {
            "bce",
            "weighted_bce",
        }
        if params["loss_type"] not in adaptive_allowed:
            logger.error(f"Invalid loss_type: {params['loss_type']}")
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
