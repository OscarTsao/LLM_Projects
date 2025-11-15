"""
HPO search wrapper for Optuna and Ray Tune.

Provides a unified interface for hyperparameter optimization with support for:
- Stage 1: Augmenter selection
- Stage 2: Hyperparameter tuning
- Pruning underperforming trials
- Distributed execution
"""

from typing import Dict, Any, Optional, Callable, List
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import yaml
from pathlib import Path


class HPOSearch:
    """
    Wrapper for hyperparameter optimization.
    
    Supports both Optuna and Ray Tune backends with a unified interface.
    
    Attributes:
        config: Configuration dictionary
        engine: HPO engine ("optuna" or "ray")
        stage: HPO stage (1 or 2)
    """
    
    def __init__(
        self,
        config_path: str = "configs/run.yaml",
        stage: int = 1,
    ):
        """
        Initialize HPO search.
        
        Args:
            config_path: Path to configuration file
            stage: HPO stage (1 or 2)
        """
        self.config = self._load_config(config_path)
        self.stage = stage
        self.engine = self.config["hpo"]["engine"]
        
        # Get stage-specific config
        stage_key = f"stage{stage}"
        self.stage_config = self.config["hpo"][stage_key]
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML."""
        path = Path(config_path)
        with open(path) as f:
            return yaml.safe_load(f)
    
    def create_study(
        self,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
    ) -> optuna.Study:
        """
        Create Optuna study.
        
        Args:
            study_name: Name for the study
            storage: Storage URL (e.g., sqlite:///hpo.db)
            
        Returns:
            Optuna study instance
        """
        if self.engine != "optuna":
            raise ValueError(f"create_study only supported for optuna, got {self.engine}")
        
        # Determine optimization direction
        direction = self.stage_config.get("objective", "maximize")
        
        # Create sampler
        sampler = TPESampler(seed=self.config["global"]["seed"])
        
        # Create pruner if enabled
        pruner = None
        if self.stage_config.get("pruning", {}).get("enabled", False):
            patience = self.stage_config["pruning"].get("patience", 3)
            pruner = MedianPruner(
                n_startup_trials=self.stage_config["pruning"].get("min_epochs", 2),
                n_warmup_steps=patience,
            )
        
        # Create study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )
        
        return study
    
    def suggest_combo(
        self,
        trial: optuna.Trial,
        valid_combos: Dict[int, List[List[str]]],
    ) -> List[str]:
        """
        Suggest an augmentation combination for Stage 1.
        
        Args:
            trial: Optuna trial
            valid_combos: Dictionary of valid combinations by k
            
        Returns:
            Selected combo as list of augmenter names
        """
        if self.stage != 1:
            raise ValueError("suggest_combo only valid for Stage 1")
        
        # Suggest k value
        k_values = sorted(valid_combos.keys())
        k = trial.suggest_categorical("k", k_values)
        
        # Suggest combo from valid options for this k
        combos_for_k = valid_combos[k]
        combo_idx = trial.suggest_int("combo_idx", 0, len(combos_for_k) - 1)
        
        return combos_for_k[combo_idx]
    
    def suggest_hyperparams(
        self,
        trial: optuna.Trial,
        search_space: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Suggest hyperparameters for Stage 2.
        
        Args:
            trial: Optuna trial
            search_space: Custom search space (uses config if None)
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        if search_space is None:
            search_space = self.stage_config.get("search_space", {})
        
        params = {}
        
        # Model hyperparameters
        if "model" in search_space:
            model_space = search_space["model"]
            
            if "learning_rate" in model_space:
                lr_range = model_space["learning_rate"]
                params["learning_rate"] = trial.suggest_float(
                    "learning_rate", lr_range[0], lr_range[1], log=True
                )
            
            if "batch_size" in model_space:
                batch_sizes = model_space["batch_size"]
                params["batch_size"] = trial.suggest_categorical(
                    "batch_size", batch_sizes
                )
            
            if "weight_decay" in model_space:
                wd_range = model_space["weight_decay"]
                params["weight_decay"] = trial.suggest_float(
                    "weight_decay", wd_range[0], wd_range[1]
                )
            
            if "warmup_ratio" in model_space:
                warmup_range = model_space["warmup_ratio"]
                params["warmup_ratio"] = trial.suggest_float(
                    "warmup_ratio", warmup_range[0], warmup_range[1]
                )
            
            if "max_epochs" in model_space:
                epoch_range = model_space["max_epochs"]
                params["max_epochs"] = trial.suggest_int(
                    "max_epochs", epoch_range[0], epoch_range[1]
                )
        
        # Augmentation intensity
        if "augmentation" in search_space:
            aug_space = search_space["augmentation"]
            
            if "intensity" in aug_space:
                intensity_range = aug_space["intensity"]
                params["aug_intensity"] = trial.suggest_float(
                    "aug_intensity", intensity_range[0], intensity_range[1]
                )
        
        return params
    
    def run_optimization(
        self,
        objective_fn: Callable,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
    ) -> optuna.Study:
        """
        Run optimization.
        
        Args:
            objective_fn: Objective function to minimize/maximize
            n_trials: Number of trials (uses config if None)
            timeout: Timeout in seconds (uses config if None)
            study_name: Name for study
            
        Returns:
            Completed study
        """
        if self.engine != "optuna":
            raise NotImplementedError(f"Engine {self.engine} not yet supported")
        
        # Get trial count and timeout from config if not provided
        if n_trials is None:
            n_trials = self.stage_config.get("trials", 100)
        
        if timeout is None:
            timeout_hours = self.stage_config.get("timeout_hours")
            if timeout_hours:
                timeout = timeout_hours * 3600
        
        # Create study
        study = self.create_study(study_name=study_name)
        
        # Run optimization
        study.optimize(
            objective_fn,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )
        
        return study
    
    def get_best_params(self, study: optuna.Study) -> Dict[str, Any]:
        """
        Get best parameters from completed study.
        
        Args:
            study: Completed study
            
        Returns:
            Dictionary of best parameters
        """
        return study.best_params
    
    def get_best_trial(self, study: optuna.Study) -> optuna.Trial:
        """Get best trial from study."""
        return study.best_trial
    
    def save_study(
        self,
        study: optuna.Study,
        output_path: str,
    ) -> None:
        """
        Save study results to file.
        
        Args:
            study: Study to save
            output_path: Path to output file
        """
        import pickle
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            pickle.dump(study, f)
        
        print(f"Saved study to {output_path}")
