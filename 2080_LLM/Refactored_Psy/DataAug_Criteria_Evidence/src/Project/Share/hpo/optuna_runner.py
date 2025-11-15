"""Comprehensive Optuna-based hyperparameter optimization."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import TPESampler
from omegaconf import DictConfig, OmegaConf


class OptunaRunner:
    """
    Unified Optuna HPO runner with multi-stage support.
    
    Features:
    - TPE sampler with multivariate optimization
    - MedianPruner and HyperbandPruner support
    - MLflow integration for tracking
    - Study persistence
    - Unified search space across all stages
    """
    
    def __init__(
        self,
        study_name: str,
        direction: str = "maximize",
        metric: str = "val_f1_macro",
        storage: Optional[str] = None,
        sampler_config: Optional[Dict[str, Any]] = None,
        pruner_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize OptunaRunner.
        
        Args:
            study_name: Name of the Optuna study
            direction: Optimization direction ('minimize' or 'maximize')
            metric: Metric to optimize
            storage: Optional database URL for study persistence
            sampler_config: Sampler configuration
            pruner_config: Pruner configuration
        """
        self.study_name = study_name
        self.direction = direction
        self.metric = metric
        
        # Create sampler
        sampler = self._create_sampler(sampler_config)
        
        # Create pruner
        pruner = self._create_pruner(pruner_config)
        
        # Create or load study
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
        )
    
    def _create_sampler(
        self,
        config: Optional[Dict[str, Any]]
    ) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler from configuration."""
        if config is None:
            return TPESampler()
        
        sampler_type = config.get("type", "tpe")
        
        if sampler_type == "tpe":
            return TPESampler(
                n_startup_trials=config.get("n_startup_trials", 10),
                multivariate=config.get("multivariate", True),
                group=config.get("group", True),
            )
        elif sampler_type == "random":
            return optuna.samplers.RandomSampler()
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
    
    def _create_pruner(
        self,
        config: Optional[Dict[str, Any]]
    ) -> Optional[optuna.pruners.BasePruner]:
        """Create Optuna pruner from configuration."""
        if config is None:
            return None
        
        pruner_type = config.get("type")
        
        if pruner_type == "median":
            return MedianPruner(
                n_startup_trials=config.get("n_startup_trials", 5),
                n_warmup_steps=config.get("n_warmup_steps", 2),
                interval_steps=config.get("interval_steps", 1),
            )
        elif pruner_type == "hyperband":
            return HyperbandPruner(
                min_resource=config.get("min_resource", 1),
                max_resource=config.get("max_resource", 10),
                reduction_factor=config.get("reduction_factor", 3),
            )
        elif pruner_type is None or pruner_type == "none":
            return None
        else:
            raise ValueError(f"Unknown pruner type: {pruner_type}")
    
    def suggest_hyperparameters(
        self,
        trial: optuna.Trial,
        search_space: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Suggest hyperparameters from unified search space.
        
        Supports all stages with flexible parameter definitions.
        
        Args:
            trial: Optuna trial
            search_space: Search space configuration
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {}
        
        for param_name, param_config in search_space.items():
            param_type = param_config["type"]
            
            if param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )
            
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            
            elif param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            
            elif param_type == "uniform":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=False,
                )
            
            elif param_type == "loguniform":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=True,
                )
            
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        
        return params
    
    def optimize(
        self,
        objective_fn: Callable,
        n_trials: int,
        search_space: Dict[str, Dict[str, Any]],
        mlflow_tracking_uri: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """
        Run hyperparameter optimization.
        
        Args:
            objective_fn: Objective function to optimize
            n_trials: Number of trials to run
            search_space: Hyperparameter search space
            mlflow_tracking_uri: Optional MLflow tracking URI
            timeout: Optional timeout in seconds
        """
        # Setup MLflow callback
        callbacks = []
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow_callback = MLflowCallback(
                tracking_uri=mlflow_tracking_uri,
                metric_name=self.metric,
            )
            callbacks.append(mlflow_callback)
        
        # Create wrapped objective
        def wrapped_objective(trial):
            # Suggest hyperparameters
            params = self.suggest_hyperparameters(trial, search_space)
            
            # Call user objective function
            return objective_fn(trial, params)
        
        # Run optimization
        self.study.optimize(
            wrapped_objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks if callbacks else None,
            show_progress_bar=True,
        )
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best hyperparameters from study."""
        return self.study.best_params
    
    def get_best_value(self) -> float:
        """Get best objective value from study."""
        return self.study.best_value
    
    def get_best_trial(self) -> optuna.trial.FrozenTrial:
        """Get best trial from study."""
        return self.study.best_trial
    
    def print_best_trial(self):
        """Print information about best trial."""
        best_trial = self.study.best_trial
        
        print("\n" + "=" * 70)
        print("Best Trial Information".center(70))
        print("=" * 70)
        print(f"  Trial number: {best_trial.number}")
        print(f"  {self.metric}: {best_trial.value:.6f}")
        print("\n  Best hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        print("=" * 70 + "\n")
    
    def save_study(self, output_path: Path):
        """
        Save study to file.
        
        Args:
            output_path: Path to save study
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save using joblib
        import joblib
        joblib.dump(self.study, output_path)
        print(f"Study saved to {output_path}")
    
    def export_best_config(self, output_path: Path):
        """
        Export best configuration as YAML.
        
        Args:
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        best_params = self.get_best_params()
        
        # Convert to OmegaConf for pretty YAML
        config = OmegaConf.create(best_params)
        
        with open(output_path, "w") as f:
            OmegaConf.save(config, f)
        
        print(f"Best configuration exported to {output_path}")
    
    def export_trials_history(self, output_path: Path):
        """
        Export trials history as JSON.
        
        Args:
            output_path: Path to save trials history
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        trials_data = []
        for trial in self.study.trials:
            trial_dict = {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            }
            trials_data.append(trial_dict)
        
        with open(output_path, "w") as f:
            json.dump(trials_data, f, indent=2)
        
        print(f"Trials history exported to {output_path}")
    
    @staticmethod
    def load_study(study_path: Path) -> "OptunaRunner":
        """
        Load study from file.
        
        Args:
            study_path: Path to saved study
            
        Returns:
            OptunaRunner instance with loaded study
        """
        import joblib
        study = joblib.load(study_path)
        
        runner = OptunaRunner(
            study_name=study.study_name,
            direction=study.direction.name.lower()
        )
        runner.study = study
        
        return runner


def create_search_space_from_config(
    config: DictConfig
) -> Dict[str, Dict[str, Any]]:
    """
    Create Optuna search space from Hydra config.
    
    Args:
        config: Hydra configuration with search_space section
        
    Returns:
        Optuna search space dictionary
    """
    if not hasattr(config, "search_space"):
        return {}
    
    search_space = OmegaConf.to_container(config.search_space, resolve=True)
    return search_space
