"""Optuna-based hyperparameter optimization."""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback


class OptunaHPO:
    """
    Hyperparameter optimization using Optuna.
    
    Supports:
    - Multi-stage HPO (coarse -> fine)
    - MLflow integration
    - Study persistence
    """
    
    def __init__(
        self,
        study_name: str,
        direction: str = "minimize",
        storage: Optional[str] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ):
        """
        Initialize HPO runner.
        
        Args:
            study_name: Name of the Optuna study
            direction: Optimization direction ('minimize' or 'maximize')
            storage: Optional database URL for study persistence
            sampler: Optional Optuna sampler
            pruner: Optional Optuna pruner
        """
        self.study_name = study_name
        self.direction = direction
        
        # Create or load study
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
        )
    
    def suggest_hyperparameters(
        self,
        trial: optuna.Trial,
        search_space: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Suggest hyperparameters from search space.
        
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
            objective_fn: Objective function to minimize/maximize
            n_trials: Number of trials to run
            search_space: Hyperparameter search space
            mlflow_tracking_uri: Optional MLflow tracking URI
            timeout: Optional timeout in seconds
        """
        # Setup MLflow callback
        mlflow_callback = None
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow_callback = MLflowCallback(
                tracking_uri=mlflow_tracking_uri,
                metric_name="objective_value",
            )
        
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
            callbacks=[mlflow_callback] if mlflow_callback else None,
        )
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best hyperparameters from study."""
        return self.study.best_params
    
    def get_best_value(self) -> float:
        """Get best objective value from study."""
        return self.study.best_value
    
    def print_best_trial(self):
        """Print information about best trial."""
        best_trial = self.study.best_trial
        
        print("\n" + "=" * 60)
        print("Best Trial Information")
        print("=" * 60)
        print(f"  Trial number: {best_trial.number}")
        print(f"  Objective value: {best_trial.value:.6f}")
        print("\n  Best hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        print("=" * 60 + "\n")
    
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
    
    @staticmethod
    def load_study(study_path: Path) -> "OptunaHPO":
        """
        Load study from file.
        
        Args:
            study_path: Path to saved study
            
        Returns:
            OptunaHPO instance with loaded study
        """
        import joblib
        study = joblib.load(study_path)
        
        hpo = OptunaHPO(study_name=study.study_name)
        hpo.study = study
        
        return hpo


def create_search_space_from_config(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Create Optuna search space from Hydra config.
    
    Args:
        config: Hydra configuration dictionary
        
    Returns:
        Optuna search space dictionary
    """
    search_space = {}
    
    for key, value in config.items():
        if isinstance(value, dict) and "type" in value:
            search_space[key] = value
    
    return search_space
