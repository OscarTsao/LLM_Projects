"""MLflow utilities for experiment tracking."""

from pathlib import Path
from typing import Any, Dict, Optional

import mlflow


def start_mlflow_run(
    experiment_name: str,
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
):
    """
    Start MLflow run with experiment tracking.
    
    Args:
        experiment_name: Name of the experiment
        run_name: Optional name for the run
        tracking_uri: Optional MLflow tracking URI
        tags: Optional tags for the run
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except Exception:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    
    mlflow.start_run(experiment_id=experiment_id, run_name=run_name, tags=tags)


def log_config(config: Dict[str, Any], prefix: str = ""):
    """
    Log configuration parameters to MLflow.
    
    Args:
        config: Configuration dictionary
        prefix: Prefix for parameter names
    """
    for key, value in config.items():
        param_name = f"{prefix}{key}" if prefix else key
        
        if isinstance(value, dict):
            # Recursively log nested configs
            log_config(value, prefix=f"{param_name}.")
        else:
            # Log scalar parameters
            try:
                mlflow.log_param(param_name, value)
            except Exception as e:
                print(f"Warning: Could not log parameter {param_name}: {e}")


def log_artifact_directory(directory: Path):
    """
    Log all files in a directory as MLflow artifacts.
    
    Args:
        directory: Directory to log
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Warning: Directory {directory} does not exist")
        return
    
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            try:
                mlflow.log_artifact(str(file_path))
            except Exception as e:
                print(f"Warning: Could not log artifact {file_path}: {e}")


def save_model_to_mlflow(
    model,
    artifact_path: str = "model",
    registered_model_name: Optional[str] = None,
):
    """
    Save PyTorch model to MLflow.
    
    Args:
        model: PyTorch model to save
        artifact_path: Artifact path in MLflow
        registered_model_name: Optional name for model registry
    """
    import torch
    
    mlflow.pytorch.log_model(
        model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
    )
