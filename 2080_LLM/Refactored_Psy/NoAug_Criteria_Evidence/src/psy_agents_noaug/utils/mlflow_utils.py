"""Enhanced MLflow utilities for experiment tracking."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from omegaconf import DictConfig, OmegaConf


def get_git_sha() -> Optional[str]:
    """Get current git SHA if in a git repository."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_config_hash(config: DictConfig) -> str:
    """Generate hash of configuration for reproducibility tracking."""
    config_str = OmegaConf.to_yaml(config)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def configure_mlflow(
    tracking_uri: str,
    experiment_name: str,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    config: Optional[DictConfig] = None,
) -> str:
    """
    Configure and start MLflow run with comprehensive tracking.
    
    Args:
        tracking_uri: MLflow tracking URI
        experiment_name: Name of the experiment
        run_name: Optional name for the run
        tags: Optional additional tags
        config: Optional Hydra config for automatic tagging
    
    Returns:
        Run ID
    """
    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create or get experiment
    try:
        artifact_location = None
        if tracking_uri.startswith("/"):
            artifact_location = f"{tracking_uri.rstrip('/')}/{experiment_name}"
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
        )
    except Exception:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise
        experiment_id = experiment.experiment_id
    
    # Prepare tags
    run_tags = tags or {}
    
    # Add git SHA if available
    git_sha = get_git_sha()
    if git_sha:
        run_tags["git_sha"] = git_sha
    
    # Add config hash if available
    if config:
        run_tags["config_hash"] = get_config_hash(config)
        
        # Extract key config elements as tags
        task_name = None
        if hasattr(config, "task"):
            task_name = getattr(config.task, "name", None) or getattr(config.task, "task_name", None)
        if task_name:
            run_tags["task"] = str(task_name)

        if hasattr(config, "model"):
            model_name = getattr(config.model, "encoder_name", None)
            if not model_name and hasattr(config.model, "_target_"):
                model_name = config.model._target_.split(".")[-1]
            if model_name:
                run_tags["model"] = str(model_name)
        
        if hasattr(config, "seed"):
            run_tags["seed"] = str(config.seed)
    
    # Start run
    run = mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags=run_tags
    )
    
    return run.info.run_id


def log_config(config: DictConfig, prefix: str = ""):
    """
    Log Hydra configuration to MLflow with flattening.
    
    Args:
        config: Hydra configuration
        prefix: Prefix for parameter names
    """
    # Convert OmegaConf to dict
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = config
    
    # Flatten and log
    _log_dict_recursive(config_dict, prefix)
    
    # Also log as artifact
    config_path = Path("config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)
    mlflow.log_artifact(str(config_path))
    config_path.unlink()


def _log_dict_recursive(d: Dict[str, Any], prefix: str = ""):
    """Recursively log nested dictionaries."""
    for key, value in d.items():
        param_name = f"{prefix}{key}" if prefix else key
        
        if isinstance(value, dict):
            _log_dict_recursive(value, prefix=f"{param_name}.")
        elif isinstance(value, (list, tuple)):
            # Log lists as JSON strings
            try:
                mlflow.log_param(param_name, json.dumps(value))
            except Exception:
                pass
        else:
            # Log scalar parameters
            try:
                # MLflow has 500 char limit for param values
                str_value = str(value)
                if len(str_value) > 500:
                    str_value = str_value[:497] + "..."
                mlflow.log_param(param_name, str_value)
            except Exception as e:
                print(f"Warning: Could not log parameter {param_name}: {e}")


def log_artifacts(
    artifact_dir: Path,
    artifact_path: Optional[str] = None
):
    """
    Log directory of artifacts to MLflow.
    
    Args:
        artifact_dir: Directory containing artifacts
        artifact_path: Optional path within MLflow artifacts
    """
    artifact_dir = Path(artifact_dir)
    
    if not artifact_dir.exists():
        print(f"Warning: Artifact directory {artifact_dir} does not exist")
        return
    
    try:
        mlflow.log_artifacts(str(artifact_dir), artifact_path=artifact_path)
    except Exception as e:
        print(f"Warning: Could not log artifacts from {artifact_dir}: {e}")


def log_model_checkpoint(
    checkpoint_path: Path,
    artifact_path: str = "checkpoints"
):
    """
    Log model checkpoint to MLflow.
    
    Args:
        checkpoint_path: Path to checkpoint file
        artifact_path: Path within MLflow artifacts
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint {checkpoint_path} does not exist")
        return
    
    try:
        mlflow.log_artifact(str(checkpoint_path), artifact_path=artifact_path)
    except Exception as e:
        print(f"Warning: Could not log checkpoint {checkpoint_path}: {e}")


def log_metrics_dict(metrics: Dict[str, float], step: Optional[int] = None):
    """
    Log multiple metrics to MLflow.
    
    Args:
        metrics: Dictionary of metric names to values
        step: Optional step number
    """
    try:
        mlflow.log_metrics(metrics, step=step)
    except Exception as e:
        print(f"Warning: Could not log metrics: {e}")


def log_evaluation_report(
    report: Dict[str, Any],
    report_path: Path,
    artifact_path: str = "reports"
):
    """
    Log evaluation report to MLflow.
    
    Args:
        report: Evaluation report dictionary
        report_path: Path to save report
        artifact_path: Path within MLflow artifacts
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save report as JSON
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Log as artifact
    try:
        mlflow.log_artifact(str(report_path), artifact_path=artifact_path)
    except Exception as e:
        print(f"Warning: Could not log report: {e}")


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
    try:
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )
    except Exception as e:
        print(f"Warning: Could not save model to MLflow: {e}")


def end_run(status: str = "FINISHED"):
    """
    End current MLflow run.
    
    Args:
        status: Run status (FINISHED, FAILED, KILLED)
    """
    try:
        mlflow.end_run(status=status)
    except Exception as e:
        print(f"Warning: Error ending MLflow run: {e}")
