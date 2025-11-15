"""Enhanced MLflow utilities for experiment tracking.

These helpers centralise MLflow configuration and robust logging with
defensive fallbacks so training loops remain resilient when the tracking
backend is unreachable or users run locally without a server.
"""

import contextlib
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow
from omegaconf import DictConfig, OmegaConf


def get_git_sha() -> str | None:
    """Get current git SHA if in a git repository."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_config_hash(config: DictConfig) -> str:
    """Generate hash of configuration for reproducibility tracking."""
    config_str = OmegaConf.to_yaml(config)
    # Using MD5 for non-security purposes (config fingerprinting only)
    return hashlib.md5(config_str.encode(), usedforsecurity=False).hexdigest()[:8]


def resolve_tracking_uri(tracking_uri: str, base_dir: Path) -> str:
    """Resolve MLflow tracking URI to an absolute value when needed."""
    parsed = urlparse(tracking_uri)

    if parsed.scheme == "sqlite":
        if tracking_uri.startswith("sqlite:////"):
            db_path = Path(parsed.path)
        else:
            db_path = Path(parsed.path.lstrip("/"))
            db_path = (base_dir / db_path).resolve()
        return f"sqlite:///{db_path}"

    if parsed.scheme in {
        "http",
        "https",
        "databricks",
        "mysql",
        "postgresql",
        "postgresql+psycopg2",
        "mysql+pymysql",
    }:
        return tracking_uri

    if parsed.scheme == "file":
        candidate = (
            Path(f"{parsed.netloc}{parsed.path}")
            if parsed.netloc
            else Path(parsed.path)
        )
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        return candidate.as_uri()

    if not parsed.scheme:
        candidate = Path(tracking_uri)
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        return str(candidate)

    return tracking_uri


def resolve_artifact_location(
    artifact_location: str | None, base_dir: Path
) -> str | None:
    """Resolve artifact store location to an absolute URI when applicable."""
    if not artifact_location:
        return None

    parsed = urlparse(artifact_location)

    if parsed.scheme in {"http", "https", "s3", "gs", "wasbs", "azure", "dbfs"}:
        return artifact_location

    if parsed.scheme == "file":
        candidate = (
            Path(f"{parsed.netloc}{parsed.path}")
            if parsed.netloc
            else Path(parsed.path)
        )
    else:
        candidate = Path(artifact_location)

    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()

    return candidate.as_uri()


def configure_mlflow(
    tracking_uri: str,
    experiment_name: str,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    config: DictConfig | None = None,
    artifact_location: str | None = None,
) -> str:
    """
    Configure and start MLflow run with comprehensive tracking.

    Args:
        tracking_uri: MLflow tracking URI
        artifact_location: Optional artifact store location
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
            task_name = getattr(config.task, "name", None) or getattr(
                config.task, "task_name", None
            )
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
        experiment_id=experiment_id, run_name=run_name, tags=run_tags
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


def _log_dict_recursive(d: dict[str, Any], prefix: str = ""):
    """Recursively log nested dictionaries."""
    for key, value in d.items():
        param_name = f"{prefix}{key}" if prefix else key

        if isinstance(value, dict):
            _log_dict_recursive(value, prefix=f"{param_name}.")
        elif isinstance(value, list | tuple):
            # Log lists as JSON strings
            with contextlib.suppress(Exception):
                mlflow.log_param(param_name, json.dumps(value))
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


def log_artifacts(artifact_dir: Path, artifact_path: str | None = None):
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


def log_model_checkpoint(checkpoint_path: Path, artifact_path: str = "checkpoints"):
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


def log_metrics_dict(metrics: dict[str, float], step: int | None = None):
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
    report: dict[str, Any], report_path: Path, artifact_path: str = "reports"
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
    registered_model_name: str | None = None,
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
