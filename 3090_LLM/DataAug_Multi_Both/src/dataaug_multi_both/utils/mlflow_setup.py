"""MLflow tracking backend setup and utilities.

This module provides utilities for setting up and managing MLflow tracking,
including experiment creation, run management, and metric logging.

Implements FR-003: MLflow tracking for all experiments
Implements FR-012: Auditability via MLflow
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

logger = logging.getLogger(__name__)


def setup_mlflow(
    tracking_uri: str = "sqlite:///experiments/mlflow_db/mlflow.db",
    experiment_name: str = "storage-optimized-hpo",
    create_db_dir: bool = True,
) -> str:
    """Set up MLflow tracking backend.

    Args:
        tracking_uri: MLflow tracking URI (default: SQLite database)
        experiment_name: Name of the MLflow experiment
        create_db_dir: Whether to create the database directory if it doesn't exist

    Returns:
        Experiment ID

    Example:
        experiment_id = setup_mlflow(
            tracking_uri="sqlite:///experiments/mlflow_db/mlflow.db",
            experiment_name="my_experiment"
        )
    """
    # Create database directory if needed
    if create_db_dir and tracking_uri.startswith("sqlite:///"):
        db_path = Path(tracking_uri.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created MLflow database directory: {db_path.parent}")

    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI set to: {tracking_uri}")

    # Create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created MLflow experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            logger.info(
                f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})"
            )
    except Exception as e:
        logger.error(f"Failed to create/get MLflow experiment: {e}")
        raise

    # Set experiment as active
    mlflow.set_experiment(experiment_name)

    return experiment_id


@contextmanager
def mlflow_run(
    run_name: str | None = None, tags: dict[str, Any] | None = None, nested: bool = False
):
    """Context manager for MLflow runs.

    Args:
        run_name: Name for the run (optional)
        tags: Dictionary of tags to set on the run
        nested: Whether this is a nested run

    Yields:
        MLflow run object

    Example:
        with mlflow_run(run_name="trial_001", tags={"trial_id": "001"}):
            mlflow.log_param("learning_rate", 0.001)
            mlflow.log_metric("accuracy", 0.95)
    """
    try:
        run = mlflow.start_run(run_name=run_name, nested=nested)

        # Set tags if provided
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        logger.info(f"Started MLflow run: {run.info.run_id}")

        yield run

    except Exception as e:
        logger.error(f"Error in MLflow run: {e}")
        raise
    finally:
        mlflow.end_run()
        logger.info("Ended MLflow run")


def log_params_safe(params: dict[str, Any]):
    """Log parameters to MLflow with error handling.

    Args:
        params: Dictionary of parameters to log

    Note:
        Silently skips parameters that fail to log (e.g., invalid types)
    """
    for key, value in params.items():
        try:
            # Convert value to string if it's not a basic type
            if not isinstance(value, str | int | float | bool):
                value = str(value)
            mlflow.log_param(key, value)
        except Exception as e:
            logger.warning(f"Failed to log parameter {key}: {e}")


def log_metrics_safe(metrics: dict[str, float], step: int | None = None):
    """Log metrics to MLflow with error handling.

    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number for the metrics

    Note:
        Silently skips metrics that fail to log
    """
    for key, value in metrics.items():
        try:
            if step is not None:
                mlflow.log_metric(key, value, step=step)
            else:
                mlflow.log_metric(key, value)
        except Exception as e:
            logger.warning(f"Failed to log metric {key}: {e}")


def log_artifact_safe(local_path: str, artifact_path: str | None = None):
    """Log artifact to MLflow with error handling.

    Args:
        local_path: Path to the local file to log
        artifact_path: Optional path within the artifact directory

    Note:
        Logs warning if artifact fails to upload
    """
    try:
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Logged artifact: {local_path}")
    except Exception as e:
        logger.warning(f"Failed to log artifact {local_path}: {e}")


def get_run_info(run_id: str) -> dict[str, Any] | None:
    """Get information about an MLflow run.

    Args:
        run_id: MLflow run ID

    Returns:
        Dictionary with run information, or None if run not found
    """
    try:
        run = mlflow.get_run(run_id)
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "artifact_uri": run.info.artifact_uri,
            "params": run.data.params,
            "metrics": run.data.metrics,
            "tags": run.data.tags,
        }
    except Exception as e:
        logger.error(f"Failed to get run info for {run_id}: {e}")
        return None


def search_runs_by_tag(tag_key: str, tag_value: str, experiment_ids: list | None = None) -> list:
    """Search for runs by tag.

    Args:
        tag_key: Tag key to search for
        tag_value: Tag value to match
        experiment_ids: Optional list of experiment IDs to search in

    Returns:
        List of matching runs
    """
    try:
        filter_string = f"tags.{tag_key} = '{tag_value}'"
        runs = mlflow.search_runs(experiment_ids=experiment_ids, filter_string=filter_string)
        return runs.to_dict("records") if not runs.empty else []
    except Exception as e:
        logger.error(f"Failed to search runs: {e}")
        return []
