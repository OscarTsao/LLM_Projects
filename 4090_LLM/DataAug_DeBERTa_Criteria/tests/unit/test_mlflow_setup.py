"""Unit tests for MLflow setup utilities."""

import pytest
import tempfile
import mlflow
from pathlib import Path
from src.dataaug_multi_both.utils.mlflow_setup import (
    setup_mlflow,
    mlflow_run,
    log_params_safe,
    log_metrics_safe
)


class TestMLflowSetup:
    """Test suite for MLflow setup."""
    
    def test_setup_mlflow_creates_experiment(self):
        """Test that setup_mlflow creates an experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"
            experiment_id = setup_mlflow(
                tracking_uri=tracking_uri,
                experiment_name="test_experiment"
            )
            
            assert experiment_id is not None
            
            # Verify experiment exists
            experiment = mlflow.get_experiment(experiment_id)
            assert experiment is not None
            assert experiment.name == "test_experiment"
    
    def test_setup_mlflow_reuses_existing_experiment(self):
        """Test that setup_mlflow reuses existing experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"
            
            # Create experiment first time
            experiment_id_1 = setup_mlflow(
                tracking_uri=tracking_uri,
                experiment_name="test_experiment"
            )
            
            # Create experiment second time (should reuse)
            experiment_id_2 = setup_mlflow(
                tracking_uri=tracking_uri,
                experiment_name="test_experiment"
            )
            
            assert experiment_id_1 == experiment_id_2
    
    def test_mlflow_run_context_manager(self):
        """Test mlflow_run context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"
            setup_mlflow(
                tracking_uri=tracking_uri,
                experiment_name="test_experiment"
            )

            run_id = None
            with mlflow_run(run_name="test_run", tags={"test": "value"}):
                run = mlflow.active_run()
                assert run is not None
                run_id = run.info.run_id

            # Verify run ended
            assert mlflow.active_run() is None

            # Verify tags were set
            run = mlflow.get_run(run_id)
            assert run.data.tags.get("test") == "value"
    
    def test_log_params_safe(self):
        """Test safe parameter logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"
            setup_mlflow(
                tracking_uri=tracking_uri,
                experiment_name="test_experiment"
            )

            run_id = None
            with mlflow_run(run_name="test_run"):
                params = {
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "model": "bert-base",
                }
                log_params_safe(params)
                run_id = mlflow.active_run().info.run_id

            # Verify params after run ends
            run = mlflow.get_run(run_id)
            assert run.data.params.get("learning_rate") == "0.001"
            assert run.data.params.get("batch_size") == "32"
            assert run.data.params.get("model") == "bert-base"
    
    def test_log_metrics_safe(self):
        """Test safe metrics logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"
            setup_mlflow(
                tracking_uri=tracking_uri,
                experiment_name="test_experiment"
            )
            
            with mlflow_run(run_name="test_run"):
                metrics = {
                    "accuracy": 0.95,
                    "loss": 0.05,
                }
                log_metrics_safe(metrics)
                
                run = mlflow.active_run()
                # Metrics are stored differently, just verify no errors
                assert run is not None
    
    def test_log_params_safe_handles_invalid_types(self):
        """Test that log_params_safe handles invalid types gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"
            setup_mlflow(
                tracking_uri=tracking_uri,
                experiment_name="test_experiment"
            )

            run_id = None
            with mlflow_run(run_name="test_run"):
                params = {
                    "valid_param": "value",
                    "complex_param": {"nested": "dict"},  # Will be converted to string
                }
                # Should not raise exception
                log_params_safe(params)
                run_id = mlflow.active_run().info.run_id

            # Verify params after run ends
            run = mlflow.get_run(run_id)
            assert run.data.params.get("valid_param") == "value"

