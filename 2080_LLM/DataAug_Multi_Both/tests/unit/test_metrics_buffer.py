"""Unit tests for metrics buffer."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock
from src.dataaug_multi_both.hpo.metrics_buffer import (
    MetricEntry,
    MetricsBuffer,
    MLflowMetricsLogger
)


class TestMetricEntry:
    """Test suite for MetricEntry."""
    
    def test_entry_creation(self):
        """Test that metric entry can be created."""
        entry = MetricEntry(
            key="accuracy",
            value=0.85,
            step=100,
            timestamp=1234567890.0
        )
        assert entry.key == "accuracy"
        assert entry.value == 0.85
        assert entry.step == 100
        assert entry.timestamp == 1234567890.0
    
    def test_entry_to_dict(self):
        """Test converting entry to dictionary."""
        entry = MetricEntry(
            key="loss",
            value=0.5,
            step=50,
            timestamp=1234567890.0
        )
        entry_dict = entry.to_dict()
        
        assert entry_dict["key"] == "loss"
        assert entry_dict["value"] == 0.5
        assert entry_dict["step"] == 50
        assert entry_dict["timestamp"] == 1234567890.0
    
    def test_entry_from_dict(self):
        """Test creating entry from dictionary."""
        data = {
            "key": "f1",
            "value": 0.9,
            "step": 200,
            "timestamp": 1234567890.0
        }
        entry = MetricEntry.from_dict(data)
        
        assert entry.key == "f1"
        assert entry.value == 0.9
        assert entry.step == 200


class TestMetricsBuffer:
    """Test suite for MetricsBuffer."""
    
    def test_buffer_initialization(self):
        """Test that buffer can be initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_file = Path(tmpdir) / "metrics.jsonl"
            buffer = MetricsBuffer(buffer_file)
            
            assert buffer.buffer_file == buffer_file
            assert buffer.max_buffer_size_mb == 100.0
    
    def test_buffer_single_metric(self):
        """Test buffering a single metric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_file = Path(tmpdir) / "metrics.jsonl"
            buffer = MetricsBuffer(buffer_file)
            
            buffer.buffer_metric("accuracy", 0.85, step=100)
            
            assert buffer_file.exists()
            
            # Read and verify
            with open(buffer_file, 'r') as f:
                line = f.readline()
                data = json.loads(line)
                assert data["key"] == "accuracy"
                assert data["value"] == 0.85
                assert data["step"] == 100
    
    def test_buffer_multiple_metrics(self):
        """Test buffering multiple metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_file = Path(tmpdir) / "metrics.jsonl"
            buffer = MetricsBuffer(buffer_file)
            
            metrics = {
                "accuracy": 0.85,
                "loss": 0.5,
                "f1": 0.9
            }
            buffer.buffer_metrics(metrics, step=100)
            
            # Verify all metrics buffered
            count = buffer.get_buffered_count()
            assert count == 3
    
    def test_get_buffered_count(self):
        """Test getting buffered count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_file = Path(tmpdir) / "metrics.jsonl"
            buffer = MetricsBuffer(buffer_file)
            
            # Initially empty
            assert buffer.get_buffered_count() == 0
            
            # Add metrics
            buffer.buffer_metric("accuracy", 0.85, step=100)
            buffer.buffer_metric("loss", 0.5, step=100)
            
            assert buffer.get_buffered_count() == 2
    
    def test_replay_buffer_success(self):
        """Test successful buffer replay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_file = Path(tmpdir) / "metrics.jsonl"
            buffer = MetricsBuffer(buffer_file)
            
            # Buffer some metrics
            buffer.buffer_metric("accuracy", 0.85, step=100)
            buffer.buffer_metric("loss", 0.5, step=100)
            
            # Mock log function
            log_fn = Mock()
            
            # Replay
            success = buffer.replay_buffer(log_fn)
            
            assert success
            assert log_fn.call_count == 2
            assert not buffer_file.exists()  # Buffer cleared after success
    
    def test_replay_buffer_with_retry(self):
        """Test buffer replay with retry on failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_file = Path(tmpdir) / "metrics.jsonl"
            buffer = MetricsBuffer(buffer_file)
            
            buffer.buffer_metric("accuracy", 0.85, step=100)
            
            # Mock log function that fails once then succeeds
            log_fn = Mock(side_effect=[Exception("Network error"), None])
            
            success = buffer.replay_buffer(log_fn, max_retries=3, base_delay=0.01)
            
            assert success
            assert log_fn.call_count == 2  # Failed once, succeeded on retry
    
    def test_replay_buffer_failure(self):
        """Test buffer replay failure after max retries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_file = Path(tmpdir) / "metrics.jsonl"
            buffer = MetricsBuffer(buffer_file)
            
            buffer.buffer_metric("accuracy", 0.85, step=100)
            
            # Mock log function that always fails
            log_fn = Mock(side_effect=Exception("Network error"))
            
            success = buffer.replay_buffer(log_fn, max_retries=2, base_delay=0.01)
            
            assert not success
            assert buffer_file.exists()  # Buffer not cleared on failure
    
    def test_clear_buffer(self):
        """Test clearing buffer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_file = Path(tmpdir) / "metrics.jsonl"
            buffer = MetricsBuffer(buffer_file)
            
            buffer.buffer_metric("accuracy", 0.85, step=100)
            assert buffer_file.exists()
            
            buffer.clear_buffer()
            assert not buffer_file.exists()


class TestMLflowMetricsLogger:
    """Test suite for MLflowMetricsLogger."""
    
    def test_logger_initialization(self):
        """Test that logger can be initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_file = Path(tmpdir) / "metrics.jsonl"
            logger = MLflowMetricsLogger(buffer_file)
            
            assert logger.enable_buffering is True
            assert logger.buffer is not None
    
    def test_log_metric_success(self):
        """Test logging metric successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_file = Path(tmpdir) / "metrics.jsonl"
            
            # Mock MLflow client
            mlflow_client = Mock()
            
            logger = MLflowMetricsLogger(buffer_file, mlflow_client)
            logger.log_metric("accuracy", 0.85, step=100)
            
            mlflow_client.log_metric.assert_called_once_with("accuracy", 0.85, 100)
    
    def test_log_metric_failure_with_buffering(self):
        """Test logging metric failure with buffering enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_file = Path(tmpdir) / "metrics.jsonl"
            
            # Mock MLflow client that fails
            mlflow_client = Mock()
            mlflow_client.log_metric.side_effect = Exception("Network error")
            
            logger = MLflowMetricsLogger(buffer_file, mlflow_client, enable_buffering=True)
            logger.log_metric("accuracy", 0.85, step=100)
            
            # Metric should be buffered
            assert logger.buffer.get_buffered_count() == 1
    
    def test_log_metrics_success(self):
        """Test logging multiple metrics successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_file = Path(tmpdir) / "metrics.jsonl"
            
            mlflow_client = Mock()
            
            logger = MLflowMetricsLogger(buffer_file, mlflow_client)
            metrics = {"accuracy": 0.85, "loss": 0.5}
            logger.log_metrics(metrics, step=100)
            
            assert mlflow_client.log_metric.call_count == 2
    
    def test_log_metrics_failure_with_buffering(self):
        """Test logging multiple metrics failure with buffering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_file = Path(tmpdir) / "metrics.jsonl"
            
            mlflow_client = Mock()
            mlflow_client.log_metric.side_effect = Exception("Network error")
            
            logger = MLflowMetricsLogger(buffer_file, mlflow_client, enable_buffering=True)
            metrics = {"accuracy": 0.85, "loss": 0.5}
            logger.log_metrics(metrics, step=100)
            
            # All metrics should be buffered
            assert logger.buffer.get_buffered_count() == 2
    
    def test_replay_buffered_metrics(self):
        """Test replaying buffered metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_file = Path(tmpdir) / "metrics.jsonl"
            
            # First, fail to log and buffer
            mlflow_client = Mock()
            mlflow_client.log_metric.side_effect = Exception("Network error")
            
            logger = MLflowMetricsLogger(buffer_file, mlflow_client, enable_buffering=True)
            logger.log_metric("accuracy", 0.85, step=100)
            
            # Now replay with working client
            mlflow_client.log_metric.side_effect = None
            success = logger.replay_buffered_metrics()
            
            assert success
            assert logger.buffer.get_buffered_count() == 0
    
    def test_buffering_disabled(self):
        """Test that buffering can be disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_file = Path(tmpdir) / "metrics.jsonl"
            
            mlflow_client = Mock()
            mlflow_client.log_metric.side_effect = Exception("Network error")
            
            logger = MLflowMetricsLogger(buffer_file, mlflow_client, enable_buffering=False)
            
            # Should raise exception when buffering disabled
            with pytest.raises(Exception, match="Network error"):
                logger.log_metric("accuracy", 0.85, step=100)

