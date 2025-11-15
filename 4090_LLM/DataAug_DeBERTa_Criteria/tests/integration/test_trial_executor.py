"""Integration tests for trial executor."""

import pytest
import time
from unittest.mock import Mock
from src.dataaug_multi_both.hpo.trial_executor import (
    TrialSpec,
    TrialResult,
    TrialExecutor
)


class TestTrialSpec:
    """Test suite for TrialSpec."""
    
    def test_trial_spec_creation(self):
        """Test that trial spec can be created."""
        spec = TrialSpec(config={"learning_rate": 0.001})
        assert spec.config["learning_rate"] == 0.001
        assert spec.trial_id is None
    
    def test_trial_spec_ensure_id(self):
        """Test that ensure_id generates ID."""
        spec = TrialSpec(config={"learning_rate": 0.001})
        trial_id = spec.ensure_id()
        
        assert trial_id is not None
        assert spec.trial_id == trial_id
    
    def test_trial_spec_ensure_id_idempotent(self):
        """Test that ensure_id is idempotent."""
        spec = TrialSpec(config={"learning_rate": 0.001})
        trial_id_1 = spec.ensure_id()
        trial_id_2 = spec.ensure_id()
        
        assert trial_id_1 == trial_id_2


class TestTrialResult:
    """Test suite for TrialResult."""
    
    def test_trial_result_creation(self):
        """Test that trial result can be created."""
        result = TrialResult(
            trial_id="test_trial",
            metric=0.85,
            status="success",
            duration_seconds=120.5
        )
        
        assert result.trial_id == "test_trial"
        assert result.metric == 0.85
        assert result.status == "success"
        assert result.duration_seconds == 120.5


class TestTrialExecutor:
    """Test suite for TrialExecutor."""
    
    def create_mock_run_trial(self, success=True, metric=0.85):
        """Create a mock run_trial function."""
        def run_trial(spec: TrialSpec) -> TrialResult:
            trial_id = spec.ensure_id()
            return TrialResult(
                trial_id=trial_id,
                metric=metric if success else None,
                status="success" if success else "failed",
                duration_seconds=1.0
            )
        return run_trial
    
    def test_executor_initialization(self):
        """Test that executor can be initialized."""
        run_trial = self.create_mock_run_trial()
        executor = TrialExecutor(run_trial=run_trial)
        
        assert executor.run_trial == run_trial
        assert len(executor.results) == 0
    
    def test_execute_single_trial(self):
        """Test executing a single trial."""
        run_trial = self.create_mock_run_trial()
        executor = TrialExecutor(run_trial=run_trial)
        
        trials = [TrialSpec(config={"learning_rate": 0.001})]
        results = executor.execute(trials)
        
        assert len(results) == 1
        assert results[0].status == "success"
        assert results[0].metric == 0.85
    
    def test_execute_multiple_trials(self):
        """Test executing multiple trials."""
        run_trial = self.create_mock_run_trial()
        executor = TrialExecutor(run_trial=run_trial)
        
        trials = [
            TrialSpec(config={"learning_rate": 0.001}),
            TrialSpec(config={"learning_rate": 0.01}),
            TrialSpec(config={"learning_rate": 0.1})
        ]
        results = executor.execute(trials)
        
        assert len(results) == 3
        assert all(r.status == "success" for r in results)
    
    def test_execute_with_failure(self):
        """Test executing trials with failures."""
        # Create run_trial that fails on second trial
        call_count = [0]
        
        def run_trial(spec: TrialSpec) -> TrialResult:
            call_count[0] += 1
            trial_id = spec.ensure_id()
            
            if call_count[0] == 2:
                # Fail second trial
                return TrialResult(
                    trial_id=trial_id,
                    metric=None,
                    status="failed",
                    duration_seconds=1.0
                )
            else:
                return TrialResult(
                    trial_id=trial_id,
                    metric=0.85,
                    status="success",
                    duration_seconds=1.0
                )
        
        executor = TrialExecutor(run_trial=run_trial)
        
        trials = [
            TrialSpec(config={"learning_rate": 0.001}),
            TrialSpec(config={"learning_rate": 0.01}),
            TrialSpec(config={"learning_rate": 0.1})
        ]
        results = executor.execute(trials)
        
        assert len(results) == 3
        assert results[0].status == "success"
        assert results[1].status == "failed"
        assert results[2].status == "success"
    
    def test_executor_results_property(self):
        """Test that results property returns tuple."""
        run_trial = self.create_mock_run_trial()
        executor = TrialExecutor(run_trial=run_trial)
        
        trials = [TrialSpec(config={"learning_rate": 0.001})]
        executor.execute(trials)
        
        results = executor.results
        assert isinstance(results, tuple)
        assert len(results) == 1
    
    def test_executor_with_mlflow_client(self):
        """Test executor with MLflow client."""
        run_trial = self.create_mock_run_trial()
        mlflow_client = Mock()
        
        executor = TrialExecutor(
            run_trial=run_trial,
            mlflow_client=mlflow_client
        )
        
        trials = [TrialSpec(config={"learning_rate": 0.001})]
        executor.execute(trials)
        
        # Verify MLflow client is accessible
        assert executor.mlflow_client == mlflow_client


class TestProgressObservability:
    """Test suite for progress observability (FR-033)."""
    
    def test_progress_tracking(self):
        """Test that progress is tracked during execution."""
        run_trial_calls = []
        
        def run_trial(spec: TrialSpec) -> TrialResult:
            run_trial_calls.append(spec)
            trial_id = spec.ensure_id()
            return TrialResult(
                trial_id=trial_id,
                metric=0.85,
                status="success",
                duration_seconds=0.1
            )
        
        executor = TrialExecutor(run_trial=run_trial)
        
        trials = [
            TrialSpec(config={"learning_rate": 0.001}),
            TrialSpec(config={"learning_rate": 0.01}),
            TrialSpec(config={"learning_rate": 0.1})
        ]
        
        results = executor.execute(trials)
        
        # Verify all trials were executed
        assert len(run_trial_calls) == 3
        assert len(results) == 3
    
    def test_sequential_execution(self):
        """Test that trials are executed sequentially (FR-021)."""
        execution_order = []
        
        def run_trial(spec: TrialSpec) -> TrialResult:
            execution_order.append(spec.config["trial_num"])
            time.sleep(0.01)  # Small delay to ensure sequential
            trial_id = spec.ensure_id()
            return TrialResult(
                trial_id=trial_id,
                metric=0.85,
                status="success",
                duration_seconds=0.01
            )
        
        executor = TrialExecutor(run_trial=run_trial)
        
        trials = [
            TrialSpec(config={"trial_num": 1}),
            TrialSpec(config={"trial_num": 2}),
            TrialSpec(config={"trial_num": 3})
        ]
        
        executor.execute(trials)
        
        # Verify sequential execution
        assert execution_order == [1, 2, 3]


class TestGracefulFailureHandling:
    """Test suite for graceful failure handling (FR-021)."""
    
    def test_continue_after_failure(self):
        """Test that execution continues after trial failure."""
        call_count = [0]
        
        def run_trial(spec: TrialSpec) -> TrialResult:
            call_count[0] += 1
            trial_id = spec.ensure_id()
            
            # Fail every other trial
            if call_count[0] % 2 == 0:
                return TrialResult(
                    trial_id=trial_id,
                    metric=None,
                    status="failed",
                    duration_seconds=1.0
                )
            else:
                return TrialResult(
                    trial_id=trial_id,
                    metric=0.85,
                    status="success",
                    duration_seconds=1.0
                )
        
        executor = TrialExecutor(run_trial=run_trial)
        
        trials = [TrialSpec(config={}) for _ in range(5)]
        results = executor.execute(trials)
        
        # All trials should complete (some failed, some succeeded)
        assert len(results) == 5
        
        # Count successes and failures
        successes = sum(1 for r in results if r.status == "success")
        failures = sum(1 for r in results if r.status == "failed")
        
        assert successes == 3  # Trials 1, 3, 5
        assert failures == 2   # Trials 2, 4

