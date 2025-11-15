#!/usr/bin/env python
"""Test script for Phase 16: CI/CD Integration & Automated Workflows.

This script tests:
1. Workflow management and orchestration
2. Quality gate validation
3. Pipeline execution with artifacts
4. Error handling and edge cases
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from psy_agents_noaug.cicd import (
    Pipeline,
    PipelineStage,
    QualityGate,
    WorkflowManager,
    WorkflowStep,
    validate_quality_gates,
)
from psy_agents_noaug.cicd.gates import GateType
from psy_agents_noaug.cicd.workflow import StepStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def test_workflow_manager() -> bool:
    """Test workflow management and orchestration.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 1: Workflow Manager")
    LOGGER.info("=" * 80)

    try:
        # Create temporary workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize workflow
            workflow = WorkflowManager(
                workflow_name="test_workflow",
                workspace_dir=tmpdir,
            )

            # Define steps
            step1 = WorkflowStep(
                name="step1",
                command=lambda: "Step 1 executed",
                description="First step",
            )

            step2 = WorkflowStep(
                name="step2",
                command=lambda: "Step 2 executed",
                description="Second step (depends on step1)",
                depends_on=["step1"],
            )

            step3 = WorkflowStep(
                name="step3",
                command=lambda: "Step 3 executed",
                description="Third step (depends on step2)",
                depends_on=["step2"],
            )

            # Add steps
            workflow.add_step(step1)
            workflow.add_step(step2)
            workflow.add_step(step3)

            # Execute workflow
            LOGGER.info("Executing workflow with dependencies...")
            results = workflow.execute()

            # Verify results
            assert results["status"] == "success", "Workflow should succeed"
            assert results["total_steps"] == 3, "Should have 3 steps"
            assert results["successful_steps"] == 3, "All steps should succeed"
            assert results["failed_steps"] == 0, "No steps should fail"

            # Verify execution order
            assert workflow.execution_order == ["step1", "step2", "step3"]

            # Verify step statuses
            for step_name in ["step1", "step2", "step3"]:
                step = workflow.steps[step_name]
                assert step.status == StepStatus.SUCCESS
                assert step.start_time is not None
                assert step.end_time is not None

            LOGGER.info("✅ Workflow Manager: PASSED")
            LOGGER.info(f"   - Execution order: {workflow.execution_order}")
            LOGGER.info(f"   - Duration: {results['duration']:.2f}s")
            LOGGER.info(f"   - Status: {results['status']}")

            return True

    except Exception as e:
        LOGGER.exception(f"❌ Workflow Manager: FAILED - {e}")
        return False


def test_workflow_with_failure() -> bool:
    """Test workflow with failed step.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 2: Workflow with Failure")
    LOGGER.info("=" * 80)

    try:
        # Create temporary workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize workflow
            workflow = WorkflowManager(
                workflow_name="test_failure_workflow",
                workspace_dir=tmpdir,
            )

            # Define steps
            step1 = WorkflowStep(
                name="step1",
                command=lambda: "Step 1 executed",
                description="First step (succeeds)",
            )

            step2 = WorkflowStep(
                name="step2",
                command=lambda: (_ for _ in ()).throw(
                    ValueError("Intentional failure")
                ),
                description="Second step (fails)",
                depends_on=["step1"],
            )

            step3 = WorkflowStep(
                name="step3",
                command=lambda: "Step 3 executed",
                description="Third step (should be skipped)",
                depends_on=["step2"],
            )

            # Add steps
            workflow.add_step(step1)
            workflow.add_step(step2)
            workflow.add_step(step3)

            # Execute workflow
            LOGGER.info("Executing workflow with failing step...")
            results = workflow.execute()

            # Verify results
            assert results["status"] == "failed", "Workflow should fail"
            assert results["successful_steps"] == 1, "Only step1 should succeed"
            assert results["failed_steps"] == 1, "Step2 should fail"
            assert results["skipped_steps"] == 1, "Step3 should be skipped"

            # Verify step statuses
            assert workflow.steps["step1"].status == StepStatus.SUCCESS
            assert workflow.steps["step2"].status == StepStatus.FAILED
            assert workflow.steps["step3"].status == StepStatus.SKIPPED

            LOGGER.info("✅ Workflow with Failure: PASSED")
            LOGGER.info(f"   - Step1: {workflow.steps['step1'].status.value}")
            LOGGER.info(f"   - Step2: {workflow.steps['step2'].status.value}")
            LOGGER.info(f"   - Step3: {workflow.steps['step3'].status.value}")

            return True

    except Exception as e:
        LOGGER.exception(f"❌ Workflow with Failure: FAILED - {e}")
        return False


def test_quality_gates() -> bool:
    """Test quality gate validation.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 3: Quality Gates")
    LOGGER.info("=" * 80)

    try:
        # Define test metrics
        test_metrics = {
            "accuracy": 0.85,
            "coverage": 0.75,
            "latency": 100,  # ms
        }

        # Define quality gates
        gates = [
            QualityGate(
                name="accuracy_threshold",
                gate_type=GateType.METRIC,
                threshold=0.80,
                comparison=">=",
                value_getter=lambda: test_metrics["accuracy"],
                description="Accuracy must be >= 80%",
            ),
            QualityGate(
                name="coverage_threshold",
                gate_type=GateType.COVERAGE,
                threshold=0.80,
                comparison=">=",
                value_getter=lambda: test_metrics["coverage"],
                description="Coverage must be >= 80%",
                error_on_fail=False,  # Warning only
            ),
            QualityGate(
                name="latency_threshold",
                gate_type=GateType.PERFORMANCE,
                threshold=150,
                comparison="<=",
                value_getter=lambda: test_metrics["latency"],
                description="Latency must be <= 150ms",
            ),
        ]

        # Validate gates
        LOGGER.info("Validating quality gates...")
        results = validate_quality_gates(gates)

        # Verify results
        assert results["status"] == "passed", "Overall status should pass"
        assert results["passed"] == 2, "2 gates should pass"
        assert results["warnings"] == 1, "1 gate should warn"
        assert results["failed"] == 0, "0 gates should fail"

        # Check individual gates
        gate_results = {r["gate_name"]: r for r in results["results"]}

        assert gate_results["accuracy_threshold"]["status"] == "passed"
        assert gate_results["coverage_threshold"]["status"] == "warning"
        assert gate_results["latency_threshold"]["status"] == "passed"

        LOGGER.info("✅ Quality Gates: PASSED")
        for result in results["results"]:
            LOGGER.info(f"   - {result['gate_name']}: {result['status']}")
            LOGGER.info(f"     {result['message']}")

        return True

    except Exception as e:
        LOGGER.exception(f"❌ Quality Gates: FAILED - {e}")
        return False


def test_quality_gates_with_failures() -> bool:
    """Test quality gates with hard failures.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 4: Quality Gates with Failures")
    LOGGER.info("=" * 80)

    try:
        # Define test metrics
        test_metrics = {
            "accuracy": 0.70,  # Below threshold
            "f1_score": 0.60,  # Below threshold
        }

        # Define quality gates with failures
        gates = [
            QualityGate(
                name="accuracy_gate",
                gate_type=GateType.METRIC,
                threshold=0.80,
                comparison=">=",
                value_getter=lambda: test_metrics["accuracy"],
                error_on_fail=True,
            ),
            QualityGate(
                name="f1_gate",
                gate_type=GateType.METRIC,
                threshold=0.75,
                comparison=">=",
                value_getter=lambda: test_metrics["f1_score"],
                error_on_fail=True,
            ),
        ]

        # Validate gates
        LOGGER.info("Validating quality gates with failures...")
        results = validate_quality_gates(gates)

        # Verify results
        assert results["status"] == "failed", "Overall status should fail"
        assert results["passed"] == 0, "0 gates should pass"
        assert results["failed"] == 2, "2 gates should fail"

        LOGGER.info("✅ Quality Gates with Failures: PASSED")
        LOGGER.info(f"   - Overall status: {results['status']}")
        LOGGER.info(f"   - Failed gates: {results['failed']}")

        return True

    except Exception as e:
        LOGGER.exception(f"❌ Quality Gates with Failures: FAILED - {e}")
        return False


def test_pipeline_with_artifacts() -> bool:
    """Test pipeline with artifact passing.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 5: Pipeline with Artifacts")
    LOGGER.info("=" * 80)

    try:
        # Create temporary artifact directory
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"

            # Initialize pipeline
            pipeline = Pipeline(
                pipeline_name="test_pipeline",
                artifact_dir=artifact_dir,
            )

            # Define stage executors
            def stage1_executor(inputs: dict) -> dict:
                """Generate data."""
                return {
                    "data": [1, 2, 3, 4, 5],
                    "metadata": {"source": "stage1"},
                }

            def stage2_executor(inputs: dict) -> dict:
                """Process data from stage1."""
                data = inputs.get("stage1.data", [])
                processed = [x * 2 for x in data]
                return {
                    "processed_data": processed,
                    "count": len(processed),
                }

            def stage3_executor(inputs: dict) -> dict:
                """Aggregate results from stage2."""
                processed = inputs.get("stage2.processed_data", [])
                return {
                    "sum": sum(processed),
                    "mean": sum(processed) / len(processed) if processed else 0,
                }

            # Define stages
            stage1 = PipelineStage(
                name="stage1",
                executor=stage1_executor,
                description="Generate data",
                artifacts=["data", "metadata"],
            )

            stage2 = PipelineStage(
                name="stage2",
                executor=stage2_executor,
                description="Process data",
                depends_on=["stage1"],
                artifacts=["processed_data", "count"],
            )

            stage3 = PipelineStage(
                name="stage3",
                executor=stage3_executor,
                description="Aggregate results",
                depends_on=["stage2"],
                artifacts=["sum", "mean"],
            )

            # Add stages
            pipeline.add_stage(stage1)
            pipeline.add_stage(stage2)
            pipeline.add_stage(stage3)

            # Execute pipeline
            LOGGER.info("Executing pipeline with artifacts...")
            results = pipeline.execute()

            # Verify results
            assert results["status"] == "success", "Pipeline should succeed"
            assert results["successful_stages"] == 3, "All stages should succeed"

            # Verify artifacts were saved
            assert (artifact_dir / "stage1_data.json").exists()
            assert (artifact_dir / "stage1_metadata.json").exists()
            assert (artifact_dir / "stage2_processed_data.json").exists()
            assert (artifact_dir / "stage2_count.json").exists()
            assert (artifact_dir / "stage3_sum.json").exists()
            assert (artifact_dir / "stage3_mean.json").exists()

            # Verify artifact contents
            with (artifact_dir / "stage1_data.json").open() as f:
                data = json.load(f)
                assert data == [1, 2, 3, 4, 5]

            with (artifact_dir / "stage2_processed_data.json").open() as f:
                processed = json.load(f)
                assert processed == [2, 4, 6, 8, 10]

            with (artifact_dir / "stage3_sum.json").open() as f:
                total = json.load(f)
                assert total == 30

            LOGGER.info("✅ Pipeline with Artifacts: PASSED")
            LOGGER.info(f"   - Stages executed: {results['successful_stages']}")
            LOGGER.info(f"   - Duration: {results['duration']:.2f}s")
            LOGGER.info(
                f"   - Artifacts saved: {len(list(artifact_dir.glob('*.json')))}"
            )

            return True

    except Exception as e:
        LOGGER.exception(f"❌ Pipeline with Artifacts: FAILED - {e}")
        return False


def test_pipeline_with_stage_failure() -> bool:
    """Test pipeline with failed stage.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 6: Pipeline with Stage Failure")
    LOGGER.info("=" * 80)

    try:
        # Create temporary artifact directory
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"

            # Initialize pipeline
            pipeline = Pipeline(
                pipeline_name="test_failure_pipeline",
                artifact_dir=artifact_dir,
            )

            # Define stage executors
            def stage1_executor(inputs: dict) -> dict:
                return {"data": [1, 2, 3]}

            def stage2_executor(inputs: dict) -> dict:
                raise ValueError("Intentional stage failure")

            def stage3_executor(inputs: dict) -> dict:
                return {"result": "should not execute"}

            # Define stages
            stage1 = PipelineStage(
                name="stage1",
                executor=stage1_executor,
                description="First stage (succeeds)",
                artifacts=["data"],
            )

            stage2 = PipelineStage(
                name="stage2",
                executor=stage2_executor,
                description="Second stage (fails)",
                depends_on=["stage1"],
            )

            stage3 = PipelineStage(
                name="stage3",
                executor=stage3_executor,
                description="Third stage (should be skipped)",
                depends_on=["stage2"],
            )

            # Add stages
            pipeline.add_stage(stage1)
            pipeline.add_stage(stage2)
            pipeline.add_stage(stage3)

            # Execute pipeline
            LOGGER.info("Executing pipeline with failing stage...")
            results = pipeline.execute()

            # Verify results
            assert results["status"] == "failed", "Pipeline should fail"
            assert results["successful_stages"] == 1, "Only stage1 should succeed"
            assert results["failed_stages"] == 1, "Stage2 should fail"
            assert results["skipped_stages"] == 1, "Stage3 should be skipped"

            LOGGER.info("✅ Pipeline with Stage Failure: PASSED")
            LOGGER.info("   - Stage1: success")
            LOGGER.info("   - Stage2: failed")
            LOGGER.info("   - Stage3: skipped")

            return True

    except Exception as e:
        LOGGER.exception(f"❌ Pipeline with Stage Failure: FAILED - {e}")
        return False


def main():
    """Run all CI/CD tests."""
    LOGGER.info("Starting Phase 16 CI/CD Tests")
    LOGGER.info("=" * 80)

    tests = [
        ("Workflow Manager", test_workflow_manager),
        ("Workflow with Failure", test_workflow_with_failure),
        ("Quality Gates", test_quality_gates),
        ("Quality Gates with Failures", test_quality_gates_with_failures),
        ("Pipeline with Artifacts", test_pipeline_with_artifacts),
        ("Pipeline with Stage Failure", test_pipeline_with_stage_failure),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            LOGGER.exception(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("TEST SUMMARY")
    LOGGER.info("=" * 80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        LOGGER.info(f"{status}: {test_name}")

    LOGGER.info("=" * 80)
    LOGGER.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        LOGGER.info("🎉 All tests passed!")
        return 0
    LOGGER.error(f"❌ {total - passed} test(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
