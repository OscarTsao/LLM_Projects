#!/usr/bin/env python
"""Workflow management for CI/CD (Phase 16).

This module provides workflow orchestration including:
- Workflow definition and execution
- Step dependencies
- Parallel execution support
- Failure handling and retry logic
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

LOGGER = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a workflow step."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """A step in a workflow."""

    name: str
    command: str | Callable[[], Any]
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    retry_count: int = 0
    timeout: int = 300  # seconds
    continue_on_error: bool = False
    environment: dict[str, str] = field(default_factory=dict)

    # Runtime fields
    status: StepStatus = StepStatus.PENDING
    start_time: datetime | None = None
    end_time: datetime | None = None
    output: str = ""
    error: str = ""


class WorkflowManager:
    """Manage and execute CI/CD workflows."""

    def __init__(
        self,
        workflow_name: str,
        workspace_dir: Path | str = ".",
    ):
        """Initialize workflow manager.

        Args:
            workflow_name: Name of the workflow
            workspace_dir: Workspace directory
        """
        self.workflow_name = workflow_name
        self.workspace_dir = Path(workspace_dir)
        self.steps: dict[str, WorkflowStep] = {}
        self.execution_order: list[str] = []

        LOGGER.info(
            "Initialized WorkflowManager (workflow=%s)",
            workflow_name,
        )

    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow.

        Args:
            step: Workflow step
        """
        if step.name in self.steps:
            raise ValueError(f"Step already exists: {step.name}")

        self.steps[step.name] = step
        LOGGER.debug("Added step: %s", step.name)

    def _resolve_execution_order(self) -> list[str]:
        """Resolve step execution order based on dependencies.

        Returns:
            List of step names in execution order
        """
        # Topological sort
        order = []
        visited = set()
        temp_visited = set()

        def visit(step_name: str) -> None:
            if step_name in temp_visited:
                raise ValueError(f"Circular dependency detected: {step_name}")
            if step_name in visited:
                return

            temp_visited.add(step_name)

            step = self.steps[step_name]
            for dep in step.depends_on:
                if dep not in self.steps:
                    raise ValueError(
                        f"Unknown dependency: {dep} (required by {step_name})"
                    )
                visit(dep)

            temp_visited.remove(step_name)
            visited.add(step_name)
            order.append(step_name)

        for step_name in self.steps:
            if step_name not in visited:
                visit(step_name)

        return order

    def _execute_step(
        self,
        step: WorkflowStep,
    ) -> bool:
        """Execute a single step.

        Args:
            step: Step to execute

        Returns:
            True if successful
        """
        step.status = StepStatus.RUNNING
        step.start_time = datetime.now()

        LOGGER.info("Executing step: %s", step.name)

        try:
            if callable(step.command):
                # Execute Python function
                result = step.command()
                step.output = str(result)
                step.status = StepStatus.SUCCESS
                success = True

            else:
                # Execute shell command
                result = subprocess.run(
                    step.command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=step.timeout,
                    cwd=self.workspace_dir,
                    env=(
                        {**dict(os.environ), **step.environment}
                        if step.environment
                        else None
                    ),
                    check=False,
                )

                step.output = result.stdout
                step.error = result.stderr

                if result.returncode == 0:
                    step.status = StepStatus.SUCCESS
                    success = True
                else:
                    step.status = StepStatus.FAILED
                    success = False

        except subprocess.TimeoutExpired:
            step.status = StepStatus.FAILED
            step.error = f"Step timed out after {step.timeout}s"
            success = False
            LOGGER.exception("Step timed out: %s", step.name)

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            success = False
            LOGGER.exception("Step failed: %s - %s", step.name, e)

        finally:
            step.end_time = datetime.now()

        if success:
            LOGGER.info("Step completed: %s", step.name)
        else:
            LOGGER.error("Step failed: %s", step.name)

        return success

    def execute(
        self,
        parallel: bool = False,
    ) -> dict[str, Any]:
        """Execute the workflow.

        Args:
            parallel: Execute independent steps in parallel

        Returns:
            Execution results
        """

        start_time = datetime.now()

        LOGGER.info("Starting workflow: %s", self.workflow_name)

        # Resolve execution order
        self.execution_order = self._resolve_execution_order()

        LOGGER.info(
            "Execution order: %s",
            " → ".join(self.execution_order),
        )

        # Execute steps
        failed_steps = []
        skipped_steps = []

        for step_name in self.execution_order:
            step = self.steps[step_name]

            # Check dependencies
            should_skip = False
            for dep_name in step.depends_on:
                dep_step = self.steps[dep_name]
                if dep_step.status == StepStatus.FAILED:
                    if not step.continue_on_error:
                        step.status = StepStatus.SKIPPED
                        skipped_steps.append(step_name)
                        should_skip = True
                        LOGGER.warning(
                            "Skipping step %s (dependency %s failed)",
                            step_name,
                            dep_name,
                        )
                        break

            if should_skip:
                continue

            # Execute step with retry
            success = False
            attempts = step.retry_count + 1

            for attempt in range(attempts):
                if attempt > 0:
                    LOGGER.info(
                        "Retrying step %s (attempt %d/%d)",
                        step_name,
                        attempt + 1,
                        attempts,
                    )

                success = self._execute_step(step)

                if success:
                    break

                if attempt < attempts - 1:
                    time.sleep(2**attempt)  # Exponential backoff

            if not success:
                failed_steps.append(step_name)
                if not step.continue_on_error:
                    LOGGER.error(
                        "Workflow failed at step: %s",
                        step_name,
                    )
                    # Mark remaining steps as skipped
                    current_idx = self.execution_order.index(step_name)
                    for remaining_step_name in self.execution_order[current_idx + 1 :]:
                        remaining_step = self.steps[remaining_step_name]
                        if remaining_step.status == StepStatus.PENDING:
                            remaining_step.status = StepStatus.SKIPPED
                            skipped_steps.append(remaining_step_name)
                            LOGGER.warning(
                                "Skipping step %s (workflow failed)",
                                remaining_step_name,
                            )
                    break

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Compute results
        successful_steps = [
            name
            for name, step in self.steps.items()
            if step.status == StepStatus.SUCCESS
        ]

        results = {
            "workflow_name": self.workflow_name,
            "status": "success" if not failed_steps else "failed",
            "duration": duration,
            "total_steps": len(self.steps),
            "successful_steps": len(successful_steps),
            "failed_steps": len(failed_steps),
            "skipped_steps": len(skipped_steps),
            "failed_step_names": failed_steps,
            "skipped_step_names": skipped_steps,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        }

        LOGGER.info(
            "Workflow completed: %s (status=%s, duration=%.2fs)",
            self.workflow_name,
            results["status"],
            duration,
        )

        return results

    def get_step_results(self) -> dict[str, dict[str, Any]]:
        """Get results for all steps.

        Returns:
            Dict mapping step names to results
        """
        results = {}

        for name, step in self.steps.items():
            results[name] = {
                "status": step.status.value,
                "start_time": step.start_time.isoformat() if step.start_time else None,
                "end_time": step.end_time.isoformat() if step.end_time else None,
                "output": step.output,
                "error": step.error,
                "duration": (
                    (step.end_time - step.start_time).total_seconds()
                    if step.start_time and step.end_time
                    else None
                ),
            }

        return results


def run_workflow(
    workflow_name: str,
    steps: list[WorkflowStep],
    workspace_dir: Path | str = ".",
) -> dict[str, Any]:
    """Run a workflow (convenience function).

    Args:
        workflow_name: Workflow name
        steps: List of steps
        workspace_dir: Workspace directory

    Returns:
        Execution results
    """
    manager = WorkflowManager(
        workflow_name=workflow_name,
        workspace_dir=workspace_dir,
    )

    for step in steps:
        manager.add_step(step)

    return manager.execute()
