#!/usr/bin/env python
"""Pipeline orchestration for CI/CD (Phase 16).

This module provides pipeline orchestration including:
- Multi-stage pipelines
- Stage dependencies
- Artifact passing between stages
- Deployment pipelines
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

LOGGER = logging.getLogger(__name__)


class StageStatus(Enum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStage:
    """A stage in a CI/CD pipeline."""

    name: str
    executor: Callable[[dict[str, Any]], dict[str, Any]]
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)

    # Runtime fields
    status: StageStatus = StageStatus.PENDING
    start_time: datetime | None = None
    end_time: datetime | None = None
    output: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class Pipeline:
    """Orchestrate multi-stage CI/CD pipelines."""

    def __init__(
        self,
        pipeline_name: str,
        artifact_dir: Path | str = "artifacts",
    ):
        """Initialize pipeline.

        Args:
            pipeline_name: Pipeline name
            artifact_dir: Directory for artifacts
        """
        self.pipeline_name = pipeline_name
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.stages: dict[str, PipelineStage] = {}
        self.shared_context: dict[str, Any] = {}

        LOGGER.info(
            "Initialized Pipeline (name=%s, artifact_dir=%s)",
            pipeline_name,
            self.artifact_dir,
        )

    def add_stage(self, stage: PipelineStage) -> None:
        """Add a stage to the pipeline.

        Args:
            stage: Pipeline stage
        """
        if stage.name in self.stages:
            raise ValueError(f"Stage already exists: {stage.name}")

        self.stages[stage.name] = stage
        LOGGER.debug("Added stage: %s", stage.name)

    def _resolve_stage_order(self) -> list[str]:
        """Resolve stage execution order.

        Returns:
            List of stage names in order
        """
        order = []
        visited = set()

        def visit(stage_name: str) -> None:
            if stage_name in visited:
                return

            stage = self.stages[stage_name]
            for dep in stage.depends_on:
                if dep not in self.stages:
                    raise ValueError(f"Unknown dependency: {dep}")
                visit(dep)

            visited.add(stage_name)
            order.append(stage_name)

        for stage_name in self.stages:
            visit(stage_name)

        return order

    def _save_artifacts(
        self,
        stage_name: str,
        artifacts: dict[str, Any],
    ) -> None:
        """Save stage artifacts.

        Args:
            stage_name: Stage name
            artifacts: Artifacts to save
        """
        for artifact_name, artifact_data in artifacts.items():
            artifact_path = self.artifact_dir / f"{stage_name}_{artifact_name}.json"

            with artifact_path.open("w") as f:
                json.dump(artifact_data, f, indent=2, default=str)

            LOGGER.debug("Saved artifact: %s", artifact_path)

    def _load_artifacts(
        self,
        stage_name: str,
    ) -> dict[str, Any]:
        """Load artifacts from dependencies.

        Args:
            stage_name: Stage name

        Returns:
            Combined artifacts from dependencies
        """
        stage = self.stages[stage_name]
        artifacts = {}

        for dep_name in stage.depends_on:
            dep_stage = self.stages[dep_name]

            for artifact_name in dep_stage.artifacts:
                artifact_path = self.artifact_dir / f"{dep_name}_{artifact_name}.json"

                if artifact_path.exists():
                    with artifact_path.open() as f:
                        artifacts[f"{dep_name}.{artifact_name}"] = json.load(f)

        return artifacts

    def _execute_stage(
        self,
        stage: PipelineStage,
    ) -> bool:
        """Execute a single stage.

        Args:
            stage: Stage to execute

        Returns:
            True if successful
        """
        stage.status = StageStatus.RUNNING
        stage.start_time = datetime.now()

        LOGGER.info("Executing stage: %s", stage.name)

        try:
            # Load artifacts from dependencies
            input_artifacts = self._load_artifacts(stage.name)

            # Execute stage
            stage.output = stage.executor(input_artifacts)

            # Save artifacts
            if stage.artifacts and stage.output:
                artifacts_to_save = {
                    name: stage.output.get(name)
                    for name in stage.artifacts
                    if name in stage.output
                }
                self._save_artifacts(stage.name, artifacts_to_save)

            stage.status = StageStatus.SUCCESS
            success = True

        except Exception as e:
            stage.status = StageStatus.FAILED
            stage.error = str(e)
            success = False
            LOGGER.exception("Stage failed: %s - %s", stage.name, e)

        finally:
            stage.end_time = datetime.now()

        if success:
            LOGGER.info("Stage completed: %s", stage.name)

        return success

    def execute(self) -> dict[str, Any]:
        """Execute the pipeline.

        Returns:
            Execution results
        """
        start_time = datetime.now()

        LOGGER.info("Starting pipeline: %s", self.pipeline_name)

        # Resolve execution order
        stage_order = self._resolve_stage_order()

        LOGGER.info("Stage order: %s", " → ".join(stage_order))

        # Execute stages
        failed_stages = []
        skipped_stages = []

        for stage_name in stage_order:
            stage = self.stages[stage_name]

            # Check dependencies
            should_skip = False
            for dep_name in stage.depends_on:
                dep_stage = self.stages[dep_name]
                if dep_stage.status == StageStatus.FAILED:
                    stage.status = StageStatus.SKIPPED
                    skipped_stages.append(stage_name)
                    should_skip = True
                    LOGGER.warning(
                        "Skipping stage %s (dependency %s failed)",
                        stage_name,
                        dep_name,
                    )
                    break

            if should_skip:
                continue

            # Execute stage
            success = self._execute_stage(stage)

            if not success:
                failed_stages.append(stage_name)
                # Mark remaining stages as skipped
                current_idx = stage_order.index(stage_name)
                for remaining_stage_name in stage_order[current_idx + 1 :]:
                    remaining_stage = self.stages[remaining_stage_name]
                    if remaining_stage.status == StageStatus.PENDING:
                        remaining_stage.status = StageStatus.SKIPPED
                        skipped_stages.append(remaining_stage_name)
                        LOGGER.warning(
                            "Skipping stage %s (pipeline failed)",
                            remaining_stage_name,
                        )
                break

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Compute results
        successful_stages = [
            name
            for name, stage in self.stages.items()
            if stage.status == StageStatus.SUCCESS
        ]

        results = {
            "pipeline_name": self.pipeline_name,
            "status": "success" if not failed_stages else "failed",
            "duration": duration,
            "total_stages": len(self.stages),
            "successful_stages": len(successful_stages),
            "failed_stages": len(failed_stages),
            "skipped_stages": len(skipped_stages),
            "failed_stage_names": failed_stages,
            "skipped_stage_names": skipped_stages,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        }

        LOGGER.info(
            "Pipeline completed: %s (status=%s, duration=%.2fs)",
            self.pipeline_name,
            results["status"],
            duration,
        )

        return results


def run_pipeline(
    pipeline_name: str,
    stages: list[PipelineStage],
    artifact_dir: Path | str = "artifacts",
) -> dict[str, Any]:
    """Run a pipeline (convenience function).

    Args:
        pipeline_name: Pipeline name
        stages: List of stages
        artifact_dir: Artifact directory

    Returns:
        Execution results
    """
    pipeline = Pipeline(
        pipeline_name=pipeline_name,
        artifact_dir=artifact_dir,
    )

    for stage in stages:
        pipeline.add_stage(stage)

    return pipeline.execute()
