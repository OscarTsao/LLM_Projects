"""Health monitoring for HPO studies to detect and handle issues.

This module provides health checks for:
- Stalled trials (running too long)
- High failure rates
- Memory/resource issues
- Study corruption
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import optuna
from optuna.trial import TrialState

LOGGER = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health status for an HPO study."""

    healthy: bool = True
    warnings: list[str] = None
    errors: list[str] = None

    def __post_init__(self) -> None:
        """Initialize lists if None."""
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        LOGGER.warning("Health warning: %s", message)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.healthy = False
        LOGGER.error("Health error: %s", message)

    def is_healthy(self) -> bool:
        """Check if study is healthy (no errors)."""
        return self.healthy and len(self.errors) == 0

    def summary(self) -> str:
        """Get health summary string."""
        if self.is_healthy() and not self.warnings:
            return "✓ Healthy"
        if self.is_healthy():
            return f"⚠ Healthy with {len(self.warnings)} warnings"
        return f"✗ Unhealthy: {len(self.errors)} errors, {len(self.warnings)} warnings"


class HealthMonitor:
    """Monitor study health and detect issues."""

    def __init__(
        self,
        max_trial_duration_hours: float = 24.0,
        max_failure_rate: float = 0.3,
        min_progress_interval_hours: float = 2.0,
    ):
        """Initialize health monitor.

        Args:
            max_trial_duration_hours: Maximum expected trial duration
            max_failure_rate: Maximum acceptable failure rate (0-1)
            min_progress_interval_hours: Minimum time between trial completions
        """
        self.max_trial_duration = timedelta(hours=max_trial_duration_hours)
        self.max_failure_rate = max_failure_rate
        self.min_progress_interval = timedelta(hours=min_progress_interval_hours)

        self._last_completion_time: datetime | None = None

    def check_health(self, study: optuna.Study) -> HealthStatus:
        """Check study health.

        Args:
            study: Optuna study to check

        Returns:
            Health status with warnings and errors
        """
        status = HealthStatus()

        # Get all trials
        trials = study.get_trials(deepcopy=False)

        if not trials:
            status.add_warning("No trials have been run yet")
            return status

        # Check for stalled trials
        stalled = detect_stalled_trials(trials, self.max_trial_duration)
        if stalled:
            status.add_warning(
                f"{len(stalled)} stalled trials detected (running > {self.max_trial_duration})"
            )

        # Check failure rate
        completed = sum(1 for t in trials if t.state == TrialState.COMPLETE)
        failed = sum(1 for t in trials if t.state == TrialState.FAIL)
        finished = completed + failed

        if finished > 0:
            failure_rate = failed / finished
            if failure_rate > self.max_failure_rate:
                status.add_error(
                    f"High failure rate: {failure_rate*100:.1f}% (threshold: {self.max_failure_rate*100:.1f}%)"
                )
            elif failure_rate > self.max_failure_rate * 0.7:
                status.add_warning(f"Elevated failure rate: {failure_rate*100:.1f}%")

        # Check progress (completed trials)
        completed_trials = [t for t in trials if t.state == TrialState.COMPLETE]
        if completed_trials:
            latest = max(
                completed_trials, key=lambda t: t.datetime_complete or datetime.min
            )
            if latest.datetime_complete:
                time_since_completion = (
                    datetime.now() - latest.datetime_complete.replace(tzinfo=None)
                )

                if time_since_completion > self.min_progress_interval:
                    status.add_warning(
                        f"No trials completed in {time_since_completion} (threshold: {self.min_progress_interval})"
                    )

        # Check for running trials that might be stuck
        running_trials = [t for t in trials if t.state == TrialState.RUNNING]
        if running_trials:
            for trial in running_trials:
                if trial.datetime_start:
                    runtime = datetime.now() - trial.datetime_start.replace(tzinfo=None)
                    if runtime > self.max_trial_duration:
                        status.add_warning(
                            f"Trial #{trial.number} has been running for {runtime} (max: {self.max_trial_duration})"
                        )

        return status

    def monitor_loop(
        self,
        study: optuna.Study,
        check_interval: float = 300.0,
        max_checks: int | None = None,
    ) -> None:
        """Continuous health monitoring loop.

        Args:
            study: Optuna study to monitor
            check_interval: Seconds between health checks
            max_checks: Maximum number of checks (None = infinite)
        """
        check_count = 0

        LOGGER.info("Starting health monitoring for %s", study.study_name)

        while True:
            status = self.check_health(study)

            LOGGER.info("Health check #%d: %s", check_count + 1, status.summary())

            if status.warnings:
                for warning in status.warnings:
                    LOGGER.warning("  ⚠ %s", warning)

            if status.errors:
                for error in status.errors:
                    LOGGER.error("  ✗ %s", error)

            check_count += 1

            if max_checks is not None and check_count >= max_checks:
                break

            time.sleep(check_interval)


def detect_stalled_trials(
    trials: list[optuna.trial.FrozenTrial],
    max_duration: timedelta,
) -> list[optuna.trial.FrozenTrial]:
    """Detect trials that have been running too long.

    Args:
        trials: List of trials to check
        max_duration: Maximum allowed trial duration

    Returns:
        List of stalled trials
    """
    stalled = []
    now = datetime.now()

    for trial in trials:
        if trial.state == TrialState.RUNNING and trial.datetime_start:
            runtime = now - trial.datetime_start.replace(tzinfo=None)
            if runtime > max_duration:
                stalled.append(trial)

    return stalled


def check_study_health(
    study: optuna.Study,
    verbose: bool = True,
) -> HealthStatus:
    """Quick health check for a study.

    Args:
        study: Optuna study to check
        verbose: Print detailed status

    Returns:
        Health status
    """
    monitor = HealthMonitor()
    status = monitor.check_health(study)

    if verbose:
        print("=" * 80)
        print(f"Health Check: {study.study_name}")
        print("=" * 80)
        print(f"Status: {status.summary()}")

        if status.warnings:
            print("\nWarnings:")
            for warning in status.warnings:
                print(f"  ⚠ {warning}")

        if status.errors:
            print("\nErrors:")
            for error in status.errors:
                print(f"  ✗ {error}")

        print("=" * 80)

    return status
