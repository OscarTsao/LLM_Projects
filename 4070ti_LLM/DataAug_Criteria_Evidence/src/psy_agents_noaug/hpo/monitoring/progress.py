"""Real-time progress tracking and visualization for HPO studies.

This module provides live monitoring of HPO progress with:
- Real-time trial completion tracking
- ETA estimation based on trial duration
- Best score tracking and improvement detection
- Failure rate monitoring
- Console and file logging
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import optuna
from optuna.trial import TrialState

LOGGER = logging.getLogger(__name__)


@dataclass
class ProgressStats:
    """Statistics for HPO progress tracking."""

    total_trials: int
    completed: int = 0
    running: int = 0
    pruned: int = 0
    failed: int = 0
    waiting: int = 0

    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)

    best_value: float | None = None
    best_trial: int | None = None
    improvements: int = 0

    trial_durations: deque = field(default_factory=lambda: deque(maxlen=50))

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time since start in seconds."""
        return time.time() - self.start_time

    @property
    def elapsed_str(self) -> str:
        """Formatted elapsed time string."""
        return str(timedelta(seconds=int(self.elapsed_seconds)))

    @property
    def progress_pct(self) -> float:
        """Progress percentage (0-100)."""
        finished = self.completed + self.pruned + self.failed
        return (finished / self.total_trials * 100) if self.total_trials > 0 else 0.0

    @property
    def avg_trial_duration(self) -> float:
        """Average trial duration in seconds."""
        return (
            sum(self.trial_durations) / len(self.trial_durations)
            if self.trial_durations
            else 0.0
        )

    @property
    def eta_seconds(self) -> float | None:
        """Estimated time to completion in seconds."""
        if not self.trial_durations:
            return None
        remaining = self.total_trials - (self.completed + self.pruned + self.failed)
        return remaining * self.avg_trial_duration if remaining > 0 else 0.0

    @property
    def eta_str(self) -> str:
        """Formatted ETA string."""
        if self.eta_seconds is None:
            return "calculating..."
        return str(timedelta(seconds=int(self.eta_seconds)))

    @property
    def failure_rate(self) -> float:
        """Failure rate (0-1)."""
        finished = self.completed + self.pruned + self.failed
        return self.failed / finished if finished > 0 else 0.0

    @property
    def prune_rate(self) -> float:
        """Prune rate (0-1)."""
        finished = self.completed + self.pruned + self.failed
        return self.pruned / finished if finished > 0 else 0.0


class ProgressTracker:
    """Real-time progress tracker for HPO studies."""

    def __init__(
        self,
        study: optuna.Study,
        total_trials: int,
        update_interval: float = 5.0,
        log_file: Path | None = None,
    ):
        """Initialize progress tracker.

        Args:
            study: Optuna study to monitor
            total_trials: Total number of trials to run
            update_interval: Seconds between progress updates
            log_file: Optional file path for progress logging
        """
        self.study = study
        self.stats = ProgressStats(total_trials=total_trials)
        self.update_interval = update_interval
        self.log_file = log_file

        self._last_print = 0.0
        self._trial_start_times: dict[int, float] = {}

    def update(self) -> None:
        """Update progress statistics from study."""
        # Get trial counts by state
        trials = self.study.get_trials(deepcopy=False)

        self.stats.completed = sum(1 for t in trials if t.state == TrialState.COMPLETE)
        self.stats.running = sum(1 for t in trials if t.state == TrialState.RUNNING)
        self.stats.pruned = sum(1 for t in trials if t.state == TrialState.PRUNED)
        self.stats.failed = sum(1 for t in trials if t.state == TrialState.FAIL)
        self.stats.waiting = sum(1 for t in trials if t.state == TrialState.WAITING)

        # Track best value
        if self.study.direction == optuna.study.StudyDirection.MAXIMIZE:
            if self.stats.best_value is None or (
                self.study.best_value is not None
                and self.study.best_value > self.stats.best_value
            ):
                self.stats.best_value = self.study.best_value
                self.stats.best_trial = (
                    self.study.best_trial.number if self.study.best_trial else None
                )
                self.stats.improvements += 1
        elif self.stats.best_value is None or (
            self.study.best_value is not None
            and self.study.best_value < self.stats.best_value
        ):
            self.stats.best_value = self.study.best_value
            self.stats.best_trial = (
                self.study.best_trial.number if self.study.best_trial else None
            )
            self.stats.improvements += 1

        # Track trial durations (only completed/pruned trials)
        for trial in trials:
            if trial.state in (TrialState.COMPLETE, TrialState.PRUNED):
                if trial.number not in self._trial_start_times and trial.datetime_start:
                    # Estimate duration from datetime
                    if trial.datetime_complete:
                        duration = (
                            trial.datetime_complete - trial.datetime_start
                        ).total_seconds()
                        self.stats.trial_durations.append(duration)
                        self._trial_start_times[trial.number] = time.time()

        self.stats.last_update = time.time()

    def print_progress(self, force: bool = False) -> None:
        """Print progress to console.

        Args:
            force: Force print even if update_interval hasn't passed
        """
        now = time.time()
        if not force and (now - self._last_print) < self.update_interval:
            return

        self.update()

        # Build progress message
        msg_lines = [
            "=" * 80,
            f"HPO Progress: {self.study.study_name}",
            "=" * 80,
            f"Progress: {self.stats.progress_pct:5.1f}% "
            f"({self.stats.completed + self.stats.pruned + self.stats.failed}/{self.stats.total_trials} trials)",
            f"  - Completed: {self.stats.completed:4d}",
            f"  - Pruned:    {self.stats.pruned:4d} ({self.stats.prune_rate*100:5.1f}%)",
            f"  - Failed:    {self.stats.failed:4d} ({self.stats.failure_rate*100:5.1f}%)",
            f"  - Running:   {self.stats.running:4d}",
            f"  - Waiting:   {self.stats.waiting:4d}",
            "",
            f"Elapsed:  {self.stats.elapsed_str}",
            f"ETA:      {self.stats.eta_str}",
            f"Avg/trial: {self.stats.avg_trial_duration:.1f}s",
            "",
        ]

        if self.stats.best_value is not None:
            msg_lines.extend(
                [
                    f"Best value: {self.stats.best_value:.6f} (trial #{self.stats.best_trial})",
                    f"Improvements: {self.stats.improvements}",
                ]
            )

        msg_lines.append("=" * 80)

        message = "\n".join(msg_lines)

        # Print to console
        print(message)

        # Log to file
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open("a") as f:
                f.write(f"\n[{datetime.now().isoformat()}]\n")
                f.write(message + "\n")

        self._last_print = now

    def on_trial_complete(self, trial_number: int, duration: float) -> None:
        """Callback when trial completes.

        Args:
            trial_number: Trial number that completed
            duration: Trial duration in seconds
        """
        self.stats.trial_durations.append(duration)
        self._trial_start_times[trial_number] = time.time()
        self.print_progress()

    def finalize(self) -> None:
        """Print final progress report."""
        self.update()
        self.print_progress(force=True)

        LOGGER.info("=" * 80)
        LOGGER.info("HPO COMPLETE: %s", self.study.study_name)
        LOGGER.info("=" * 80)
        LOGGER.info("Total trials: %d", self.stats.total_trials)
        LOGGER.info("  - Completed: %d", self.stats.completed)
        LOGGER.info(
            "  - Pruned: %d (%.1f%%)", self.stats.pruned, self.stats.prune_rate * 100
        )
        LOGGER.info(
            "  - Failed: %d (%.1f%%)", self.stats.failed, self.stats.failure_rate * 100
        )
        LOGGER.info("Total time: %s", self.stats.elapsed_str)
        LOGGER.info("Avg trial: %.1fs", self.stats.avg_trial_duration)

        if self.stats.best_value is not None:
            LOGGER.info(
                "Best value: %.6f (trial #%d)",
                self.stats.best_value,
                self.stats.best_trial,
            )
            LOGGER.info("Improvements: %d", self.stats.improvements)


class StudyMonitor:
    """High-level study monitor with automatic updates."""

    def __init__(
        self,
        study: optuna.Study,
        total_trials: int,
        update_interval: float = 10.0,
        log_dir: Path | None = None,
    ):
        """Initialize study monitor.

        Args:
            study: Optuna study to monitor
            total_trials: Total number of trials
            update_interval: Seconds between updates
            log_dir: Directory for progress logs
        """
        self.study = study
        self.total_trials = total_trials
        self.update_interval = update_interval

        # Setup log file
        if log_dir:
            log_dir = Path(log_dir)
            log_file = log_dir / f"{study.study_name}_progress.log"
        else:
            log_file = None

        self.tracker = ProgressTracker(
            study=study,
            total_trials=total_trials,
            update_interval=update_interval,
            log_file=log_file,
        )

        self._last_check = 0.0

    def check_and_update(self) -> None:
        """Check for updates and print progress if needed."""
        now = time.time()
        if (now - self._last_check) >= self.update_interval:
            self.tracker.print_progress()
            self._last_check = now

    def finalize(self) -> None:
        """Finalize monitoring and print final report."""
        self.tracker.finalize()


def create_progress_callback(
    tracker: ProgressTracker,
) -> optuna.study.StudyCallback:
    """Create Optuna callback for progress tracking.

    Args:
        tracker: Progress tracker instance

    Returns:
        Optuna callback function
    """

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Callback invoked after each trial."""
        if trial.state in (TrialState.COMPLETE, TrialState.PRUNED):
            if trial.datetime_start and trial.datetime_complete:
                duration = (
                    trial.datetime_complete - trial.datetime_start
                ).total_seconds()
                tracker.on_trial_complete(trial.number, duration)

    return callback
