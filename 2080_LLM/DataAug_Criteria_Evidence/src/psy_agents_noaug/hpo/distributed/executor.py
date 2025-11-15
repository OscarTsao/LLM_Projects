#!/usr/bin/env python
"""Parallel trial execution for distributed HPO (Phase 12).

This module provides tools for executing HPO trials in parallel across
multiple processes and GPUs with efficient resource management.

Key Features:
- Process pool management
- GPU allocation per worker
- Trial queue coordination
- Error handling and recovery
- Progress tracking
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import optuna
from optuna.trial import TrialState

LOGGER = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Configuration for a parallel worker."""

    worker_id: int
    gpu_id: int | None = None
    env_vars: dict[str, str] | None = None
    max_retries: int = 3


@dataclass
class ExecutionResult:
    """Result from parallel execution."""

    total_trials: int
    completed_trials: int
    failed_trials: int
    pruned_trials: int
    execution_time: float
    best_value: float | None = None
    worker_stats: dict[int, dict[str, Any]] | None = None


class ParallelExecutor:
    """Execute HPO trials in parallel across multiple workers."""

    def __init__(
        self,
        n_workers: int = 4,
        gpu_ids: list[int] | None = None,
        timeout: float | None = None,
    ):
        """Initialize parallel executor.

        Args:
            n_workers: Number of parallel workers
            gpu_ids: List of GPU IDs to use (None = CPU only)
            timeout: Timeout per trial in seconds (None = no timeout)
        """
        self.n_workers = n_workers
        self.gpu_ids = gpu_ids or []
        self.timeout = timeout

        # Validate GPU availability
        if self.gpu_ids:
            import torch

            n_gpus = torch.cuda.device_count()
            invalid_gpus = [g for g in self.gpu_ids if g >= n_gpus]
            if invalid_gpus:
                raise ValueError(
                    f"Invalid GPU IDs: {invalid_gpus} (available: 0-{n_gpus-1})"
                )

        LOGGER.info(
            "Initialized ParallelExecutor: %d workers, GPUs: %s",
            n_workers,
            gpu_ids if gpu_ids else "CPU only",
        )

    def optimize(
        self,
        study: optuna.Study,
        objective: Callable[[optuna.Trial], float],
        n_trials: int,
        show_progress_bar: bool = True,
    ) -> ExecutionResult:
        """Run parallel optimization.

        Args:
            study: Optuna study (must use shared storage)
            objective: Objective function
            n_trials: Total number of trials to run
            show_progress_bar: Show progress bar

        Returns:
            Execution result with statistics
        """
        LOGGER.info(
            "Starting parallel optimization: %d trials across %d workers",
            n_trials,
            self.n_workers,
        )

        # Check storage is suitable for parallel execution
        if not self._is_storage_shared(study):
            LOGGER.warning(
                "Study storage may not be shared across processes. "
                "Consider using RDB storage (PostgreSQL/MySQL) for parallel execution."
            )

        start_time = time.time()

        # Create worker configs
        worker_configs = self._create_worker_configs()

        # Split trials across workers
        trials_per_worker = n_trials // self.n_workers
        extra_trials = n_trials % self.n_workers

        # Create process pool
        with mp.Pool(processes=self.n_workers) as pool:
            # Launch workers
            worker_args = []
            for i, config in enumerate(worker_configs):
                n_worker_trials = trials_per_worker + (1 if i < extra_trials else 0)
                # Get storage URL - handle different storage types
                storage_obj = study._storage
                storage_url = None

                # Handle CachedStorage wrapper
                if hasattr(storage_obj, "_backend"):
                    backend = storage_obj._backend
                    if hasattr(backend, "url"):
                        storage_url = backend.url
                    elif hasattr(backend, "engine"):
                        storage_url = str(backend.engine.url)
                # Direct storage object
                elif hasattr(storage_obj, "url"):
                    storage_url = storage_obj.url
                elif hasattr(storage_obj, "engine"):
                    storage_url = str(storage_obj.engine.url)

                worker_args.append(
                    (
                        study.study_name,
                        storage_url,
                        objective,
                        n_worker_trials,
                        config,
                        show_progress_bar,
                    )
                )

            # Execute in parallel
            results = pool.starmap(self._worker_optimize, worker_args)

        execution_time = time.time() - start_time

        # Aggregate results
        total_completed = sum(r["completed"] for r in results)
        total_failed = sum(r["failed"] for r in results)
        total_pruned = sum(r["pruned"] for r in results)

        # Get best value
        best_trial = study.best_trial
        best_value = best_trial.value if best_trial else None

        # Worker statistics
        worker_stats = {
            i: {
                "completed": r["completed"],
                "failed": r["failed"],
                "pruned": r["pruned"],
                "avg_time": r["avg_time"],
            }
            for i, r in enumerate(results)
        }

        result = ExecutionResult(
            total_trials=n_trials,
            completed_trials=total_completed,
            failed_trials=total_failed,
            pruned_trials=total_pruned,
            execution_time=execution_time,
            best_value=best_value,
            worker_stats=worker_stats,
        )

        LOGGER.info(
            "Parallel optimization complete: %d trials in %.2f seconds",
            n_trials,
            execution_time,
        )
        LOGGER.info(
            "Results: %d completed, %d failed, %d pruned, best: %s",
            total_completed,
            total_failed,
            total_pruned,
            f"{best_value:.6f}" if best_value else "N/A",
        )

        return result

    def _create_worker_configs(self) -> list[WorkerConfig]:
        """Create worker configurations with GPU assignments.

        Returns:
            List of worker configs
        """
        configs = []
        for i in range(self.n_workers):
            # Assign GPU in round-robin fashion
            gpu_id = self.gpu_ids[i % len(self.gpu_ids)] if self.gpu_ids else None

            # Environment variables for this worker
            env_vars = {}
            if gpu_id is not None:
                env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            configs.append(
                WorkerConfig(
                    worker_id=i,
                    gpu_id=gpu_id,
                    env_vars=env_vars,
                )
            )

        return configs

    @staticmethod
    def _worker_optimize(
        study_name: str,
        storage: str | None,
        objective: Callable[[optuna.Trial], float],
        n_trials: int,
        config: WorkerConfig,
        show_progress_bar: bool,
    ) -> dict[str, Any]:
        """Worker function for parallel optimization.

        Args:
            study_name: Name of study
            storage: Storage URL
            objective: Objective function
            n_trials: Number of trials for this worker
            config: Worker configuration
            show_progress_bar: Show progress bar

        Returns:
            Worker statistics
        """
        # Set environment variables
        if config.env_vars:
            for key, value in config.env_vars.items():
                os.environ[key] = value

        # Load study
        if storage:
            study = optuna.load_study(study_name=study_name, storage=storage)
        else:
            # Fallback to in-memory storage (not recommended for parallel execution)
            raise ValueError(
                "No storage URL provided. Parallel execution requires shared storage."
            )

        LOGGER.info(
            "Worker %d: Starting %d trials (GPU: %s)",
            config.worker_id,
            n_trials,
            config.gpu_id if config.gpu_id is not None else "CPU",
        )

        # Track statistics
        completed = 0
        failed = 0
        pruned = 0
        trial_times = []

        start_time = time.time()

        # Run trials
        try:
            for _ in range(n_trials):
                trial_start = time.time()
                try:
                    study.optimize(
                        objective,
                        n_trials=1,
                        show_progress_bar=show_progress_bar and config.worker_id == 0,
                    )
                    trial_time = time.time() - trial_start
                    trial_times.append(trial_time)

                    # Check trial state
                    last_trial = study.trials[-1]
                    if last_trial.state == TrialState.COMPLETE:
                        completed += 1
                    elif last_trial.state == TrialState.PRUNED:
                        pruned += 1
                    else:
                        failed += 1

                except Exception as e:
                    LOGGER.error("Worker %d: Trial failed: %s", config.worker_id, e)
                    failed += 1

        except Exception as e:
            LOGGER.error("Worker %d: Fatal error: %s", config.worker_id, e)

        total_time = time.time() - start_time
        avg_time = sum(trial_times) / len(trial_times) if trial_times else 0.0

        LOGGER.info(
            "Worker %d: Completed %d/%d trials in %.2f seconds (avg: %.2f s/trial)",
            config.worker_id,
            completed,
            n_trials,
            total_time,
            avg_time,
        )

        return {
            "worker_id": config.worker_id,
            "completed": completed,
            "failed": failed,
            "pruned": pruned,
            "avg_time": avg_time,
            "total_time": total_time,
        }

    @staticmethod
    def _is_storage_shared(study: optuna.Study) -> bool:
        """Check if study storage is suitable for parallel execution.

        Args:
            study: Optuna study

        Returns:
            True if storage is shared (RDB-based)
        """
        # Access the underlying storage object
        storage = study._storage
        storage_str = str(storage)
        # Check for RDB backends
        rdb_backends = ["postgresql", "mysql", "sqlite"]
        return any(backend in storage_str.lower() for backend in rdb_backends)


class AsyncExecutor:
    """Execute HPO trials asynchronously with result collection."""

    def __init__(self, n_workers: int = 4):
        """Initialize async executor.

        Args:
            n_workers: Number of parallel workers
        """
        self.n_workers = n_workers
        self.pool = mp.Pool(processes=n_workers)
        self.pending_results: list[mp.pool.AsyncResult] = []

        LOGGER.info("Initialized AsyncExecutor with %d workers", n_workers)

    def submit(
        self,
        study: optuna.Study,
        objective: Callable[[optuna.Trial], float],
        n_trials: int = 1,
    ) -> mp.pool.AsyncResult:
        """Submit trials for async execution.

        Args:
            study: Optuna study
            objective: Objective function
            n_trials: Number of trials

        Returns:
            AsyncResult handle
        """
        result = self.pool.apply_async(
            self._execute_trials,
            args=(study.study_name, study.storage, objective, n_trials),
        )
        self.pending_results.append(result)
        return result

    def wait_all(self, timeout: float | None = None) -> list[dict[str, Any]]:
        """Wait for all pending results.

        Args:
            timeout: Timeout in seconds (None = wait forever)

        Returns:
            List of results
        """
        results = []
        for async_result in self.pending_results:
            try:
                result = async_result.get(timeout=timeout)
                results.append(result)
            except mp.TimeoutError:
                LOGGER.warning("Async result timed out")
                results.append({"error": "timeout"})

        self.pending_results.clear()
        return results

    def close(self) -> None:
        """Close the executor pool."""
        self.pool.close()
        self.pool.join()
        LOGGER.info("AsyncExecutor closed")

    @staticmethod
    def _execute_trials(
        study_name: str,
        storage: Any,
        objective: Callable[[optuna.Trial], float],
        n_trials: int,
    ) -> dict[str, Any]:
        """Execute trials (async worker function).

        Args:
            study_name: Study name
            storage: Storage object
            objective: Objective function
            n_trials: Number of trials

        Returns:
            Execution statistics
        """
        study = optuna.load_study(study_name=study_name, storage=storage)

        start_time = time.time()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        execution_time = time.time() - start_time

        return {
            "n_trials": n_trials,
            "execution_time": execution_time,
            "best_value": study.best_value,
        }


def recommend_parallel_config(
    n_trials: int,
    n_gpus: int,
    trial_duration: float = 60.0,
) -> dict[str, Any]:
    """Recommend parallel execution configuration.

    Args:
        n_trials: Total number of trials to run
        n_gpus: Number of available GPUs
        trial_duration: Estimated trial duration in seconds

    Returns:
        Recommended configuration

    Examples:
        >>> config = recommend_parallel_config(n_trials=100, n_gpus=4)
        >>> # Returns: {"n_workers": 4, "trials_per_worker": 25, ...}
    """
    if n_gpus == 0:
        # CPU-only: Use CPU cores
        n_workers = min(mp.cpu_count(), n_trials)
        gpu_ids = []
    else:
        # GPU: One worker per GPU
        n_workers = min(n_gpus, n_trials)
        gpu_ids = list(range(n_workers))

    trials_per_worker = n_trials // n_workers
    estimated_time = (trials_per_worker * trial_duration) / 3600  # hours

    return {
        "n_workers": n_workers,
        "gpu_ids": gpu_ids,
        "trials_per_worker": trials_per_worker,
        "estimated_time_hours": estimated_time,
        "recommendation": (
            f"Use {n_workers} workers with {trials_per_worker} trials each. "
            f"Estimated time: {estimated_time:.1f} hours."
        ),
    }
