#!/usr/bin/env python
"""GPU coordination and resource management (Phase 12).

This module provides tools for managing GPU resources across parallel
workers, preventing conflicts and optimizing utilization.

Key Features:
- GPU allocation and locking
- Resource monitoring (memory, utilization)
- Automatic GPU selection
- Load balancing across GPUs
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class GPUAllocation:
    """Information about GPU allocation."""

    gpu_id: int
    allocated_memory_mb: float
    utilization_percent: float
    process_id: int


class GPUCoordinator:
    """Coordinate GPU allocation across parallel workers."""

    def __init__(
        self,
        gpu_ids: list[int] | None = None,
        memory_fraction: float = 0.9,
    ):
        """Initialize GPU coordinator.

        Args:
            gpu_ids: List of GPU IDs to manage (None = auto-detect)
            memory_fraction: Maximum memory fraction to allocate per GPU
        """
        import torch

        self.gpu_ids = gpu_ids or list(range(torch.cuda.device_count()))
        self.memory_fraction = memory_fraction

        if not self.gpu_ids:
            raise ValueError("No GPUs available")

        # Shared state for GPU allocation
        self.manager = mp.Manager()
        self.gpu_locks = {gpu_id: mp.Lock() for gpu_id in self.gpu_ids}
        self.allocation_state = self.manager.dict()

        # Initialize allocation state
        for gpu_id in self.gpu_ids:
            self.allocation_state[gpu_id] = {
                "allocated": False,
                "process_id": None,
                "timestamp": None,
            }

        LOGGER.info(
            "Initialized GPUCoordinator with %d GPUs: %s",
            len(self.gpu_ids),
            self.gpu_ids,
        )

    @contextmanager
    def allocate_gpu(self, preferred_gpu: int | None = None):
        """Context manager for GPU allocation.

        Args:
            preferred_gpu: Preferred GPU ID (None = auto-select)

        Yields:
            Allocated GPU ID

        Example:
            >>> coordinator = GPUCoordinator([0, 1, 2, 3])
            >>> with coordinator.allocate_gpu() as gpu_id:
            >>>     device = f"cuda:{gpu_id}"
            >>>     model.to(device)
        """
        import os

        # Select GPU
        if preferred_gpu is not None:
            gpu_id = preferred_gpu
        else:
            gpu_id = self._select_best_gpu()

        # Acquire lock
        lock = self.gpu_locks[gpu_id]
        lock.acquire()

        try:
            # Mark as allocated
            self.allocation_state[gpu_id] = {
                "allocated": True,
                "process_id": os.getpid(),
                "timestamp": time.time(),
            }

            LOGGER.debug("Allocated GPU %d to process %d", gpu_id, os.getpid())

            # Set environment variable
            original_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            yield gpu_id

        finally:
            # Release GPU
            self.allocation_state[gpu_id] = {
                "allocated": False,
                "process_id": None,
                "timestamp": None,
            }

            # Restore environment
            if original_visible_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_visible_devices
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

            lock.release()
            LOGGER.debug("Released GPU %d", gpu_id)

    def _select_best_gpu(self) -> int:
        """Select GPU with most free memory.

        Returns:
            GPU ID
        """
        import torch

        best_gpu = None
        max_free_memory = -1

        for gpu_id in self.gpu_ids:
            # Check if already allocated
            if self.allocation_state[gpu_id]["allocated"]:
                continue

            # Get free memory
            try:
                torch.cuda.set_device(gpu_id)
                free_memory = torch.cuda.mem_get_info()[0] / 1024**3  # GB

                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_gpu = gpu_id

            except Exception as e:
                LOGGER.warning("Failed to query GPU %d: %s", gpu_id, e)

        if best_gpu is None:
            # All GPUs allocated, wait for one to free
            LOGGER.warning("All GPUs allocated, waiting for availability...")
            time.sleep(1.0)
            return self._select_best_gpu()

        LOGGER.debug(
            "Selected GPU %d (%.2f GB free)",
            best_gpu,
            max_free_memory,
        )
        return best_gpu

    def get_allocation_state(self) -> dict[int, dict[str, Any]]:
        """Get current GPU allocation state.

        Returns:
            Allocation state for each GPU
        """
        return dict(self.allocation_state)

    def get_gpu_stats(self) -> list[dict[str, Any]]:
        """Get GPU statistics.

        Returns:
            List of GPU stats
        """
        import torch

        stats = []
        for gpu_id in self.gpu_ids:
            try:
                torch.cuda.set_device(gpu_id)

                # Memory info
                total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
                allocated_memory = torch.cuda.memory_allocated(gpu_id)
                free_memory = total_memory - allocated_memory

                stats.append(
                    {
                        "gpu_id": gpu_id,
                        "name": torch.cuda.get_device_name(gpu_id),
                        "total_memory_gb": total_memory / 1024**3,
                        "allocated_memory_gb": allocated_memory / 1024**3,
                        "free_memory_gb": free_memory / 1024**3,
                        "allocated": self.allocation_state[gpu_id]["allocated"],
                        "process_id": self.allocation_state[gpu_id]["process_id"],
                    }
                )

            except Exception as e:
                LOGGER.warning("Failed to get stats for GPU %d: %s", gpu_id, e)

        return stats


class ResourceMonitor:
    """Monitor system resources during HPO."""

    def __init__(self, log_interval: float = 60.0):
        """Initialize resource monitor.

        Args:
            log_interval: Logging interval in seconds
        """
        self.log_interval = log_interval
        self.monitoring = False
        self.monitor_process = None

    def start(self) -> None:
        """Start monitoring."""
        if self.monitoring:
            LOGGER.warning("Monitor already running")
            return

        self.monitoring = True
        self.monitor_process = mp.Process(target=self._monitor_loop)
        self.monitor_process.start()
        LOGGER.info("Started resource monitor")

    def stop(self) -> None:
        """Stop monitoring."""
        if not self.monitoring:
            return

        self.monitoring = False
        if self.monitor_process:
            self.monitor_process.terminate()
            self.monitor_process.join()
        LOGGER.info("Stopped resource monitor")

    def _monitor_loop(self) -> None:
        """Monitoring loop."""
        import torch

        while self.monitoring:
            try:
                # Log GPU stats
                if torch.cuda.is_available():
                    for gpu_id in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                        LOGGER.info(
                            "GPU %d: Allocated %.2f GB, Reserved %.2f GB",
                            gpu_id,
                            allocated,
                            reserved,
                        )

                time.sleep(self.log_interval)

            except Exception as e:
                LOGGER.error("Monitor error: %s", e)


def allocate_gpus_round_robin(
    n_workers: int,
    gpu_ids: list[int],
) -> dict[int, int]:
    """Allocate GPUs to workers in round-robin fashion.

    Args:
        n_workers: Number of workers
        gpu_ids: Available GPU IDs

    Returns:
        Mapping from worker_id to gpu_id

    Examples:
        >>> allocation = allocate_gpus_round_robin(8, [0, 1, 2, 3])
        >>> # Returns: {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 3}
    """
    if not gpu_ids:
        raise ValueError("No GPUs provided")

    allocation = {}
    for worker_id in range(n_workers):
        gpu_id = gpu_ids[worker_id % len(gpu_ids)]
        allocation[worker_id] = gpu_id

    return allocation


def allocate_gpus_balanced(
    n_workers: int,
    gpu_ids: list[int],
    memory_usage: dict[int, float] | None = None,
) -> dict[int, int]:
    """Allocate GPUs to workers with load balancing.

    Distributes workers to minimize maximum load per GPU.

    Args:
        n_workers: Number of workers
        gpu_ids: Available GPU IDs
        memory_usage: Current memory usage per GPU (GB)

    Returns:
        Mapping from worker_id to gpu_id

    Examples:
        >>> allocation = allocate_gpus_balanced(10, [0, 1, 2, 3])
        >>> # Distributes 10 workers: 3-3-2-2 or similar
    """
    if not gpu_ids:
        raise ValueError("No GPUs provided")

    # Initialize load counters
    if memory_usage is None:
        gpu_loads = {gpu_id: 0 for gpu_id in gpu_ids}
    else:
        gpu_loads = {gpu_id: memory_usage.get(gpu_id, 0.0) for gpu_id in gpu_ids}

    allocation = {}
    for worker_id in range(n_workers):
        # Assign to GPU with minimum load
        min_load_gpu = min(gpu_loads, key=gpu_loads.get)  # type: ignore
        allocation[worker_id] = min_load_gpu
        gpu_loads[min_load_gpu] += 1  # Increment load

    return allocation


def get_optimal_batch_size(
    model_size_mb: float,
    gpu_memory_gb: float,
    memory_fraction: float = 0.8,
) -> int:
    """Estimate optimal batch size given model and GPU memory.

    Args:
        model_size_mb: Model size in MB
        gpu_memory_gb: GPU memory in GB
        memory_fraction: Fraction of memory to use

    Returns:
        Recommended batch size

    Examples:
        >>> batch_size = get_optimal_batch_size(500, 24, 0.8)
        >>> # Returns: Estimated batch size for 500MB model on 24GB GPU
    """
    available_memory_gb = gpu_memory_gb * memory_fraction
    available_memory_mb = available_memory_gb * 1024

    # Rough estimate: model + activations + gradients ≈ 3x model size per sample
    memory_per_sample = model_size_mb * 3

    batch_size = int(available_memory_mb / memory_per_sample)

    # Clamp to reasonable range
    batch_size = max(1, min(batch_size, 512))

    return batch_size


def check_gpu_availability() -> dict[str, Any]:
    """Check GPU availability and properties.

    Returns:
        GPU information

    Examples:
        >>> info = check_gpu_availability()
        >>> print(f"Available GPUs: {info['n_gpus']}")
    """
    try:
        import torch

        n_gpus = torch.cuda.device_count()
        gpu_info = []

        for gpu_id in range(n_gpus):
            props = torch.cuda.get_device_properties(gpu_id)
            gpu_info.append(
                {
                    "gpu_id": gpu_id,
                    "name": torch.cuda.get_device_name(gpu_id),
                    "total_memory_gb": props.total_memory / 1024**3,
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )

        return {
            "available": n_gpus > 0,
            "n_gpus": n_gpus,
            "gpu_info": gpu_info,
        }

    except Exception as e:
        return {
            "available": False,
            "n_gpus": 0,
            "error": str(e),
        }
