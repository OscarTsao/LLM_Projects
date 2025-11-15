"""Distributed HPO and parallel execution (Phase 12).

This module provides tools for running HPO across multiple processes,
GPUs, and nodes with efficient resource management.

Key Features:
- Parallel trial execution across multiple workers
- Multi-GPU coordination and resource allocation
- Distributed storage backend support
- Process pool management
- Trial queue coordination
- Resource monitoring

Example Usage:

    # Simple parallel execution
    from psy_agents_noaug.hpo.distributed import ParallelExecutor
    import optuna

    study = optuna.create_study(
        study_name="parallel-hpo",
        storage="postgresql://localhost/optuna",  # Shared storage
    )

    executor = ParallelExecutor(
        n_workers=4,
        gpu_ids=[0, 1, 2, 3],
    )

    executor.optimize(
        study,
        objective_func,
        n_trials=100,
    )

    # Multi-GPU with automatic allocation
    from psy_agents_noaug.hpo.distributed import GPUCoordinator

    coordinator = GPUCoordinator(gpu_ids=[0, 1, 2, 3])

    with coordinator.allocate_gpu() as gpu_id:
        # Train on allocated GPU
        device = f"cuda:{gpu_id}"
        model.to(device)

    # Advanced: Distributed storage
    from psy_agents_noaug.hpo.distributed import create_distributed_storage

    storage = create_distributed_storage(
        backend="postgresql",
        host="localhost",
        database="optuna_db",
    )

    study = optuna.create_study(storage=storage)
"""

# Parallel execution
from psy_agents_noaug.hpo.distributed.executor import (
    ParallelExecutor,
    WorkerConfig,
    ExecutionResult,
)

# GPU coordination
from psy_agents_noaug.hpo.distributed.coordinator import (
    GPUCoordinator,
    ResourceMonitor,
    GPUAllocation,
    check_gpu_availability,
)

# Storage backends
from psy_agents_noaug.hpo.distributed.storage import (
    create_distributed_storage,
    StorageConfig,
    check_storage_health,
    get_storage_recommendations,
)

__all__ = [
    # Execution
    "ParallelExecutor",
    "WorkerConfig",
    "ExecutionResult",
    # Coordination
    "GPUCoordinator",
    "ResourceMonitor",
    "GPUAllocation",
    "check_gpu_availability",
    # Storage
    "create_distributed_storage",
    "StorageConfig",
    "check_storage_health",
    "get_storage_recommendations",
]
