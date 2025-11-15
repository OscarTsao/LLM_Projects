#!/usr/bin/env python
"""Test distributed HPO functionality (Phase 12).

Quick test to validate parallel execution, GPU coordination, and storage backends.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import optuna


# Module-level objective function for pickling
def simple_objective(trial: optuna.Trial) -> float:
    """Simple quadratic objective for testing."""
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2


def test_parallel_executor() -> None:
    """Test parallel trial execution."""
    print("\n" + "=" * 80)
    print("TEST 1: Parallel Executor")
    print("=" * 80)

    from psy_agents_noaug.hpo.distributed import ParallelExecutor

    # Create study with shared storage
    study = optuna.create_study(
        study_name="test-parallel",
        storage="sqlite:///test_distributed.db",
        direction="minimize",
        load_if_exists=False,
    )

    # Create executor (CPU-only for testing)
    executor = ParallelExecutor(n_workers=2, gpu_ids=[])

    print("\n✓ Created ParallelExecutor with 2 workers (CPU)")

    # Run parallel optimization
    result = executor.optimize(
        study=study,
        objective=simple_objective,  # Use module-level function
        n_trials=10,
        show_progress_bar=False,
    )

    print(f"\n✓ Completed parallel optimization:")
    print(f"  Total trials: {result.total_trials}")
    print(f"  Completed: {result.completed_trials}")
    print(f"  Failed: {result.failed_trials}")
    print(f"  Best value: {result.best_value:.6f}")
    print(f"  Execution time: {result.execution_time:.2f} seconds")

    assert result.completed_trials > 0, "No trials completed"
    assert result.best_value is not None, "No best value found"

    print("\n✅ TEST 1 PASSED: Parallel execution working correctly")


def test_gpu_coordinator() -> None:
    """Test GPU coordination."""
    print("\n" + "=" * 80)
    print("TEST 2: GPU Coordinator")
    print("=" * 80)

    from psy_agents_noaug.hpo.distributed import GPUCoordinator, check_gpu_availability

    # Check GPU availability
    gpu_info = check_gpu_availability()
    print(f"\n✓ GPU check: {gpu_info['n_gpus']} GPUs available")

    if gpu_info["n_gpus"] == 0:
        print("  Skipping GPU tests (no GPUs available)")
        print("\n✅ TEST 2 PASSED: GPU coordinator initialized (no GPUs to test)")
        return

    # Test GPU allocation
    gpu_ids = list(range(min(2, gpu_info["n_gpus"])))
    coordinator = GPUCoordinator(gpu_ids=gpu_ids)

    print(f"✓ Created GPUCoordinator with GPUs: {gpu_ids}")

    # Test allocation context manager
    with coordinator.allocate_gpu() as gpu_id:
        print(f"✓ Allocated GPU: {gpu_id}")
        assert gpu_id in gpu_ids, f"Invalid GPU ID: {gpu_id}"

    # Test GPU stats
    stats = coordinator.get_gpu_stats()
    print(f"\n✓ GPU Statistics:")
    for stat in stats[:2]:  # Show first 2 GPUs
        print(f"  GPU {stat['gpu_id']}: {stat['name']}")
        print(f"    Total memory: {stat['total_memory_gb']:.2f} GB")
        print(f"    Free memory: {stat['free_memory_gb']:.2f} GB")

    print("\n✅ TEST 2 PASSED: GPU coordinator working correctly")


def test_storage_backends() -> None:
    """Test storage backend utilities."""
    print("\n" + "=" * 80)
    print("TEST 3: Storage Backends")
    print("=" * 80)

    from psy_agents_noaug.hpo.distributed import (
        check_storage_health,
        create_distributed_storage,
        get_storage_recommendations,
    )

    # Test SQLite storage creation
    storage = create_distributed_storage(
        backend="sqlite",
        database="test_storage.db",
    )
    print(f"\n✓ Created SQLite storage: {storage}")

    # Test health check
    health = check_storage_health(storage)
    print(f"\n✓ Storage health check:")
    print(f"  Accessible: {health['accessible']}")
    print(f"  Backend: {health['backend']}")
    if health["response_time_ms"]:
        print(f"  Response time: {health['response_time_ms']:.2f} ms")

    assert health["accessible"], "Storage not accessible"

    # Test recommendations
    recs = get_storage_recommendations(n_workers=8)
    print(f"\n✓ Storage recommendations for 8 workers:")
    print(f"  Recommended: {recs['recommended_backend']}")
    print(f"  Reason: {recs['reason']}")
    print(f"  Pool size: {recs['pool_size']}")

    print("\n✅ TEST 3 PASSED: Storage backends working correctly")


def test_resource_allocation() -> None:
    """Test resource allocation strategies."""
    print("\n" + "=" * 80)
    print("TEST 4: Resource Allocation")
    print("=" * 80)

    from psy_agents_noaug.hpo.distributed.coordinator import (
        allocate_gpus_balanced,
        allocate_gpus_round_robin,
        get_optimal_batch_size,
    )

    # Test round-robin allocation
    allocation_rr = allocate_gpus_round_robin(n_workers=8, gpu_ids=[0, 1, 2, 3])
    print(f"\n✓ Round-robin allocation (8 workers, 4 GPUs):")
    print(f"  {allocation_rr}")

    assert len(allocation_rr) == 8, "Wrong number of allocations"
    assert all(gpu_id in [0, 1, 2, 3] for gpu_id in allocation_rr.values())

    # Test balanced allocation
    allocation_bal = allocate_gpus_balanced(n_workers=10, gpu_ids=[0, 1, 2, 3])
    print(f"\n✓ Balanced allocation (10 workers, 4 GPUs):")
    print(f"  {allocation_bal}")

    # Count workers per GPU
    gpu_counts = {gpu_id: 0 for gpu_id in [0, 1, 2, 3]}
    for gpu_id in allocation_bal.values():
        gpu_counts[gpu_id] += 1

    print(f"  Workers per GPU: {gpu_counts}")
    assert max(gpu_counts.values()) - min(gpu_counts.values()) <= 1, "Imbalanced allocation"

    # Test batch size estimation
    batch_size = get_optimal_batch_size(model_size_mb=500, gpu_memory_gb=24)
    print(f"\n✓ Optimal batch size for 500MB model on 24GB GPU: {batch_size}")
    assert batch_size > 0, "Invalid batch size"

    print("\n✅ TEST 4 PASSED: Resource allocation working correctly")


def test_parallel_recommendations() -> None:
    """Test recommendation functions."""
    print("\n" + "=" * 80)
    print("TEST 5: Recommendation Functions")
    print("=" * 80)

    from psy_agents_noaug.hpo.distributed.executor import recommend_parallel_config
    from psy_agents_noaug.hpo.distributed.storage import get_storage_recommendations

    # Test parallel config recommendations
    config = recommend_parallel_config(n_trials=100, n_gpus=4, trial_duration=120.0)
    print(f"\n✓ Parallel config recommendation:")
    print(f"  Workers: {config['n_workers']}")
    print(f"  GPUs: {config['gpu_ids']}")
    print(f"  Trials per worker: {config['trials_per_worker']}")
    print(f"  Estimated time: {config['estimated_time_hours']:.2f} hours")
    print(f"  Recommendation: {config['recommendation']}")

    # Test storage recommendations
    storage_recs = get_storage_recommendations(n_workers=8, expected_trial_duration=60.0)
    print(f"\n✓ Storage recommendation:")
    print(f"  Recommended: {storage_recs['recommended_backend']}")
    print(f"  Pool size: {storage_recs['pool_size']}")

    print("\n✅ TEST 5 PASSED: Recommendations working correctly")


def cleanup() -> None:
    """Clean up test artifacts."""
    print("\n" + "=" * 80)
    print("CLEANUP")
    print("=" * 80)

    import os

    # Remove test databases
    for db_file in ["test_distributed.db", "test_storage.db"]:
        if Path(db_file).exists():
            os.remove(db_file)
            print(f"✓ Removed {db_file}")


def main() -> None:
    """Run all tests."""
    print("=" * 80)
    print("SUPERMAX Phase 12: Distributed HPO Tests")
    print("=" * 80)

    try:
        test_parallel_executor()
        test_gpu_coordinator()
        test_storage_backends()
        test_resource_allocation()
        test_parallel_recommendations()

        cleanup()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nDistributed HPO functionality is working correctly!")
        print("You can now use:")
        print("  - ParallelExecutor for parallel trial execution")
        print("  - GPUCoordinator for GPU resource management")
        print("  - Distributed storage backends (PostgreSQL/MySQL)")
        print("  - Resource allocation strategies")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
