#!/usr/bin/env python
"""Distributed storage backend support (Phase 12).

This module provides utilities for setting up and managing distributed
storage backends for Optuna studies.

Key Features:
- PostgreSQL/MySQL storage setup
- Storage health checking
- Connection pooling
- Migration utilities
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import optuna

LOGGER = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Configuration for distributed storage backend."""

    backend: Literal["postgresql", "mysql", "sqlite"]
    host: str = "localhost"
    port: int | None = None
    database: str = "optuna"
    username: str | None = None
    password: str | None = None
    connection_timeout: int = 30
    pool_size: int = 5


def create_distributed_storage(
    backend: Literal["postgresql", "mysql", "sqlite"] = "postgresql",
    host: str = "localhost",
    port: int | None = None,
    database: str = "optuna",
    username: str | None = None,
    password: str | None = None,
) -> str:
    """Create storage URL for distributed backend.

    Args:
        backend: Storage backend type
        host: Database host
        port: Database port (None = default)
        database: Database name
        username: Database username
        password: Database password

    Returns:
        Storage URL string

    Examples:
        >>> storage = create_distributed_storage(
        ...     backend="postgresql",
        ...     host="localhost",
        ...     database="optuna_db",
        ...     username="optuna_user",
        ...     password="secret",
        ... )
        >>> study = optuna.create_study(storage=storage)
    """
    # Default ports
    if port is None:
        if backend == "postgresql":
            port = 5432
        elif backend == "mysql":
            port = 3306
        elif backend == "sqlite":
            # SQLite doesn't use network
            return f"sqlite:///{database}"

    # Build connection string
    if username and password:
        auth = f"{username}:{password}@"
    elif username:
        auth = f"{username}@"
    else:
        auth = ""

    storage_url = f"{backend}://{auth}{host}:{port}/{database}"

    LOGGER.info("Created distributed storage URL: %s://%s:%d/%s", backend, host, port, database)

    return storage_url


def check_storage_health(storage: str) -> dict[str, Any]:
    """Check health of storage backend.

    Args:
        storage: Storage URL

    Returns:
        Health check results

    Examples:
        >>> storage = "postgresql://localhost/optuna"
        >>> health = check_storage_health(storage)
        >>> if health["accessible"]:
        >>>     print("Storage is healthy")
    """
    import time

    result = {
        "accessible": False,
        "backend": None,
        "response_time_ms": None,
        "error": None,
    }

    # Determine backend
    if storage.startswith("postgresql"):
        result["backend"] = "postgresql"
    elif storage.startswith("mysql"):
        result["backend"] = "mysql"
    elif storage.startswith("sqlite"):
        result["backend"] = "sqlite"
    else:
        result["error"] = f"Unknown backend: {storage}"
        return result

    # Test connection
    try:
        start_time = time.time()

        # Try to create/load a test study
        test_study_name = f"_health_check_{int(time.time())}"
        study = optuna.create_study(
            study_name=test_study_name,
            storage=storage,
            load_if_exists=False,
        )

        # Cleanup test study
        optuna.delete_study(study_name=test_study_name, storage=storage)

        response_time = (time.time() - start_time) * 1000
        result["accessible"] = True
        result["response_time_ms"] = response_time

        LOGGER.info("Storage health check passed (%.2f ms)", response_time)

    except Exception as e:
        result["error"] = str(e)
        LOGGER.error("Storage health check failed: %s", e)

    return result


def migrate_study(
    source_storage: str,
    target_storage: str,
    study_name: str,
) -> dict[str, Any]:
    """Migrate study from one storage backend to another.

    Args:
        source_storage: Source storage URL
        target_storage: Target storage URL
        study_name: Study name to migrate

    Returns:
        Migration statistics

    Examples:
        >>> stats = migrate_study(
        ...     source_storage="sqlite:///local.db",
        ...     target_storage="postgresql://localhost/optuna",
        ...     study_name="my-study",
        ... )
        >>> print(f"Migrated {stats['n_trials']} trials")
    """
    LOGGER.info("Migrating study '%s' from %s to %s", study_name, source_storage, target_storage)

    # Load source study
    source_study = optuna.load_study(study_name=study_name, storage=source_storage)

    # Create target study
    target_study = optuna.create_study(
        study_name=study_name,
        storage=target_storage,
        direction=source_study.direction,
        load_if_exists=False,
    )

    # Copy trials
    n_trials = 0
    for trial in source_study.trials:
        # Add trial to target study
        target_study.add_trial(trial)
        n_trials += 1

    LOGGER.info("Migration complete: %d trials migrated", n_trials)

    return {
        "study_name": study_name,
        "n_trials": n_trials,
        "source_backend": source_storage.split("://")[0],
        "target_backend": target_storage.split("://")[0],
    }


def create_postgresql_storage(
    host: str = "localhost",
    port: int = 5432,
    database: str = "optuna",
    username: str = "optuna_user",
    password: str | None = None,
) -> str:
    """Create PostgreSQL storage (recommended for production).

    PostgreSQL is the recommended backend for distributed HPO due to:
    - ACID compliance
    - Concurrent access support
    - High performance
    - Robust transaction handling

    Args:
        host: PostgreSQL host
        port: PostgreSQL port
        database: Database name
        username: Database username
        password: Database password

    Returns:
        Storage URL

    Examples:
        >>> storage = create_postgresql_storage(
        ...     host="10.0.0.5",
        ...     database="hpo_production",
        ...     username="hpo_user",
        ...     password="secure_password",
        ... )
    """
    return create_distributed_storage(
        backend="postgresql",
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
    )


def create_mysql_storage(
    host: str = "localhost",
    port: int = 3306,
    database: str = "optuna",
    username: str = "optuna_user",
    password: str | None = None,
) -> str:
    """Create MySQL storage (alternative for production).

    Args:
        host: MySQL host
        port: MySQL port
        database: Database name
        username: Database username
        password: Database password

    Returns:
        Storage URL
    """
    return create_distributed_storage(
        backend="mysql",
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
    )


def get_storage_recommendations(
    n_workers: int,
    expected_trial_duration: float = 60.0,
) -> dict[str, Any]:
    """Get storage backend recommendations.

    Args:
        n_workers: Number of parallel workers
        expected_trial_duration: Expected trial duration in seconds

    Returns:
        Recommendations

    Examples:
        >>> recs = get_storage_recommendations(n_workers=8, expected_trial_duration=120)
        >>> print(recs["recommended_backend"])
    """
    if n_workers == 1:
        # Single worker: SQLite is sufficient
        recommended = "sqlite"
        reason = "Single worker - SQLite provides good performance with no setup overhead"
    elif n_workers <= 4:
        # Few workers: SQLite may work but PostgreSQL recommended
        recommended = "postgresql"
        reason = "Multiple workers - PostgreSQL provides better concurrent access"
    else:
        # Many workers: PostgreSQL strongly recommended
        recommended = "postgresql"
        reason = "Many workers - PostgreSQL required for safe concurrent access"

    # Connection pool size recommendation
    pool_size = min(n_workers + 5, 20)  # Workers + overhead, capped at 20

    return {
        "recommended_backend": recommended,
        "reason": reason,
        "pool_size": pool_size,
        "notes": [
            "PostgreSQL is strongly recommended for production HPO",
            "SQLite has file locking issues with many concurrent writers",
            "Ensure database has sufficient connection limits",
        ],
    }


def setup_storage_from_config(config: StorageConfig) -> str:
    """Create storage URL from configuration.

    Args:
        config: Storage configuration

    Returns:
        Storage URL
    """
    return create_distributed_storage(
        backend=config.backend,
        host=config.host,
        port=config.port,
        database=config.database,
        username=config.username,
        password=config.password,
    )


def test_storage_performance(storage: str, n_trials: int = 100) -> dict[str, Any]:
    """Test storage backend performance.

    Args:
        storage: Storage URL
        n_trials: Number of test trials

    Returns:
        Performance metrics
    """
    import time

    import numpy as np

    LOGGER.info("Testing storage performance with %d trials", n_trials)

    # Create test study
    test_study_name = f"_perf_test_{int(time.time())}"
    study = optuna.create_study(
        study_name=test_study_name,
        storage=storage,
        load_if_exists=False,
    )

    # Simple objective
    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        return x**2

    # Measure trial creation time
    trial_times = []
    start_time = time.time()

    for _ in range(n_trials):
        trial_start = time.time()
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        trial_time = time.time() - trial_start
        trial_times.append(trial_time)

    total_time = time.time() - start_time

    # Cleanup
    optuna.delete_study(study_name=test_study_name, storage=storage)

    # Compute statistics
    trial_times_array = np.array(trial_times)
    metrics = {
        "total_time_seconds": total_time,
        "trials_per_second": n_trials / total_time,
        "avg_trial_time_ms": trial_times_array.mean() * 1000,
        "median_trial_time_ms": np.median(trial_times_array) * 1000,
        "p95_trial_time_ms": np.percentile(trial_times_array, 95) * 1000,
        "min_trial_time_ms": trial_times_array.min() * 1000,
        "max_trial_time_ms": trial_times_array.max() * 1000,
    }

    LOGGER.info(
        "Storage performance: %.2f trials/sec, avg: %.2f ms/trial",
        metrics["trials_per_second"],
        metrics["avg_trial_time_ms"],
    )

    return metrics
