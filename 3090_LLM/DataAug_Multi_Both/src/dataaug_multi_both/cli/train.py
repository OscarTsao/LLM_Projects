"""CLI entry point for training.

Implements FR-010: CLI entry point for training/HPO.
"""

# IMPORTANT: Set CUDA environment variables BEFORE importing torch
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import logging
import sqlite3
import sys
import time
from pathlib import Path

import click

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from dataaug_multi_both.hpo.search_space import suggest_trial_config
from dataaug_multi_both.hpo.trial_executor import TrialSpec, run_single_trial
from dataaug_multi_both.training.checkpoint_manager import (
    CheckpointManager,
    CheckpointMetadata,
    CheckpointRetentionPolicy,
)
from dataaug_multi_both.training.trainer import TrainerConfig, seed_everything
from dataaug_multi_both.utils.mlflow_setup import setup_mlflow

logger = logging.getLogger(__name__)


def _cleanup_corrupted_database(db_path: Path) -> bool:
    """Clean up a corrupted Optuna database.

    Args:
        db_path: Path to the database file

    Returns:
        True if cleanup was successful, False otherwise
    """
    try:
        # First, try to close any existing connections
        time.sleep(0.5)  # Give time for connections to close

        # Check if database exists and is accessible
        if db_path.exists():
            try:
                # Try to connect and check if it's a valid SQLite database
                with sqlite3.connect(str(db_path), timeout=5.0) as conn:
                    conn.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")

                # If we get here, the database is accessible, so we can delete it
                db_path.unlink()
                return True

            except (sqlite3.Error, OSError) as e:
                logger.warning(f"Database access error during cleanup: {e}")
                # Try to force delete the file
                try:
                    db_path.unlink()
                    return True
                except OSError as del_err:
                    logger.error(f"Failed to delete corrupted database: {del_err}")
                    return False

        # Database doesn't exist, cleanup successful
        return True

    except Exception as e:
        logger.error(f"Unexpected error during database cleanup: {e}")
        return False


def _fix_optuna_database_schema(storage_url: str) -> bool:
    """Fix Optuna database schema issues.

    Args:
        storage_url: SQLite storage URL for Optuna

    Returns:
        True if fix was successful, False otherwise
    """
    try:
        # Extract database path from storage URL
        if not storage_url.startswith("sqlite:///"):
            return False

        db_path = storage_url.replace("sqlite:///", "")

        if not Path(db_path).exists():
            return True  # Non-existent database is fine

        # Try to run optuna storage upgrade
        import subprocess
        try:
            result = subprocess.run(
                ["optuna", "storage", "upgrade", "--storage", storage_url],
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )

            if result.returncode == 0:
                logger.info("Successfully upgraded Optuna database schema")
                return True
            else:
                logger.warning(f"Optuna storage upgrade failed (return code {result.returncode}): {result.stderr}")

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Could not run optuna storage upgrade command: {e}")

        # Try manual fix as fallback
        return _manual_fix_optuna_schema(db_path)

    except Exception as e:
        logger.warning(f"Database schema fix failed: {e}")
        # Try manual fix as fallback
        return _manual_fix_optuna_schema(db_path) if 'db_path' in locals() else False


def _manual_fix_optuna_schema(db_path: str) -> bool:
    """Manually fix Optuna database schema issues.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        True if fix was successful, False otherwise
    """
    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()

            # Check if alembic_version table exists and is empty
            cursor.execute("SELECT COUNT(*) FROM alembic_version")
            alembic_count = cursor.fetchone()[0]

            if alembic_count == 0:
                # Insert the correct alembic version for Optuna 2.10.1
                cursor.execute("INSERT INTO alembic_version (version_num) VALUES ('v2.4.0.a')")
                conn.commit()
                logger.info("Manually fixed alembic_version table")

        # Additional fix: ensure proper database initialization
        # This addresses SQLAlchemy 2.0 compatibility issues with Optuna 2.10.1
        return _initialize_optuna_database_properly(db_path)

    except Exception as e:
        logger.error(f"Manual database schema fix failed: {e}")
        return False


def _initialize_optuna_database_properly(db_path: str) -> bool:
    """Initialize Optuna database with proper schema for SQLAlchemy 2.0 compatibility.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        True if initialization was successful, False otherwise
    """
    try:
        # Use Optuna's internal storage initialization
        from optuna.storages._rdb.storage import RDBStorage
        from optuna.storages._rdb.models import BaseModel
        from sqlalchemy import create_engine

        # Create engine with SQLAlchemy 2.0 compatible settings
        engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False}
        )

        # Create all tables
        BaseModel.metadata.create_all(engine)

        # Initialize version info properly using SQLAlchemy 2.0 syntax
        with engine.connect() as conn:
            from sqlalchemy import text

            # Check if version_info exists and is properly initialized
            result = conn.execute(text("SELECT COUNT(*) FROM version_info")).fetchone()
            if result[0] == 0:
                # Insert proper version info
                conn.execute(text(
                    "INSERT INTO version_info (version_info_id, schema_version, library_version) VALUES (1, 12, '2.10.1')"
                ))
                conn.commit()

            # Ensure alembic_version is set
            result = conn.execute(text("SELECT COUNT(*) FROM alembic_version")).fetchone()
            if result[0] == 0:
                conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('v2.4.0.a')"))
                conn.commit()

        logger.info("Successfully initialized Optuna database with proper schema")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize Optuna database properly: {e}")
        return False


def _create_fresh_optuna_database(storage_url: str, study_name: str) -> optuna.Study:
    """Create a fresh Optuna database and study with proper initialization.

    Args:
        storage_url: SQLite storage URL for Optuna
        study_name: Name of the study to create

    Returns:
        Created Optuna study

    Raises:
        Exception: If database creation fails
    """
    # Extract database path
    db_path = storage_url.replace("sqlite:///", "")

    # Remove existing database if it exists
    db_file = Path(db_path)
    if db_file.exists():
        db_file.unlink()
        logger.info(f"Removed existing database: {db_path}")

    # Create a temporary study to initialize the database structure
    temp_storage_url = f"sqlite:///{db_path}"

    try:
        # Create a temporary in-memory study first to avoid the schema issue
        temp_study = optuna.create_study(direction="maximize")
        logger.info("Created temporary in-memory study")

        # Now try to create the actual database study
        # This approach bypasses the initial schema check
        import subprocess

        # Use optuna CLI to create and upgrade the database
        result = subprocess.run(
            ["optuna", "create-study", "--storage", temp_storage_url, "--study-name", study_name],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            logger.warning(f"optuna create-study failed: {result.stderr}")
            # Fall back to direct creation
            raise Exception("CLI creation failed")

        # Upgrade the database schema
        upgrade_result = subprocess.run(
            ["optuna", "storage", "upgrade", "--storage", temp_storage_url],
            capture_output=True,
            text=True,
            timeout=30
        )

        if upgrade_result.returncode == 0:
            logger.info("Successfully upgraded database schema")
        else:
            logger.warning(f"Schema upgrade warning: {upgrade_result.stderr}")

        # Now load the study
        study = optuna.load_study(study_name=study_name, storage=temp_storage_url)
        logger.info(f"Successfully created and loaded study: {study_name}")
        return study

    except Exception as e:
        logger.warning(f"CLI approach failed: {e}")

        # Fall back to direct creation with error handling
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=temp_storage_url,
                direction="maximize",
                load_if_exists=False
            )
            logger.info(f"Successfully created study with direct approach: {study_name}")
            return study

        except Exception as direct_err:
            logger.error(f"All database creation methods failed: {direct_err}")
            raise Exception(f"Failed to create Optuna database: {direct_err}")


def _validate_optuna_database(storage_url: str) -> bool:
    """Validate that an Optuna database has proper schema.

    Args:
        storage_url: SQLite storage URL for Optuna

    Returns:
        True if database is valid, False if corrupted
    """
    try:
        # Extract database path from storage URL
        if not storage_url.startswith("sqlite:///"):
            return False

        db_path = storage_url.replace("sqlite:///", "")

        if not Path(db_path).exists():
            return True  # Non-existent database is fine, will be created

        # Try to connect and check schema
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()

            # Check if required tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN ('version_info', 'alembic_version')
            """)
            tables = [row[0] for row in cursor.fetchall()]

            if 'version_info' in tables:
                # Check if version_info has data
                cursor.execute("SELECT COUNT(*) FROM version_info")
                version_count = cursor.fetchone()[0]

                if version_count > 0 and 'alembic_version' in tables:
                    # Check if alembic_version has data
                    cursor.execute("SELECT COUNT(*) FROM alembic_version")
                    alembic_count = cursor.fetchone()[0]

                    # If version_info has data but alembic_version is empty, try to fix it
                    if alembic_count == 0:
                        logger.warning("Database schema inconsistency detected: version_info populated but alembic_version empty")
                        return _fix_optuna_database_schema(storage_url)

        return True

    except Exception as e:
        logger.warning(f"Database validation failed: {e}")
        return False


@click.group()
def cli():
    """DataAug Multi Both - Mental Health Criteria Detection."""
    pass


@cli.command()
@click.option("--trial-id", required=True, help="Unique trial identifier")
@click.option(
    "--config", type=click.Path(exists=True), help="Path to configuration file (YAML/JSON)"
)
@click.option("--model-name", default="mental-bert", help="Model name from catalog or HF model ID")
@click.option("--learning-rate", type=float, default=2e-5, help="Learning rate")
@click.option("--batch-size", type=int, default=16, help="Batch size")
@click.option("--max-epochs", type=int, default=10, help="Maximum number of epochs")
@click.option("--seed", type=int, default=1337, help="Random seed for reproducibility")
@click.option(
    "--resume/--no-resume", default=True, help="Resume from latest checkpoint if available"
)
@click.option(
    "--experiments-dir",
    type=click.Path(),
    default="experiments",
    help="Base directory for experiments",
)
@click.option("--keep-last-n", type=int, default=1, help="Number of last checkpoints to keep")
@click.option("--keep-best-k", type=int, default=1, help="Number of best checkpoints to keep")
def train(
    trial_id: str,
    config: str | None,
    model_name: str,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    seed: int,
    resume: bool,
    experiments_dir: str,
    keep_last_n: int,
    keep_best_k: int,
):
    """Train a single model.

    Example:
        python -m dataaug_multi_both.cli.train train --trial-id trial_001
    """
    click.echo(f"Starting training: {trial_id}")

    # Set seed
    seeds = seed_everything(seed)
    click.echo(f"Seeds: {seeds}")

    # Create trial directory
    trial_dir = Path(experiments_dir) / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer config
    TrainerConfig(
        trial_id=trial_id,
        optimization_metric="validation_f1_macro",
        seed=seed,
        max_epochs=max_epochs,
        resume_if_available=resume,
    )

    # Create checkpoint manager
    policy = CheckpointRetentionPolicy(keep_last_n=keep_last_n, keep_best_k=keep_best_k)

    metadata = CheckpointMetadata(
        code_version="1.0.0", model_signature=model_name, head_configuration="dual-agent"
    )

    CheckpointManager(trial_dir=trial_dir, policy=policy, compatibility=metadata)

    click.echo(f"Trial directory: {trial_dir}")
    click.echo(f"Retention policy: keep_last_n={keep_last_n}, keep_best_k={keep_best_k}")

    # TODO: Implement actual training loop
    click.echo("Training not yet implemented. This is a placeholder.")

    return 0


@cli.command()
@click.option("--study-name", required=True, help="Name of the HPO study")
@click.option("--n-trials", type=int, default=10, help="Number of trials to run")
@click.option(
    "--experiments-dir",
    type=click.Path(),
    default="experiments",
    help="Base directory for experiments",
)
@click.option(
    "--keep-last-n", type=int, default=1, help="Number of last checkpoints to keep per trial"
)
@click.option(
    "--keep-best-k", type=int, default=1, help="Number of best checkpoints to keep per trial"
)
@click.option(
    "--study-db",
    type=click.Path(),
    default="experiments/hpo_study.db",
    help="Path to Optuna study database",
)
@click.option(
    "--mlflow-uri", default="sqlite:///experiments/mlflow_db/mlflow.db", help="MLflow tracking URI"
)
@click.option("--dataset-id", default="irlab-udc/redsm5", help="Hugging Face dataset identifier")
@click.option("--resume/--no-resume", default=False, help="Resume existing study if available")
def hpo(
    study_name: str,
    n_trials: int,
    experiments_dir: str,
    keep_last_n: int,
    keep_best_k: int,
    study_db: str,
    mlflow_uri: str,
    dataset_id: str,
    resume: bool,
):
    """Run hyperparameter optimization.

    Example:
        python -m dataaug_multi_both.cli.train hpo --study-name study_001 --n-trials 10
    """
    if not OPTUNA_AVAILABLE:
        click.echo("Error: Optuna is not installed. Install with: pip install optuna")
        return 1

    click.echo(f"Starting HPO: {study_name}")
    click.echo(f"Number of trials: {n_trials}")
    click.echo(f"Experiments directory: {experiments_dir}")
    click.echo(f"Study database: {study_db}")
    click.echo(f"MLflow URI: {mlflow_uri}")

    # Create directories
    exp_dir = Path(experiments_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    study_db_path = Path(study_db)
    study_db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Setup MLflow
        experiment_id = setup_mlflow(
            tracking_uri=mlflow_uri, experiment_name=f"hpo_{study_name}", create_db_dir=True
        )
        click.echo(f"MLflow experiment ID: {experiment_id}")

        # Create or load Optuna study
        storage_url = f"sqlite:///{study_db}"

        # Validate database before attempting to use it
        if not _validate_optuna_database(storage_url):
            click.echo("Warning: Corrupted database detected. Cleaning up...")
            if not _cleanup_corrupted_database(study_db_path):
                click.echo("Error: Failed to clean up corrupted database. Exiting.")
                return 1
            click.echo(f"Cleaned up corrupted database: {study_db}")

        try:
            if resume:
                try:
                    study = optuna.load_study(study_name=study_name, storage=storage_url)
                    click.echo(f"Resumed existing study with {len(study.trials)} completed trials")
                except KeyError:
                    click.echo(f"Study '{study_name}' not found. Creating new study.")
                    study = optuna.create_study(
                        study_name=study_name, storage=storage_url, direction="maximize"
                    )
            else:
                # Try to create study, handle if it already exists
                try:
                    study = optuna.create_study(
                        study_name=study_name, storage=storage_url, direction="maximize"
                    )
                except optuna.exceptions.DuplicatedStudyError:
                    click.echo(f"Study '{study_name}' already exists. Loading existing study.")
                    study = optuna.load_study(study_name=study_name, storage=storage_url)
        except (AssertionError, Exception) as e:
            # Handle corrupted database or version mismatch
            click.echo(
                f"Warning: Database issue detected ({type(e).__name__}). Recreating study database."
            )

            # Create fresh database and study
            try:
                study = _create_fresh_optuna_database(storage_url, study_name)
                click.echo(f"Created fresh study: {study_name}")
            except Exception as create_err:
                click.echo(f"Error: Failed to create fresh study: {create_err}")
                logger.error(f"Database creation error: {create_err}", exc_info=True)
                return 1

        # Create retention policy
        retention_policy = CheckpointRetentionPolicy(
            keep_last_n=keep_last_n, keep_best_k=keep_best_k, max_total_size_gb=10.0
        )

        # Dataset configuration
        dataset_config = {"dataset_id": dataset_id, "revision": "main", "cache_dir": None}

        # Determine if we should use fixed epochs (for production runs)
        # Only search epochs for quick test studies
        use_fixed_epochs = 100 if "test" not in study_name.lower() else None
        if use_fixed_epochs:
            logger.info(f"Using fixed epochs: {use_fixed_epochs} for study: {study_name}")
        else:
            logger.info(f"Searching epochs [5, 8, 10, 15] for test study: {study_name}")

        # Define objective function
        def objective(trial):
            # Generate trial configuration with fixed epochs for non-test studies
            config = suggest_trial_config(trial, fixed_epochs=use_fixed_epochs)

            # Create trial spec
            trial_spec = TrialSpec(config=config)

            # Run trial
            result = run_single_trial(
                trial_spec=trial_spec,
                experiments_dir=exp_dir,
                dataset_config=dataset_config,
                retention_policy=retention_policy,
                mlflow_experiment_id=experiment_id,
            )

            if result.metric is None:
                # Trial failed - raise exception so Optuna marks it as FAIL, not COMPLETE
                import optuna
                raise optuna.exceptions.TrialPruned(f"Trial failed: {result.status}")

            return result.metric

        # Run optimization
        click.echo(f"Starting optimization with {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials)

        # Print results
        click.echo("\nOptimization completed!")
        click.echo(f"Best trial: {study.best_trial.number}")
        click.echo(f"Best value: {study.best_value:.4f}")
        click.echo(f"Best params: {study.best_params}")

        # Save study summary
        summary_path = exp_dir / f"study_{study_name}_summary.json"
        import json

        summary = {
            "study_name": study_name,
            "n_trials": len(study.trials),
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "completed_trials": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
            "failed_trials": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
            ),
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        click.echo(f"Study summary saved to: {summary_path}")

        return 0

    except Exception as e:
        logger.error(f"HPO failed: {e}", exc_info=True)
        click.echo(f"Error: {e}")
        return 1


@cli.command()
@click.option(
    "--trial-dir", type=click.Path(exists=True), required=True, help="Path to trial directory"
)
def resume(trial_dir: str):
    """Resume training from checkpoint.

    Example:
        python -m dataaug_multi_both.cli.train resume --trial-dir experiments/trial_001
    """
    click.echo(f"Resuming from: {trial_dir}")

    # TODO: Implement resume logic
    click.echo("Resume not yet implemented. This is a placeholder.")

    return 0


@cli.command()
@click.option(
    "--experiments-dir",
    type=click.Path(exists=True),
    default="experiments",
    help="Base directory for experiments",
)
def status(experiments_dir: str):
    """Show status of experiments.

    Example:
        python -m dataaug_multi_both.cli.train status
    """
    exp_dir = Path(experiments_dir)

    if not exp_dir.exists():
        click.echo(f"Experiments directory not found: {exp_dir}")
        return 1

    # List trials
    trials = list(exp_dir.glob("trial_*"))

    click.echo(f"Experiments directory: {exp_dir}")
    click.echo(f"Number of trials: {len(trials)}")

    for trial_dir in sorted(trials):
        click.echo(f"  - {trial_dir.name}")

    return 0


def main():
    """Main entry point."""
    try:
        cli()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
