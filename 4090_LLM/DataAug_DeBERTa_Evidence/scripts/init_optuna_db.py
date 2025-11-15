#!/usr/bin/env python3
"""
Initialize Optuna database properly before running HPO.

This script ensures that the Optuna SQLite database is properly initialized
with the correct Alembic version to avoid AssertionError during study creation.

Usage:
    python scripts/init_optuna_db.py [database_path]

Default database_path is 'optuna.db' in the current directory.
"""

import sqlite3
import sys
from pathlib import Path


def init_optuna_database(db_path: str = "optuna.db") -> None:
    """Initialize Optuna database with proper Alembic version."""
    db_file = Path(db_path)

    # If database doesn't exist, we'll create it with proper initialization
    if not db_file.exists():
        print(f"Database {db_path} doesn't exist. Creating and initializing...")
        try:
            # Use Optuna's RDBStorage to create the database properly
            from optuna.storages import RDBStorage

            storage = RDBStorage(
                url=f"sqlite:///{db_path}",
                engine_kwargs={"connect_args": {"timeout": 30}},
            )
            # Just creating the storage initializes the database
            print(f"✓ Created and initialized database {db_path}")
            return
        except Exception as e:
            print(f"Error initializing database with RDBStorage: {e}")
            # Fallback to manual initialization
            _manual_sqlite_init(db_path)
        return

    # Database exists - verify it's properly initialized
    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()

        # Check if alembic_version table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='alembic_version'"
        )
        if not cursor.fetchone():
            print(f"Database {db_path} missing alembic_version table. Reinitializing...")
            conn.close()
            db_file.unlink()  # Delete the corrupted database
            # Try again with RDBStorage
            try:
                from optuna.storages import RDBStorage

                storage = RDBStorage(
                    url=f"sqlite:///{db_path}",
                    engine_kwargs={"connect_args": {"timeout": 30}},
                )
                print(f"✓ Reinitialized database {db_path}")
            except Exception as e:
                print(f"Error reinitializing database: {e}")
                _manual_sqlite_init(db_path)
            return

        # Check if version is set
        cursor.execute("SELECT version_num FROM alembic_version")
        current = cursor.fetchall()

        if not current:
            print(f"Database {db_path} has empty alembic_version. Fixing...")
            revision = "v2.6.0.a"
            cursor.execute(
                "INSERT INTO alembic_version (version_num) VALUES (?)", (revision,)
            )
            conn.commit()
            print(f"✓ Fixed {db_path}: Inserted alembic revision {revision}")
        else:
            print(f"✓ Database {db_path} is properly initialized (version: {current[0][0]})")

        conn.close()

    except Exception as e:
        print(f"Error checking database {db_path}: {e}")
        sys.exit(1)


def _manual_sqlite_init(db_path: str) -> None:
    """Manually create database schema if Optuna init fails."""
    try:
        print(f"Attempting manual SQLite initialization for {db_path}...")
        from optuna.storages import RDBStorage

        # Remove any partially created file
        db_file = Path(db_path)
        if db_file.exists():
            db_file.unlink()

        # Create with RDBStorage which properly initializes everything
        storage = RDBStorage(
            url=f"sqlite:///{db_path}",
            engine_kwargs={"connect_args": {"timeout": 30}},
        )
        print(f"✓ Manually initialized database {db_path}")

    except Exception as e:
        print(f"Manual initialization failed: {e}")
        print("Attempting fallback initialization...")

        # Last resort: create a minimal database with just the version table
        try:
            db_file = Path(db_path)
            if db_file.exists():
                db_file.unlink()

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Create minimal alembic_version table
            cursor.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
            cursor.execute("INSERT INTO alembic_version (version_num) VALUES ('v2.6.0.a')")
            conn.commit()
            conn.close()

            print(f"✓ Created minimal database {db_path} with Alembic version")
            print("Note: Optuna will initialize remaining tables on first use")

        except Exception as e2:
            print(f"All initialization methods failed: {e2}")
            sys.exit(1)


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "optuna.db"
    init_optuna_database(db_path)
