#!/usr/bin/env python3
"""
Fix Optuna database alembic_version issue.

This script fixes the AssertionError that occurs when the alembic_version
table in the Optuna SQLite database is empty. This can happen when the
database is created but Alembic migrations aren't properly applied.

Usage:
    python fix_optuna_db.py [database_path]

Default database_path is 'optuna.db' in the current directory.
"""

import sqlite3
import sys
from pathlib import Path


def fix_optuna_database(db_path: str = "optuna.db") -> None:
    """Fix the Optuna database by inserting the correct Alembic revision."""
    db_file = Path(db_path)

    if not db_file.exists():
        print(f"Database file not found: {db_path}")
        print("This is normal if HPO hasn't been run yet.")
        return

    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()

        # Check if alembic_version table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='alembic_version'"
        )
        if not cursor.fetchone():
            print(f"Table 'alembic_version' not found in {db_path}")
            print("Database might not be initialized yet.")
            conn.close()
            return

        # Check current version
        cursor.execute("SELECT version_num FROM alembic_version")
        current = cursor.fetchall()

        if current:
            print(f"Database {db_path} already has alembic version: {current[0][0]}")
            print("No fix needed.")
        else:
            # Insert the correct revision for Optuna 2.10.1
            revision = "v2.6.0.a"
            cursor.execute(
                "INSERT INTO alembic_version (version_num) VALUES (?)", (revision,)
            )
            conn.commit()
            print(f"✓ Fixed {db_path}: Inserted alembic revision {revision}")

        conn.close()

    except Exception as e:
        print(f"Error fixing database {db_path}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "optuna.db"
    fix_optuna_database(db_path)
