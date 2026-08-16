"""Centralized SQLite access for the Supplier Intelligence Copilot."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TypeAlias


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "supplier.db"
REQUIRED_TABLES = frozenset(
    {
        "suppliers",
        "monthly_performance",
        "invoices",
        "incidents",
        "supplier_reviews",
    }
)

DatabasePath: TypeAlias = str | Path | None


def resolve_database_path(db_path: DatabasePath = None) -> Path:
    """Return the configured database path without depending on the working directory."""

    return Path(db_path).expanduser() if db_path is not None else DB_PATH


def get_connection(db_path: DatabasePath = None) -> sqlite3.Connection:
    """Return a configured SQLite connection and create its parent directory safely."""

    path = resolve_database_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def get_database_health(db_path: DatabasePath = None) -> dict[str, object]:
    """Report whether the database exists and contains all required application tables."""

    path = resolve_database_path(db_path)
    if not path.exists():
        return {
            "healthy": False,
            "path": str(path),
            "tables": [],
            "missing_tables": sorted(REQUIRED_TABLES),
        }

    with closing(get_connection(path)) as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    missing_tables = sorted(REQUIRED_TABLES - tables)
    return {
        "healthy": not missing_tables,
        "path": str(path),
        "tables": sorted(tables),
        "missing_tables": missing_tables,
    }
