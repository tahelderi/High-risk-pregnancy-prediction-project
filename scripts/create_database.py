"""Create the SQLite database and apply the schema."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_PATH = DATABASE_DIR / "pregnancy_risk.db"
SCHEMA_PATH = DATABASE_DIR / "database_schema.sql"


def create_database(database_path: Path = DATABASE_PATH, recreate: bool = False) -> None:
    if recreate and database_path.exists():
        database_path.unlink()

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

    with sqlite3.connect(database_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(schema_sql)
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the SQLite project database.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the existing database file before creating the schema.",
    )
    args = parser.parse_args()
    create_database(recreate=args.recreate)


if __name__ == "__main__":
    main()
