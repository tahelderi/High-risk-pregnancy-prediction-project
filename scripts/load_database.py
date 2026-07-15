"""Load the cleaned datasets into the SQLite database."""

from __future__ import annotations

import csv
import math
import sqlite3
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "database" / "pregnancy_risk.db"
SCHEMA_PATH = PROJECT_ROOT / "database" / "database_schema.sql"
DATA1_CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "data1_clean.csv"
DATA2_CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "data2_clean.csv"


def clean_value(value: Any) -> float | int | str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return int(numeric)
    return numeric


def get_value(row: dict[str, str], column: str) -> float | int | str | None:
    return clean_value(row.get(column))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_database_exists() -> None:
    if DATABASE_PATH.exists():
        return
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()


def clear_loaded_records(conn: sqlite3.Connection) -> None:
    for table in [
        "obstetric_observations",
        "lab_tests",
        "medical_history",
        "vitals",
        "pregnancies",
        "data_sources",
    ]:
        conn.execute(f"DELETE FROM {table};")


def insert_data_source(
    conn: sqlite3.Connection,
    source_dataset: str,
    cleaned_file: Path,
    loaded_row_count: int,
    notes: str,
) -> None:
    conn.execute(
        """
        INSERT INTO data_sources (source_dataset, cleaned_file, loaded_row_count, notes)
        VALUES (?, ?, ?, ?);
        """,
        (
            source_dataset,
            str(cleaned_file.relative_to(PROJECT_ROOT)),
            loaded_row_count,
            notes,
        ),
    )


def insert_data1_row(conn: sqlite3.Connection, pregnancy_id: int, source_row_id: int, row: dict[str, str]) -> None:
    conn.execute(
        """
        INSERT INTO pregnancies (
            pregnancy_id, source_dataset, source_row_id, target_risk, age, pregnancy_weeks
        )
        VALUES (?, 'Data1', ?, ?, ?, NULL);
        """,
        (
            pregnancy_id,
            source_row_id,
            get_value(row, "risk"),
            get_value(row, "age"),
        ),
    )
    conn.execute(
        """
        INSERT INTO vitals (
            pregnancy_id, systolic_bp, diastolic_bp, heart_rate, body_temp_f,
            weight_kg, height_cm, height_m, bmi
        )
        VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, ?);
        """,
        (
            pregnancy_id,
            get_value(row, "systolic_bp"),
            get_value(row, "diastolic_bp"),
            get_value(row, "heart_rate"),
            get_value(row, "body_temp_f"),
            get_value(row, "bmi"),
        ),
    )
    conn.execute(
        """
        INSERT INTO medical_history (
            pregnancy_id, previous_complications, preexisting_diabetes,
            gestational_diabetes, mental_health, gravida_n, tt_injection_n
        )
        VALUES (?, ?, ?, ?, ?, NULL, NULL);
        """,
        (
            pregnancy_id,
            get_value(row, "previous_complications"),
            get_value(row, "preexisting_diabetes"),
            get_value(row, "gestational_diabetes"),
            get_value(row, "mental_health"),
        ),
    )
    conn.execute(
        """
        INSERT INTO lab_tests (
            pregnancy_id, blood_sugar, urine_sugar_yes, vdrl_positive, hrsag_positive
        )
        VALUES (?, ?, NULL, NULL, NULL);
        """,
        (
            pregnancy_id,
            get_value(row, "blood_sugar"),
        ),
    )
    conn.execute(
        """
        INSERT INTO obstetric_observations (pregnancy_id, fetal_heartbeat_bpm)
        VALUES (?, NULL);
        """,
        (pregnancy_id,),
    )


def insert_data2_row(conn: sqlite3.Connection, pregnancy_id: int, source_row_id: int, row: dict[str, str]) -> None:
    conn.execute(
        """
        INSERT INTO pregnancies (
            pregnancy_id, source_dataset, source_row_id, target_risk, age, pregnancy_weeks
        )
        VALUES (?, 'Data2', ?, ?, ?, ?);
        """,
        (
            pregnancy_id,
            source_row_id,
            get_value(row, "high_risk"),
            get_value(row, "age"),
            get_value(row, "pregnancy_weeks"),
        ),
    )
    conn.execute(
        """
        INSERT INTO vitals (
            pregnancy_id, systolic_bp, diastolic_bp, heart_rate, body_temp_f,
            weight_kg, height_cm, height_m, bmi
        )
        VALUES (?, ?, ?, NULL, NULL, ?, ?, ?, ?);
        """,
        (
            pregnancy_id,
            get_value(row, "systolic_bp"),
            get_value(row, "diastolic_bp"),
            get_value(row, "weight_kg"),
            get_value(row, "height_cm"),
            get_value(row, "height_m"),
            get_value(row, "bmi_calc"),
        ),
    )
    conn.execute(
        """
        INSERT INTO medical_history (
            pregnancy_id, previous_complications, preexisting_diabetes,
            gestational_diabetes, mental_health, gravida_n, tt_injection_n
        )
        VALUES (?, NULL, NULL, NULL, NULL, ?, ?);
        """,
        (
            pregnancy_id,
            get_value(row, "gravida_n"),
            get_value(row, "tt_injection_n"),
        ),
    )
    conn.execute(
        """
        INSERT INTO lab_tests (
            pregnancy_id, blood_sugar, urine_sugar_yes, vdrl_positive, hrsag_positive
        )
        VALUES (?, NULL, ?, ?, ?);
        """,
        (
            pregnancy_id,
            get_value(row, "urine_sugar_yes"),
            get_value(row, "vdrl_positive"),
            get_value(row, "hrsag_positive"),
        ),
    )
    conn.execute(
        """
        INSERT INTO obstetric_observations (pregnancy_id, fetal_heartbeat_bpm)
        VALUES (?, ?);
        """,
        (
            pregnancy_id,
            get_value(row, "fetal_heartbeat_bpm"),
        ),
    )


def validate_counts(conn: sqlite3.Connection, expected_data1: int, expected_data2: int) -> None:
    actual = dict(
        conn.execute(
            """
            SELECT source_dataset, COUNT(*)
            FROM pregnancies
            GROUP BY source_dataset;
            """
        ).fetchall()
    )
    expected = {"Data1": expected_data1, "Data2": expected_data2}
    if actual != expected:
        raise ValueError(f"Pregnancy row count mismatch. Expected {expected}, got {actual}.")

    total = expected_data1 + expected_data2
    for table in ["vitals", "medical_history", "lab_tests", "obstetric_observations"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]
        if count != total:
            raise ValueError(f"{table} row count mismatch. Expected {total}, got {count}.")

    foreign_key_issues = conn.execute("SELECT COUNT(*) FROM pragma_foreign_key_check;").fetchone()[0]
    if foreign_key_issues:
        raise ValueError(f"Foreign key validation failed with {foreign_key_issues} issue(s).")


def load_database() -> None:
    ensure_database_exists()

    data1_rows = read_csv_rows(DATA1_CLEAN_PATH)
    data2_rows = read_csv_rows(DATA2_CLEAN_PATH)

    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        clear_loaded_records(conn)

        insert_data_source(
            conn,
            "Data1",
            DATA1_CLEAN_PATH,
            len(data1_rows),
            "Final cleaned Data1; feature missing values preserved for modelling-pipeline imputation.",
        )
        insert_data_source(
            conn,
            "Data2",
            DATA2_CLEAN_PATH,
            len(data2_rows),
            "Final cleaned Data2; complete numeric model-ready rows after deterministic parsing.",
        )

        pregnancy_id = 1
        for source_row_id, row in enumerate(data1_rows):
            insert_data1_row(conn, pregnancy_id, source_row_id, row)
            pregnancy_id += 1
        for source_row_id, row in enumerate(data2_rows):
            insert_data2_row(conn, pregnancy_id, source_row_id, row)
            pregnancy_id += 1

        validate_counts(conn, len(data1_rows), len(data2_rows))
        conn.commit()


if __name__ == "__main__":
    load_database()
