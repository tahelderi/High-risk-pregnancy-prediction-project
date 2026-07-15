"""Shared modelling utilities for the pregnancy risk project."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "database" / "pregnancy_risk.db"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "Data1": {
        "view": "vw_data1_modeling",
        "target": "risk",
        "id_columns": ["pregnancy_id", "source_row_id"],
        "features": [
            "age",
            "systolic_bp",
            "diastolic_bp",
            "blood_sugar",
            "body_temp_f",
            "bmi",
            "previous_complications",
            "preexisting_diabetes",
            "gestational_diabetes",
            "mental_health",
            "heart_rate",
        ],
        "notes": "Evaluated for comparison; previous near-perfect results make leakage or target-generation risk suspicious.",
    },
    "Data2": {
        "view": "vw_data2_modeling",
        "target": "high_risk",
        "id_columns": ["pregnancy_id", "source_row_id"],
        "features": [
            "age",
            "gravida_n",
            "tt_injection_n",
            "pregnancy_weeks",
            "weight_kg",
            "height_cm",
            "systolic_bp",
            "diastolic_bp",
            "fetal_heartbeat_bpm",
            "urine_sugar_yes",
            "vdrl_positive",
            "hrsag_positive",
            "bmi_calc",
        ],
        "excluded_features": {
            "height_m": "Dropped because it is an exact unit conversion of height_cm.",
        },
        "notes": "Primary deployment candidate because it is complete after cleaning and has less obvious leakage risk than Data1.",
    },
}


def ensure_output_dirs() -> None:
    for path in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_dataset(dataset_name: str) -> pd.DataFrame:
    config = DATASET_CONFIGS[dataset_name]
    query = f"SELECT * FROM {config['view']};"
    with sqlite3.connect(DATABASE_PATH) as conn:
        return pd.read_sql_query(query, conn)


def split_xy(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    config = DATASET_CONFIGS[dataset_name]
    X = df[config["features"]].copy()
    y = df[config["target"]].astype(int).copy()
    ids = df["pregnancy_id"].astype(int).copy()
    return X, y, ids


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value
