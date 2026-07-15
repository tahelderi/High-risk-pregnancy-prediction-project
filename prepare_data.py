"""Clean the raw datasets and write the processed outputs to data/processed/."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA1_RAW_PATH = PROJECT_ROOT / "data1.csv"
DATA2_RAW_PATH = PROJECT_ROOT / "data2.xlsx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA1_CLEAN_PATH = PROCESSED_DIR / "data1_clean.csv"
DATA2_CLEAN_PATH = PROCESSED_DIR / "data2_clean.csv"
QUALITY_SUMMARY_PATH = PROCESSED_DIR / "data_quality_summary.csv"
MISSING_SUMMARY_PATH = PROCESSED_DIR / "missing_values_summary.csv"
TARGET_SUMMARY_PATH = PROCESSED_DIR / "target_distribution_summary.csv"
PLAUSIBILITY_SUMMARY_PATH = PROCESSED_DIR / "plausibility_summary.csv"
CLEANING_LOG_PATH = PROCESSED_DIR / "cleaning_log.json"


@dataclass(frozen=True)
class RangeRule:
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: tuple[int | float, ...] | None = None
    unit: str = ""
    action: str = "set_missing"
    note: str = ""

    def description(self) -> str:
        if self.allowed_values is not None:
            return f"allowed values: {sorted(self.allowed_values)}"
        return f"{self.min_value} to {self.max_value} {self.unit}".strip()


DATA1_COLUMN_MAP = {
    "Age": "age",
    "Systolic BP": "systolic_bp",
    "Diastolic": "diastolic_bp",
    "BS": "blood_sugar",
    "Body Temp": "body_temp_f",
    "BMI": "bmi",
    "Previous Complications": "previous_complications",
    "Preexisting Diabetes": "preexisting_diabetes",
    "Gestational Diabetes": "gestational_diabetes",
    "Mental Health": "mental_health",
    "Heart Rate": "heart_rate",
}

DATA1_RANGE_RULES = {
    "age": RangeRule(10, 65, unit="years", note="Removes impossible maternal age values such as 325."),
    "systolic_bp": RangeRule(70, 220, unit="mmHg"),
    "diastolic_bp": RangeRule(40, 140, unit="mmHg"),
    "blood_sugar": RangeRule(2, 25, unit="mmol/L or source units"),
    "body_temp_f": RangeRule(90, 110, unit="F"),
    "bmi": RangeRule(10, 70, unit="kg/m^2", note="BMI of 0 is impossible and is set missing."),
    "previous_complications": RangeRule(allowed_values=(0, 1)),
    "preexisting_diabetes": RangeRule(allowed_values=(0, 1)),
    "gestational_diabetes": RangeRule(allowed_values=(0, 1)),
    "mental_health": RangeRule(allowed_values=(0, 1)),
    "heart_rate": RangeRule(40, 200, unit="bpm"),
}

DATA2_RANGE_RULES = {
    "age": RangeRule(10, 65, unit="years"),
    "gravida_n": RangeRule(allowed_values=(1, 2, 3)),
    "tt_injection_n": RangeRule(allowed_values=(1, 2, 3)),
    "pregnancy_weeks": RangeRule(1, 45, unit="weeks"),
    "weight_kg": RangeRule(25, 200, unit="kg"),
    "height_cm": RangeRule(120, 220, unit="cm"),
    "height_m": RangeRule(1.2, 2.2, unit="m"),
    "systolic_bp": RangeRule(70, 220, unit="mmHg"),
    "diastolic_bp": RangeRule(40, 140, unit="mmHg"),
    "fetal_heartbeat_bpm": RangeRule(60, 220, unit="bpm"),
    "urine_sugar_yes": RangeRule(allowed_values=(0, 1)),
    "vdrl_positive": RangeRule(allowed_values=(0, 1)),
    "hrsag_positive": RangeRule(allowed_values=(0, 1)),
    "bmi_calc": RangeRule(10, 70, unit="kg/m^2"),
}


def normalize_missing_strings(series: pd.Series) -> pd.Series:
    if series.dtype != "object":
        return series
    return (
        series.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan, "none": np.nan})
    )


def num_first(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else np.nan


def parse_height_cm(value: Any) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).lower().strip().replace('"', "").replace("''", "'")
    feet_inches = re.search(r"(\d+)\s*'\s*(\d+)", text)
    if feet_inches:
        feet = int(feet_inches.group(1))
        inches = int(feet_inches.group(2))
        return (feet * 12 + inches) * 2.54

    parsed = num_first(text)
    if pd.isna(parsed):
        return np.nan
    if parsed <= 8:
        return parsed * 30.48
    if 1.3 <= parsed <= 2.5:
        return parsed * 100
    if parsed > 100:
        return parsed
    return np.nan


def parse_bp(value: Any) -> tuple[float, float]:
    if pd.isna(value):
        return (np.nan, np.nan)
    match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", str(value))
    if not match:
        return (np.nan, np.nan)
    return (float(match.group(1)), float(match.group(2)))


def map_yes_no_positive(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1", "positive"}:
        return 1
    if text in {"no", "n", "false", "0", "negative"}:
        return 0
    return np.nan


def map_ordinal(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    return {"1st": 1, "2nd": 2, "3rd": 3}.get(str(value).strip().lower(), np.nan)


def find_data2_header_row(raw: pd.DataFrame) -> int:
    for idx, row in raw.iterrows():
        values = {str(v).strip().lower() for v in row.tolist() if pd.notna(v)}
        if {"age", "high risk pregnancy"}.issubset(values):
            return int(idx)
    raise ValueError("Could not find Data2 header row containing Age and high risk pregnancy.")


def apply_plausibility_rules(
    df: pd.DataFrame,
    dataset_name: str,
    rules: dict[str, RangeRule],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    cleaned = df.copy()
    rows: list[dict[str, Any]] = []

    for column, rule in rules.items():
        if column not in cleaned.columns:
            continue

        series = cleaned[column]
        if rule.allowed_values is not None:
            invalid_mask = series.notna() & ~series.isin(rule.allowed_values)
        else:
            invalid_mask = series.notna()
            if rule.min_value is not None:
                invalid_mask &= series < rule.min_value
            if rule.max_value is not None:
                invalid_mask |= series.notna() & (series > rule.max_value)

        invalid_values = sorted(series.loc[invalid_mask].dropna().unique().tolist())
        rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                "rule": rule.description(),
                "invalid_count": int(invalid_mask.sum()),
                "invalid_values": "; ".join(map(str, invalid_values[:10])),
                "action": rule.action if int(invalid_mask.sum()) else "none",
                "note": rule.note,
            }
        )

        if invalid_mask.any() and rule.action == "set_missing":
            cleaned.loc[invalid_mask, column] = np.nan

    return cleaned, rows


def quality_row(df: pd.DataFrame, dataset_name: str, target_col: str | None = None) -> dict[str, Any]:
    total_cells = int(df.shape[0] * df.shape[1])
    missing_cells = int(df.isna().sum().sum())
    row = {
        "dataset": dataset_name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": missing_cells,
        "missing_percent": round((missing_cells / total_cells) * 100, 3) if total_cells else 0.0,
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if target_col and target_col in df.columns:
        row["target_column"] = target_col
    return row


def missing_rows(df: pd.DataFrame, dataset_name: str) -> list[dict[str, Any]]:
    out = []
    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        out.append(
            {
                "dataset": dataset_name,
                "column": column,
                "missing_count": missing_count,
                "missing_percent": round(float(df[column].isna().mean() * 100), 3),
            }
        )
    return out


def target_rows(df: pd.DataFrame, dataset_name: str, target_col: str) -> list[dict[str, Any]]:
    counts = df[target_col].value_counts(dropna=False).sort_index()
    total = len(df)
    return [
        {
            "dataset": dataset_name,
            "target_column": target_col,
            "target_value": value,
            "count": int(count),
            "percent": round((int(count) / total) * 100, 3) if total else 0.0,
        }
        for value, count in counts.items()
    ]


def clean_data1(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    df = raw.copy()
    df.columns = df.columns.astype(str).str.strip()

    missing_required = [column for column in [*DATA1_COLUMN_MAP, "Risk Level"] if column not in df.columns]
    if missing_required:
        raise KeyError(f"Data1 is missing required columns: {missing_required}")

    target = df["Risk Level"].astype(str).str.strip().str.lower().map({"high": 1, "low": 0})
    missing_target_rows = int(target.isna().sum())

    out = pd.DataFrame()
    for original, cleaned_name in DATA1_COLUMN_MAP.items():
        out[cleaned_name] = pd.to_numeric(df[original], errors="coerce")
    out["risk"] = target

    out = out.dropna(subset=["risk"]).copy()
    out["risk"] = out["risk"].astype(int)

    out, plausibility_rows = apply_plausibility_rules(out, "Data1", DATA1_RANGE_RULES)

    duplicate_rows = int(out.duplicated().sum())
    out = out.drop_duplicates().reset_index(drop=True)

    log = {
        "raw_rows": int(raw.shape[0]),
        "rows_dropped_missing_target": missing_target_rows,
        "rows_removed_exact_duplicates": duplicate_rows,
        "final_rows": int(out.shape[0]),
        "final_columns": int(out.shape[1]),
    }
    return out, log, plausibility_rows


def clean_data2(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    header_row = find_data2_header_row(raw)
    header = raw.iloc[header_row].tolist()
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = header

    usable_columns = [pd.notna(column) and bool(str(column).strip()) for column in df.columns]
    df = df.loc[:, usable_columns]
    df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(how="all").copy()

    rename = {
        "Age": "age",
        "Gravida": "gravida",
        "TT injection": "tt_injection",
        "pregnancy duration": "pregnancy_duration",
        "weight": "weight",
        "hight": "height",
        "blood pressure": "blood_pressure",
        "fetal heartbeat": "fetal_heartbeat",
        "urine test - sugar": "urine_sugar",
        "VDRL": "vdrl",
        "HRsAG test": "hrsag_test",
        "high risk pregnancy": "high_risk_pregnancy",
    }
    df = df.rename(columns=rename)

    required = [
        "age",
        "gravida",
        "tt_injection",
        "pregnancy_duration",
        "weight",
        "height",
        "blood_pressure",
        "fetal_heartbeat",
        "urine_sugar",
        "vdrl",
        "hrsag_test",
        "high_risk_pregnancy",
    ]
    missing_required = [column for column in required if column not in df.columns]
    if missing_required:
        raise KeyError(f"Data2 is missing required columns: {missing_required}")

    for column in df.columns:
        df[column] = normalize_missing_strings(df[column])

    out = pd.DataFrame(index=df.index)
    out["age"] = pd.to_numeric(df["age"], errors="coerce")
    out["gravida_n"] = df["gravida"].apply(map_ordinal)
    out["tt_injection_n"] = df["tt_injection"].apply(map_ordinal)
    out["pregnancy_weeks"] = df["pregnancy_duration"].apply(num_first)
    out["weight_kg"] = df["weight"].apply(num_first)
    out["height_cm"] = df["height"].apply(parse_height_cm)
    out["height_m"] = out["height_cm"] / 100.0

    blood_pressure = df["blood_pressure"].apply(parse_bp)
    out["systolic_bp"] = [pair[0] for pair in blood_pressure]
    out["diastolic_bp"] = [pair[1] for pair in blood_pressure]

    out["fetal_heartbeat_bpm"] = df["fetal_heartbeat"].apply(num_first)
    out["urine_sugar_yes"] = df["urine_sugar"].apply(map_yes_no_positive)
    out["vdrl_positive"] = df["vdrl"].apply(map_yes_no_positive)
    out["hrsag_positive"] = df["hrsag_test"].apply(map_yes_no_positive)
    out["bmi_calc"] = out["weight_kg"] / (out["height_m"] ** 2)
    out.loc[out["weight_kg"].isna() | out["height_m"].isna(), "bmi_calc"] = np.nan
    out["high_risk"] = df["high_risk_pregnancy"].apply(map_yes_no_positive)

    missing_target_rows = int(out["high_risk"].isna().sum())
    out = out.dropna(subset=["high_risk"]).copy()
    out["high_risk"] = out["high_risk"].astype(int)

    out, plausibility_rows = apply_plausibility_rules(out, "Data2", DATA2_RANGE_RULES)

    feature_missing_rows = int(out.drop(columns=["high_risk"]).isna().any(axis=1).sum())
    out = out.dropna().copy()

    duplicate_rows = int(out.duplicated().sum())
    out = out.drop_duplicates().reset_index(drop=True)

    integer_columns = [
        "age",
        "gravida_n",
        "tt_injection_n",
        "urine_sugar_yes",
        "vdrl_positive",
        "hrsag_positive",
        "high_risk",
    ]
    for column in integer_columns:
        out[column] = out[column].astype(int)

    log = {
        "raw_rows_including_metadata": int(raw.shape[0]),
        "header_row_zero_based": header_row,
        "data_rows_after_header": int(df.shape[0]),
        "rows_dropped_missing_target": missing_target_rows,
        "rows_dropped_missing_features": feature_missing_rows,
        "rows_removed_exact_duplicates": duplicate_rows,
        "final_rows": int(out.shape[0]),
        "final_columns": int(out.shape[1]),
    }
    return out, log, plausibility_rows


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df1_raw = pd.read_csv(DATA1_RAW_PATH)
    df2_raw = pd.read_excel(DATA2_RAW_PATH, header=None)

    df1_clean, data1_log, data1_plausibility = clean_data1(df1_raw)
    df2_clean, data2_log, data2_plausibility = clean_data2(df2_raw)

    df1_clean.to_csv(DATA1_CLEAN_PATH, index=False)
    df2_clean.to_csv(DATA2_CLEAN_PATH, index=False)

    quality = [
        quality_row(df1_raw, "Data1 raw", "Risk Level"),
        quality_row(df1_clean, "Data1 clean", "risk"),
        quality_row(df2_raw, "Data2 raw including metadata"),
        quality_row(df2_clean, "Data2 clean", "high_risk"),
    ]
    pd.DataFrame(quality).to_csv(QUALITY_SUMMARY_PATH, index=False)

    missing = [
        *missing_rows(df1_raw, "Data1 raw"),
        *missing_rows(df1_clean, "Data1 clean"),
        *missing_rows(df2_clean, "Data2 clean"),
    ]
    pd.DataFrame(missing).to_csv(MISSING_SUMMARY_PATH, index=False)

    targets = [
        *target_rows(df1_raw, "Data1 raw", "Risk Level"),
        *target_rows(df1_clean, "Data1 clean", "risk"),
        *target_rows(df2_clean, "Data2 clean", "high_risk"),
    ]
    pd.DataFrame(targets).to_csv(TARGET_SUMMARY_PATH, index=False)

    plausibility = [*data1_plausibility, *data2_plausibility]
    pd.DataFrame(plausibility).to_csv(PLAUSIBILITY_SUMMARY_PATH, index=False)

    log = {
        "outputs": {
            "data1_clean": str(DATA1_CLEAN_PATH.relative_to(PROJECT_ROOT)),
            "data2_clean": str(DATA2_CLEAN_PATH.relative_to(PROJECT_ROOT)),
            "quality_summary": str(QUALITY_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
            "missing_summary": str(MISSING_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
            "target_summary": str(TARGET_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
            "plausibility_summary": str(PLAUSIBILITY_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
        },
        "data1": data1_log,
        "data2": data2_log,
    }
    CLEANING_LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
