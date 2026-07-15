"""Hebrew RTL web app for the high-risk pregnancy risk-check tool."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

sys.path.insert(0, str(SCRIPTS_DIR))
import predict as P  # noqa: E402

app = Flask(__name__)

LABELS: dict[str, str] = {
    "age": "גיל",
    "systolic_bp": "לחץ דם סיסטולי",
    "diastolic_bp": "לחץ דם דיאסטולי",
    "blood_sugar": "רמת סוכר בדם",
    "body_temp_f": "חום גוף (פרנהייט)",
    "bmi": "BMI",
    "previous_complications": "סיבוכים קודמים בהריון",
    "preexisting_diabetes": "סוכרת קיימת לפני ההריון",
    "gestational_diabetes": "סוכרת הריון",
    "mental_health": "רקע נפשי רלוונטי",
    "heart_rate": "דופק",
    "gravida_n": "מספר הריונות",
    "tt_injection_n": "מספר חיסוני טטנוס (TT)",
    "pregnancy_weeks": "שבוע הריון",
    "weight_kg": "משקל (ק״ג)",
    "height_cm": "גובה (ס״מ)",
    "fetal_heartbeat_bpm": "דופק עוברי",
    "urine_sugar_yes": "סוכר בשתן",
    "vdrl_positive": "תוצאת VDRL חיובית",
    "hrsag_positive": "תוצאת HBsAg חיובית",
    "bmi_calc": "BMI מחושב",
}

RANGES: dict[str, tuple[float, float]] = {
    "age": (10, 65),
    "systolic_bp": (70, 220),
    "diastolic_bp": (40, 140),
    "blood_sugar": (2, 25),
    "body_temp_f": (90, 110),
    "bmi": (10, 70),
    "heart_rate": (40, 200),
    "pregnancy_weeks": (1, 45),
    "weight_kg": (25, 200),
    "height_cm": (120, 220),
    "fetal_heartbeat_bpm": (60, 220),
    "bmi_calc": (10, 70),
}

ORDINAL_123 = {"gravida_n", "tt_injection_n"}
BINARY_01 = {
    "previous_complications",
    "preexisting_diabetes",
    "gestational_diabetes",
    "mental_health",
    "urine_sugar_yes",
    "vdrl_positive",
    "hrsag_positive",
}

ALL_FIELDS = set(P.DATA1_FEATURES) | set(P.DATA2_FEATURES)
NA = "לא זמין"

RISK_LABEL_HE = {
    1: "נמצא סיכון מוגבר",
    0: "לא נמצא סיכון מוגבר לפי הנתונים שהוזנו",
}
MODE_LABEL_HE = {
    "combined": "הערכה משולבת (מדדי בריאות ובדיקות הריון)",
    "data1_only": "הערכה לפי מדדי בריאות כלליים",
    "data2_only": "הערכה לפי נתוני הריון ובדיקות",
}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_csv(path: Path) -> list[dict[str, str]] | None:
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return None


def _metric_block(meta: dict[str, Any] | None) -> dict[str, Any]:
    if not meta:
        return {k: NA for k in ("model_name", "threshold", "accuracy", "roc_auc",
                                "recall", "precision", "f1", "tn", "fp", "fn", "tp")}
    m = meta.get("holdout_metrics_at_selected_threshold", {}) or {}

    def pct(key: str) -> str:
        v = m.get(key)
        return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else NA

    def num(key: str) -> str:
        v = m.get(key)
        return str(v) if v is not None else NA

    thr = meta.get("selected_threshold")
    return {
        "model_name": meta.get("model_name", NA),
        "threshold": (f"{thr:.2f}" if isinstance(thr, (int, float)) else NA),
        "accuracy": pct("accuracy"),
        "roc_auc": pct("roc_auc"),
        "recall": pct("recall"),
        "precision": pct("precision"),
        "f1": pct("f1"),
        "tn": num("tn"), "fp": num("fp"), "fn": num("fn"), "tp": num("tp"),
    }


def load_model_info() -> dict[str, Any]:
    md2 = _read_json(MODELS_DIR / "model_metadata.json")
    md1 = _read_json(MODELS_DIR / "model_metadata_data1.json")
    dual = _read_json(MODELS_DIR / "dual_model_metadata.json")

    fi_rows = _read_csv(REPORTS_DIR / "feature_importance_final.csv") or []
    feature_importance = []
    for row in fi_rows[:10]:
        name = row.get("feature", "")
        try:
            val = float(row.get("importance_mean", "nan"))
        except ValueError:
            val = float("nan")
        feature_importance.append({
            "name": name,
            "label": LABELS.get(name, name),
            "value": f"{val:.3f}" if val == val else NA,
        })

    dq_rows = _read_csv(PROCESSED_DIR / "data_quality_summary.csv") or []

    fig_specs = [
        ("confusion_matrix_final.png", "מטריצת בלבול של המודל הסופי (Data2)"),
        ("roc_curve_data2.png", "עקומת ROC (Data2)"),
        ("precision_recall_curve_data2.png", "עקומת Precision-Recall (Data2)"),
        ("threshold_analysis_final.png", "התנהגות הסף (Threshold) – Data2"),
        ("feature_importance_final.png", "חשיבות המאפיינים – Data2"),
    ]
    figures = [{"src": f"/figures/{fn}", "caption": cap}
               for fn, cap in fig_specs if (FIGURES_DIR / fn).exists()]

    weights = (dual or {}).get("weights", {})
    combined_threshold = (dual or {}).get("combined_threshold")

    return {
        "data2": _metric_block(md2),
        "data1": _metric_block(md1),
        "combined": {
            "weight_data1": (f"{weights.get('Data1'):.2f}"
                             if isinstance(weights.get("Data1"), (int, float)) else NA),
            "weight_data2": (f"{weights.get('Data2'):.2f}"
                             if isinstance(weights.get("Data2"), (int, float)) else NA),
            "threshold": (f"{combined_threshold:.3f}"
                          if isinstance(combined_threshold, (int, float)) else NA),
        },
        "feature_importance": feature_importance,
        "data_quality": dq_rows,
        "figures": figures,
    }


MODEL_INFO = load_model_info()


class InputError(Exception):
    pass


def clean_payload(data: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, raw in data.items():
        if key not in ALL_FIELDS or raw is None:
            continue
        text = str(raw).strip()
        if text == "":
            continue
        try:
            out[key] = float(text)
        except ValueError:
            raise InputError(f"ערך לא תקין בשדה: {LABELS.get(key, key)}")
    return out


def derive_bmi(values: dict[str, float]) -> dict[str, float]:
    w, h_cm = values.get("weight_kg"), values.get("height_cm")
    if w and h_cm and h_cm > 0:
        h_m = h_cm / 100.0
        bmi = round(w / (h_m * h_m), 2)
        values.setdefault("bmi", bmi)
        values.setdefault("bmi_calc", bmi)
    return values


def validate(values: dict[str, float]) -> list[str]:
    errors: list[str] = []
    for key, val in values.items():
        if key in RANGES:
            lo, hi = RANGES[key]
            if not (lo <= val <= hi):
                errors.append(f"{LABELS[key]}: הערך צריך להיות בין {lo:g} ל-{hi:g}")
        elif key in ORDINAL_123 and val not in (1, 2, 3):
            errors.append(f"{LABELS[key]}: יש לבחור ערך תקין")
        elif key in BINARY_01 and val not in (0, 1):
            errors.append(f"{LABELS[key]}: יש לבחור כן או לא")
    return errors


def _missing_he(model_key: str, values: dict[str, float]) -> list[str]:
    return [LABELS.get(f, f) for f in P.required_features(model_key) if f not in values]


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    mode = result.get("prediction_mode", "")

    def block(pred: dict[str, Any]) -> dict[str, Any]:
        cls = int(pred["predicted_class"])
        return {
            "probability_pct": round(float(pred["predicted_probability"]) * 100),
            "predicted_class": cls,
            "risk_label": RISK_LABEL_HE[cls],
            "threshold": round(float(pred["selected_threshold"]), 3),
        }

    out: dict[str, Any] = {
        "mode": mode,
        "mode_label": MODE_LABEL_HE.get(mode, "הערכת סיכון"),
        "disclaimer": result.get("disclaimer", P.DISCLAIMER),
        "details": {},
    }
    if "data1_prediction" in result:
        out["details"]["data1"] = block(result["data1_prediction"])
    if "data2_prediction" in result:
        out["details"]["data2"] = block(result["data2_prediction"])

    if mode == "combined":
        cls = int(result["combined_predicted_class"])
        out["main"] = {
            "probability_pct": round(float(result["combined_probability"]) * 100),
            "predicted_class": cls,
            "risk_label": RISK_LABEL_HE[cls],
            "threshold": round(float(result["combined_threshold"]), 3),
        }
    elif mode == "data1_only":
        out["main"] = out["details"]["data1"]
    else:
        out["main"] = out["details"]["data2"]
    return out


@app.get("/")
def index():
    return render_template("index.html", model_info=MODEL_INFO)


@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    try:
        values = derive_bmi(clean_payload(data))
    except InputError as exc:
        return jsonify(ok=False, error=str(exc)), 422

    errors = validate(values)
    if errors:
        return jsonify(ok=False, error=" • ".join(errors)), 422

    miss1, miss2 = _missing_he("Data1", values), _missing_he("Data2", values)
    if miss1 and miss2:
        closer, missing = ("בדיקות הריון", miss2) if len(miss2) <= len(miss1) else ("מדדי בריאות כלליים", miss1)
        msg = ("כדי להפיק הערכה יש להשלים לפחות קבוצת נתונים אחת. "
               f"הכי קרוב להשלמה: {closer}. חסרים: " + ", ".join(missing))
        return jsonify(ok=False, error=msg, incomplete=True), 422

    try:
        result = P.predict_combined(values, require_both=False)
    except Exception as exc:
        return jsonify(ok=False, error=f"אירעה שגיאה בחישוב ההערכה: {exc}"), 500

    return jsonify(ok=True, **normalize_result(result))


@app.get("/figures/<path:name>")
def figures(name: str):
    return send_from_directory(FIGURES_DIR, name)


if __name__ == "__main__":
    import os

    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
