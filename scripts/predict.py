"""Prediction helpers and CLI for the Data1, Data2, and combined models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd

try:
    from modeling_common import MODELS_DIR, PROJECT_ROOT
except ModuleNotFoundError:
    from scripts.modeling_common import MODELS_DIR, PROJECT_ROOT


ModelKey = Literal["Data1", "Data2"]
Mode = Literal["data1", "data2", "combined", "auto"]

DATA1_MODEL_PATH = MODELS_DIR / "final_model_data1.pkl"
DATA1_METADATA_PATH = MODELS_DIR / "model_metadata_data1.json"
DATA2_MODEL_PATH = MODELS_DIR / "final_model_data2.pkl"
DATA2_METADATA_PATH = MODELS_DIR / "model_metadata_data2.json"
DUAL_METADATA_PATH = MODELS_DIR / "dual_model_metadata.json"

DEFAULT_MODEL_PATH = MODELS_DIR / "final_model.pkl"
DEFAULT_METADATA_PATH = MODELS_DIR / "model_metadata.json"

DATA1_FEATURES = [
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
]

DATA2_FEATURES = [
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
]

SAMPLE_INPUT_DATA1 = {
    "age": 30,
    "systolic_bp": 140,
    "diastolic_bp": 90,
    "blood_sugar": 8.0,
    "body_temp_f": 98.6,
    "bmi": 32.0,
    "previous_complications": 1,
    "preexisting_diabetes": 0,
    "gestational_diabetes": 1,
    "mental_health": 0,
    "heart_rate": 95,
}

SAMPLE_INPUT_DATA2 = {
    "age": 30,
    "gravida_n": 2,
    "tt_injection_n": 2,
    "pregnancy_weeks": 34,
    "weight_kg": 82,
    "height_cm": 160,
    "systolic_bp": 140,
    "diastolic_bp": 90,
    "fetal_heartbeat_bpm": 145,
    "urine_sugar_yes": 1,
    "vdrl_positive": 0,
    "hrsag_positive": 0,
    "bmi_calc": 32.03125,
}

SAMPLE_INPUT_COMBINED = {
    **SAMPLE_INPUT_DATA2,
    **SAMPLE_INPUT_DATA1,
    "bmi_calc": SAMPLE_INPUT_DATA2["bmi_calc"],
}

DISCLAIMER = (
    "This is a statistical project prediction for high-risk pregnancy screening, "
    "not a medical diagnosis. A qualified clinician must review any decision."
)

SCIENTIFIC_FRAMING = (
    "Data1 and Data2 are not row-merged because they do not share patient or "
    "pregnancy identifiers. Combined scoring is an inference-time weighted "
    "average of two separately trained model probabilities."
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_metadata(metadata_path: Path = DEFAULT_METADATA_PATH) -> dict[str, Any]:
    """Load model metadata. Defaults to the backwards-compatible Data2 metadata."""
    return _read_json(metadata_path)


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Any:
    """Load a saved model. Defaults to the backwards-compatible Data2 model."""
    return joblib.load(model_path)


def load_dual_metadata() -> dict[str, Any]:
    if DUAL_METADATA_PATH.exists():
        return _read_json(DUAL_METADATA_PATH)
    return {
        "weights": {"Data1": 0.5, "Data2": 0.5},
        "combined_threshold": 0.5,
        "weight_rationale": "Equal weights are used because dual metadata was not found.",
        "scientific_framing": SCIENTIFIC_FRAMING,
    }


def model_paths(model_key: ModelKey) -> tuple[Path, Path]:
    if model_key == "Data1":
        return DATA1_MODEL_PATH, DATA1_METADATA_PATH
    if DATA2_MODEL_PATH.exists() and DATA2_METADATA_PATH.exists():
        return DATA2_MODEL_PATH, DATA2_METADATA_PATH
    return DEFAULT_MODEL_PATH, DEFAULT_METADATA_PATH


def load_model_artifacts(model_key: ModelKey) -> tuple[Any, dict[str, Any]]:
    model_path, metadata_path = model_paths(model_key)
    metadata = load_metadata(metadata_path)
    return load_model(model_path), metadata


def _required_features(metadata: dict[str, Any]) -> list[str]:
    return [str(feature) for feature in metadata["feature_names"]]


def _fallback_features(model_key: ModelKey) -> list[str]:
    return DATA1_FEATURES if model_key == "Data1" else DATA2_FEATURES


def required_features(model_key: ModelKey) -> list[str]:
    _, metadata_path = model_paths(model_key)
    if metadata_path.exists():
        return _required_features(load_metadata(metadata_path))
    return _fallback_features(model_key)


def _missing_features_for_record(record: dict[str, Any], feature_names: list[str]) -> list[str]:
    missing: list[str] = []
    for feature in feature_names:
        value = record.get(feature)
        if feature not in record or value is None or (isinstance(value, str) and not value.strip()):
            missing.append(feature)
    return missing


def prepare_feature_frame(
    input_data: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    feature_names = _required_features(metadata)

    if isinstance(input_data, pd.DataFrame):
        frame = input_data.copy()
    elif isinstance(input_data, dict):
        frame = pd.DataFrame([input_data])
    else:
        frame = pd.DataFrame(input_data)

    missing = [feature for feature in feature_names if feature not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    features = frame.loc[:, feature_names].copy()
    invalid: list[str] = []
    for feature in feature_names:
        converted = pd.to_numeric(features[feature], errors="coerce")
        if converted.isna().any():
            bad_values = features.loc[converted.isna(), feature].head(3).tolist()
            invalid.append(f"{feature}={bad_values}")
        features[feature] = converted

    if invalid:
        raise ValueError(f"Missing or non-numeric feature value(s): {invalid}")

    return features


def score_frame(
    input_data: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    model: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Score a frame with one model (defaults to the Data2 model)."""
    metadata = metadata or load_metadata()
    model = model or load_model()
    features = prepare_feature_frame(input_data, metadata)
    probability = model.predict_proba(features)[:, 1]
    threshold = float(metadata["selected_threshold"])

    scored = features.copy()
    scored["predicted_probability"] = probability
    scored["selected_threshold"] = threshold
    scored["predicted_class"] = (scored["predicted_probability"] >= threshold).astype(int)
    scored["risk_label"] = scored["predicted_class"].map(
        {1: "high-risk pregnancy", 0: "not high-risk pregnancy"}
    )
    scored["model_name"] = str(metadata.get("model_name", "unknown"))
    scored["dataset"] = str(metadata.get("dataset", "Data2"))
    scored["disclaimer"] = DISCLAIMER
    return scored


def _single_prediction(input_values: dict[str, Any], model_key: ModelKey) -> dict[str, Any]:
    missing = _missing_features_for_record(input_values, required_features(model_key))
    if missing:
        raise ValueError(f"{model_key} prediction requires missing feature(s): {missing}")

    model, metadata = load_model_artifacts(model_key)
    scored = score_frame(input_values, model=model, metadata=metadata).iloc[0]
    feature_names = _required_features(metadata)
    return {
        "dataset": model_key,
        "model_name": str(metadata.get("model_name", "unknown")),
        "selected_threshold": float(scored["selected_threshold"]),
        "predicted_probability": float(scored["predicted_probability"]),
        "predicted_class": int(scored["predicted_class"]),
        "risk_label": str(scored["risk_label"]),
        "feature_count": len(feature_names),
        "features_used": feature_names,
        "limitations": metadata.get("limitations", []),
        "disclaimer": DISCLAIMER,
    }


def predict_data1(input_values: dict[str, Any]) -> dict[str, Any]:
    return _single_prediction(input_values, "Data1")


def predict_data2(input_values: dict[str, Any]) -> dict[str, Any]:
    return _single_prediction(input_values, "Data2")


def _risk_label(predicted_class: int) -> str:
    return "high-risk pregnancy" if predicted_class else "not high-risk pregnancy"


def predict_combined(input_values: dict[str, Any], require_both: bool = False) -> dict[str, Any]:
    """Score Data1, Data2, or both depending on supplied feature groups."""
    data1_missing = _missing_features_for_record(input_values, required_features("Data1"))
    data2_missing = _missing_features_for_record(input_values, required_features("Data2"))
    has_data1 = not data1_missing
    has_data2 = not data2_missing

    if require_both and (not has_data1 or not has_data2):
        raise ValueError(
            "Combined mode requires all Data1 and Data2 fields. "
            f"Missing Data1: {data1_missing}; missing Data2: {data2_missing}"
        )
    if not has_data1 and not has_data2:
        raise ValueError(
            "No complete feature group was supplied. "
            f"Missing Data1: {data1_missing}; missing Data2: {data2_missing}"
        )

    result: dict[str, Any] = {
        "scientific_framing": SCIENTIFIC_FRAMING,
        "disclaimer": DISCLAIMER,
    }

    if has_data1:
        result["data1_prediction"] = predict_data1(input_values)
    if has_data2:
        result["data2_prediction"] = predict_data2(input_values)

    if has_data1 and has_data2:
        dual_metadata = load_dual_metadata()
        weights = dual_metadata.get("weights", {"Data1": 0.5, "Data2": 0.5})
        data1_weight = float(weights.get("Data1", 0.5))
        data2_weight = float(weights.get("Data2", 0.5))
        total_weight = data1_weight + data2_weight
        if total_weight <= 0:
            data1_weight = data2_weight = 0.5
            total_weight = 1.0
        data1_weight = data1_weight / total_weight
        data2_weight = data2_weight / total_weight

        data1_prediction = result["data1_prediction"]
        data2_prediction = result["data2_prediction"]
        combined_probability = (
            data1_weight * float(data1_prediction["predicted_probability"])
            + data2_weight * float(data2_prediction["predicted_probability"])
        )
        combined_threshold = float(
            dual_metadata.get(
                "combined_threshold",
                data1_weight * float(data1_prediction["selected_threshold"])
                + data2_weight * float(data2_prediction["selected_threshold"]),
            )
        )
        predicted_class = int(combined_probability >= combined_threshold)
        result.update(
            {
                "prediction_mode": "combined",
                "weights": {"Data1": data1_weight, "Data2": data2_weight},
                "weight_rationale": dual_metadata.get(
                    "weight_rationale",
                    "Equal weights are used because no common validation cohort is available.",
                ),
                "combined_probability": combined_probability,
                "combined_threshold": combined_threshold,
                "combined_predicted_class": predicted_class,
                "combined_risk_label": _risk_label(predicted_class),
            }
        )
    elif has_data1:
        result.update(
            {
                "prediction_mode": "data1_only",
                "reason": "Only the complete Data1 feature group was supplied.",
            }
        )
    else:
        result.update(
            {
                "prediction_mode": "data2_only",
                "reason": "Only the complete Data2 feature group was supplied.",
            }
        )

    return result


def _flatten_prediction(result: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "prediction_mode": result.get("prediction_mode", "single_model"),
        "disclaimer": result.get("disclaimer", DISCLAIMER),
    }
    if "data1_prediction" in result:
        data1 = result["data1_prediction"]
        row.update(
            {
                "data1_model_name": data1["model_name"],
                "data1_probability": data1["predicted_probability"],
                "data1_threshold": data1["selected_threshold"],
                "data1_predicted_class": data1["predicted_class"],
                "data1_risk_label": data1["risk_label"],
            }
        )
    if "data2_prediction" in result:
        data2 = result["data2_prediction"]
        row.update(
            {
                "data2_model_name": data2["model_name"],
                "data2_probability": data2["predicted_probability"],
                "data2_threshold": data2["selected_threshold"],
                "data2_predicted_class": data2["predicted_class"],
                "data2_risk_label": data2["risk_label"],
            }
        )
    if result.get("prediction_mode") == "combined":
        row.update(
            {
                "combined_probability": result["combined_probability"],
                "combined_threshold": result["combined_threshold"],
                "combined_predicted_class": result["combined_predicted_class"],
                "combined_risk_label": result["combined_risk_label"],
                "data1_weight": result["weights"]["Data1"],
                "data2_weight": result["weights"]["Data2"],
            }
        )
    return row


def score_records(records: list[dict[str, Any]], mode: Mode) -> list[dict[str, Any]]:
    if mode == "data1":
        return [{"prediction_mode": "data1", **predict_data1(record)} for record in records]
    if mode == "data2":
        return [{"prediction_mode": "data2", **predict_data2(record)} for record in records]
    require_both = mode == "combined"
    return [_flatten_prediction(predict_combined(record, require_both=require_both)) for record in records]


def _jsonable(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _jsonable(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_jsonable(value) for value in payload]
    if hasattr(payload, "item"):
        return payload.item()
    return payload


def write_sample_files() -> None:
    data1_prediction = predict_data1(SAMPLE_INPUT_DATA1)
    data2_prediction = predict_data2(SAMPLE_INPUT_DATA2)
    combined_prediction = predict_combined(SAMPLE_INPUT_COMBINED, require_both=True)

    _write_json(MODELS_DIR / "sample_input_data1.json", SAMPLE_INPUT_DATA1)
    _write_json(MODELS_DIR / "sample_input_data2.json", SAMPLE_INPUT_DATA2)
    _write_json(MODELS_DIR / "sample_input_combined.json", SAMPLE_INPUT_COMBINED)
    _write_json(MODELS_DIR / "sample_prediction_data1.json", _jsonable(data1_prediction))
    _write_json(MODELS_DIR / "sample_prediction_data2.json", _jsonable(data2_prediction))
    _write_json(MODELS_DIR / "sample_prediction_combined.json", _jsonable(combined_prediction))

    _write_json(MODELS_DIR / "sample_input.json", SAMPLE_INPUT_DATA2)
    _write_json(MODELS_DIR / "sample_prediction_output.json", _jsonable(data2_prediction))


def _result_for_record(record: dict[str, Any], mode: Mode) -> dict[str, Any]:
    if mode == "data1":
        return predict_data1(record)
    if mode == "data2":
        return predict_data2(record)
    return predict_combined(record, require_both=(mode == "combined"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predictions with Data1, Data2, or the dual-model workflow.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-json", type=Path, help="Path to a JSON object or list of objects containing model features.")
    input_group.add_argument("--input-csv", type=Path, help="Path to a CSV file containing model features.")
    input_group.add_argument("--sample", action="store_true", help="Use the backwards-compatible Data2 sample input.")
    input_group.add_argument("--sample-data1", action="store_true", help="Use a built-in Data1 sample input.")
    input_group.add_argument("--sample-data2", action="store_true", help="Use a built-in Data2 sample input.")
    input_group.add_argument("--sample-combined", action="store_true", help="Use a built-in sample containing both feature groups.")
    parser.add_argument(
        "--mode",
        choices=["data1", "data2", "combined", "auto"],
        default="auto",
        help="Prediction mode. auto detects complete Data1 and/or Data2 feature groups.",
    )
    parser.add_argument("--output-csv", type=Path, help="Optional output path for batch scored rows.")
    args = parser.parse_args()

    if args.sample or args.sample_data1 or args.sample_data2 or args.sample_combined:
        write_sample_files()
        if args.sample_data1:
            result = predict_data1(SAMPLE_INPUT_DATA1)
        elif args.sample_combined:
            result = predict_combined(SAMPLE_INPUT_COMBINED, require_both=True)
        else:
            result = predict_data2(SAMPLE_INPUT_DATA2)
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        return

    if args.input_json:
        payload = json.loads(args.input_json.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            result = _result_for_record(payload, args.mode)
            print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
            return
        if not isinstance(payload, list):
            raise SystemExit("Input JSON must be an object or a list of objects.")
        rows = score_records(payload, args.mode)
        if args.output_csv:
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.output_csv, index=False)
            print(f"Saved {len(rows)} scored row(s) to {args.output_csv}")
        else:
            print(json.dumps(_jsonable(rows), indent=2, sort_keys=True))
        return

    if args.input_csv:
        records = pd.read_csv(args.input_csv).to_dict(orient="records")
        rows = score_records(records, args.mode)
        scored = pd.DataFrame(rows)
        if args.output_csv:
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            scored.to_csv(args.output_csv, index=False)
            print(f"Saved {len(scored)} scored row(s) to {args.output_csv}")
        else:
            print(scored.to_csv(index=False))
        return

    raise SystemExit("Use --sample, --sample-data1, --sample-data2, --sample-combined, --input-json, or --input-csv")


if __name__ == "__main__":
    main()
