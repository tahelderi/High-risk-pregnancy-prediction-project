"""Build the Data1 model and the dual-model metadata (Data1 and Data2 stay separate)."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from modeling_common import (
    CV_SPLITS,
    DATASET_CONFIGS,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
    ensure_output_dirs,
    load_dataset,
    split_xy,
    to_jsonable,
    write_json,
)
from train_model import build_model_specs, metric_dict, select_threshold, threshold_analysis


DATA1_MODEL_PATH = MODELS_DIR / "final_model_data1.pkl"
DATA1_HOLDOUT_MODEL_PATH = MODELS_DIR / "final_holdout_model_data1.pkl"
DATA1_METADATA_PATH = MODELS_DIR / "model_metadata_data1.json"
DATA1_METRICS_PATH = REPORTS_DIR / "final_test_metrics_data1.json"
DATA1_THRESHOLD_PATH = REPORTS_DIR / "threshold_analysis_data1_model.csv"

DATA2_MODEL_PATH = MODELS_DIR / "final_model_data2.pkl"
DATA2_HOLDOUT_MODEL_PATH = MODELS_DIR / "final_holdout_model_data2.pkl"
DATA2_METADATA_PATH = MODELS_DIR / "model_metadata_data2.json"
DUAL_METADATA_PATH = MODELS_DIR / "dual_model_metadata.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_best_row(comparison: pd.DataFrame, dataset_name: str) -> pd.Series:
    candidates = comparison[
        (comparison["dataset"] == dataset_name)
        & (comparison["variant"] == "tuned")
        & (comparison["model"] != "MajorityBaseline")
    ].copy()
    candidates = candidates.sort_values(
        ["cv_recall_mean", "cv_f1_mean", "cv_roc_auc_mean"],
        ascending=[False, False, False],
    )
    return candidates.iloc[0]


def params_from_row(row: pd.Series) -> dict[str, Any]:
    return json.loads(row["params"]) if isinstance(row["params"], str) and row["params"] else {}


def model_from_row(row: pd.Series):
    specs = build_model_specs()
    model_name = str(row["model"])
    estimator = clone(specs[model_name]["estimator"])
    params = params_from_row(row)
    if params:
        estimator.set_params(**params)
    return estimator


def train_data1_model() -> dict[str, Any]:
    comparison = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    selected_row = select_best_row(comparison, "Data1")
    estimator = model_from_row(selected_row)

    df = load_dataset("Data1")
    X, y, ids = split_xy(df, "Data1")
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        ids,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = cross_val_predict(
        clone(estimator),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    train_threshold_df = threshold_analysis(y_train, oof_proba, "train_oof")
    selected_threshold = select_threshold(train_threshold_df)

    holdout_model = clone(estimator)
    holdout_model.fit(X_train, y_train)
    test_proba = holdout_model.predict_proba(X_test)[:, 1]
    test_threshold_df = threshold_analysis(y_test, test_proba, "holdout_test")
    pd.concat([train_threshold_df, test_threshold_df], ignore_index=True).to_csv(
        DATA1_THRESHOLD_PATH,
        index=False,
    )

    test_metrics = metric_dict(y_test, test_proba, selected_threshold)
    write_json(DATA1_METRICS_PATH, to_jsonable(test_metrics))

    deploy_model = clone(estimator)
    deploy_model.fit(X, y)
    joblib.dump(deploy_model, DATA1_MODEL_PATH)
    joblib.dump(holdout_model, DATA1_HOLDOUT_MODEL_PATH)

    metadata = {
        "project": "High-Risk Pregnancy Prediction Using Maternal and Clinical Data",
        "model_purpose": "Predict whether a pregnancy is likely to be classified as high-risk.",
        "not_a_medical_device": True,
        "dataset": "Data1",
        "model_name": str(selected_row["model"]),
        "variant": str(selected_row["variant"]),
        "selected_threshold": selected_threshold,
        "feature_names": DATASET_CONFIGS["Data1"]["features"],
        "target_name": DATASET_CONFIGS["Data1"]["target"],
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_splits": CV_SPLITS,
        "best_params": params_from_row(selected_row),
        "selection_rule": (
            "Select the tuned Data1 model by cross-validated recall, then use F1 and ROC-AUC as tie-breakers. "
            "Select threshold from Data1 training out-of-fold predictions."
        ),
        "holdout_metrics_at_selected_threshold": to_jsonable(test_metrics),
        "cv_metrics_from_model_comparison": {
            "accuracy_mean": selected_row["cv_accuracy_mean"],
            "precision_mean": selected_row["cv_precision_mean"],
            "recall_mean": selected_row["cv_recall_mean"],
            "f1_mean": selected_row["cv_f1_mean"],
            "roc_auc_mean": selected_row["cv_roc_auc_mean"],
        },
        "holdout_test_pregnancy_ids": [int(value) for value in ids_test.tolist()],
        "artifacts": {
            "deployment_model": "models/final_model_data1.pkl",
            "holdout_model": "models/final_holdout_model_data1.pkl",
            "metadata": "models/model_metadata_data1.json",
            "threshold_analysis": "reports/threshold_analysis_data1_model.csv",
            "holdout_metrics": "reports/final_test_metrics_data1.json",
        },
        "limitations": [
            "Predictions are statistical risk estimates, not medical diagnoses.",
            "Data1 has unusually strong historical performance and should be treated cautiously because leakage or target-generation effects may be present.",
            "Data1 and Data2 are not row-merged because they do not share patient or pregnancy identifiers.",
            "Real-world performance depends on input data quality and population similarity.",
        ],
    }
    write_json(DATA1_METADATA_PATH, to_jsonable(metadata))
    return metadata


def preserve_data2_artifacts() -> dict[str, Any]:
    data2_metadata = deepcopy(read_json(MODELS_DIR / "model_metadata.json"))
    data2_metadata["dataset"] = "Data2"
    data2_metadata["artifacts"] = {
        **data2_metadata.get("artifacts", {}),
        "deployment_model": "models/final_model_data2.pkl",
        "holdout_model": "models/final_holdout_model_data2.pkl",
        "backwards_compatible_deployment_model": "models/final_model.pkl",
        "backwards_compatible_holdout_model": "models/final_holdout_model.pkl",
        "metadata": "models/model_metadata_data2.json",
        "backwards_compatible_metadata": "models/model_metadata.json",
    }
    limitations = data2_metadata.setdefault("limitations", [])
    previous_data1_note = "Data1 was not selected for deployment because unusually strong results may indicate leakage risk."
    updated_data1_note = (
        "Data1 is available as a separate dual-model artifact, but its unusually strong results may indicate leakage risk."
    )
    data2_metadata["limitations"] = [
        updated_data1_note if item == previous_data1_note else item for item in limitations
    ]
    limitations = data2_metadata["limitations"]

    limitation = "Data2 remains the backwards-compatible default model artifact for existing prediction workflows."
    if limitation not in limitations:
        limitations.append(limitation)

    shutil.copy2(MODELS_DIR / "final_model.pkl", DATA2_MODEL_PATH)
    if (MODELS_DIR / "final_holdout_model.pkl").exists():
        shutil.copy2(MODELS_DIR / "final_holdout_model.pkl", DATA2_HOLDOUT_MODEL_PATH)
    write_json(DATA2_METADATA_PATH, to_jsonable(data2_metadata))
    return data2_metadata


def write_dual_metadata(data1_metadata: dict[str, Any], data2_metadata: dict[str, Any]) -> None:
    weights = {"Data1": 0.5, "Data2": 0.5}
    combined_threshold = (
        weights["Data1"] * float(data1_metadata["selected_threshold"])
        + weights["Data2"] * float(data2_metadata["selected_threshold"])
    )
    payload = {
        "strategy": "dual_model_inference_time_weighted_average",
        "project": "High-Risk Pregnancy Prediction Using Maternal and Clinical Data",
        "scientific_framing": (
            "Data1 and Data2 are separate training sources and are not row-merged. "
            "The combined probability is computed only at inference time when a user provides all required features for both models."
        ),
        "weights": weights,
        "weight_rationale": (
            "Equal weights are used because Data1 and Data2 do not share pregnancy identifiers or a common validation cohort. "
            "Data1 performance is strong but scientifically suspicious, so it is not overweighted despite high holdout metrics."
        ),
        "combined_threshold": combined_threshold,
        "models": {
            "Data1": {
                "model_path": "models/final_model_data1.pkl",
                "metadata_path": "models/model_metadata_data1.json",
                "feature_names": data1_metadata["feature_names"],
                "threshold": data1_metadata["selected_threshold"],
            },
            "Data2": {
                "model_path": "models/final_model_data2.pkl",
                "metadata_path": "models/model_metadata_data2.json",
                "feature_names": data2_metadata["feature_names"],
                "threshold": data2_metadata["selected_threshold"],
            },
        },
        "disclaimer": "This is a statistical project prediction, not a medical diagnosis.",
    }
    write_json(DUAL_METADATA_PATH, to_jsonable(payload))


def main() -> None:
    ensure_output_dirs()
    data1_metadata = train_data1_model()
    data2_metadata = preserve_data2_artifacts()
    write_dual_metadata(data1_metadata, data2_metadata)


if __name__ == "__main__":
    main()
