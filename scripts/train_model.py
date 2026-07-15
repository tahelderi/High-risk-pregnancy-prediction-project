"""Train, tune, evaluate, and save the pregnancy-risk models."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeling_common import (
    CV_SPLITS,
    DATASET_CONFIGS,
    FIGURES_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
    ensure_output_dirs,
    load_dataset,
    split_xy,
    to_jsonable,
    write_json,
)


matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning)


PRIMARY_THRESHOLD_RECALL_FLOOR = 0.95
PRIMARY_THRESHOLD_PRECISION_FLOOR = 0.70


def build_model_specs() -> dict[str, dict[str, Any]]:
    return {
        "MajorityBaseline": {
            "estimator": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", DummyClassifier(strategy="most_frequent")),
                ]
            ),
            "param_grid": None,
            "notes": "Majority-class baseline.",
        },
        "LogisticRegression": {
            "estimator": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
                ]
            ),
            "param_grid": {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
            "notes": "Interpretable linear baseline with scaled numeric features.",
        },
        "HistGradientBoosting": {
            "estimator": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            "param_grid": {
                "model__learning_rate": [0.03, 0.06, 0.1],
                "model__max_iter": [100, 200],
                "model__max_leaf_nodes": [15, 31],
                "model__min_samples_leaf": [10, 20],
                "model__l2_regularization": [0.0, 0.1],
            },
            "notes": "Nonlinear boosted tree model.",
        },
        "RandomForest": {
            "estimator": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 6, 10],
                "model__min_samples_leaf": [1, 5],
                "model__class_weight": [None, "balanced"],
            },
            "notes": "Bagged tree model with native feature importance.",
        },
    }


def metric_dict(y_true: pd.Series | np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) == 2 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def cv_summary(estimator: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold) -> dict[str, float]:
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    scores = cross_validate(estimator, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    out: dict[str, float] = {}
    for metric in scoring:
        values = scores[f"test_{metric}"]
        out[f"cv_{metric}_mean"] = float(np.mean(values))
        out[f"cv_{metric}_std"] = float(np.std(values))
    return out


def evaluate_fit_model(
    estimator: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> tuple[Pipeline, dict[str, float | int]]:
    fitted = clone(estimator)
    fitted.fit(X_train, y_train)
    proba = fitted.predict_proba(X_test)[:, 1]
    return fitted, metric_dict(y_test, proba, threshold)


def tune_model(
    model_name: str,
    estimator: Pipeline,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> tuple[GridSearchCV, pd.DataFrame]:
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        refit="recall",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )
    grid.fit(X_train, y_train)

    rows = []
    results = pd.DataFrame(grid.cv_results_)
    metric_columns = [
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_test_precision",
        "std_test_precision",
        "mean_test_recall",
        "std_test_recall",
        "mean_test_f1",
        "std_test_f1",
        "mean_test_roc_auc",
        "std_test_roc_auc",
        "rank_test_recall",
    ]
    for _, row in results.iterrows():
        output = {
            "model": model_name,
            "params": json.dumps(to_jsonable(row["params"]), sort_keys=True),
        }
        for column in metric_columns:
            output[column] = row[column]
        rows.append(output)
    return grid, pd.DataFrame(rows)


def best_grid_cv_metrics(grid: GridSearchCV) -> dict[str, float]:
    idx = grid.best_index_
    cv_results = grid.cv_results_
    return {
        "cv_accuracy_mean": float(cv_results["mean_test_accuracy"][idx]),
        "cv_accuracy_std": float(cv_results["std_test_accuracy"][idx]),
        "cv_precision_mean": float(cv_results["mean_test_precision"][idx]),
        "cv_precision_std": float(cv_results["std_test_precision"][idx]),
        "cv_recall_mean": float(cv_results["mean_test_recall"][idx]),
        "cv_recall_std": float(cv_results["std_test_recall"][idx]),
        "cv_f1_mean": float(cv_results["mean_test_f1"][idx]),
        "cv_f1_std": float(cv_results["std_test_f1"][idx]),
        "cv_roc_auc_mean": float(cv_results["mean_test_roc_auc"][idx]),
        "cv_roc_auc_std": float(cv_results["std_test_roc_auc"][idx]),
    }


def threshold_analysis(y_true: pd.Series | np.ndarray, proba: np.ndarray, split_name: str) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.10, 0.91, 0.05), 2):
        row = metric_dict(y_true, proba, float(threshold))
        row["split"] = split_name
        rows.append(row)
    return pd.DataFrame(rows)


def select_threshold(threshold_df: pd.DataFrame) -> float:
    candidates = threshold_df[
        (threshold_df["recall"] >= PRIMARY_THRESHOLD_RECALL_FLOOR)
        & (threshold_df["precision"] >= PRIMARY_THRESHOLD_PRECISION_FLOOR)
    ].copy()
    if candidates.empty:
        candidates = threshold_df.copy()
    candidates = candidates.sort_values(
        ["recall", "f1", "precision", "threshold"],
        ascending=[False, False, False, False],
    )
    return float(candidates.iloc[0]["threshold"])


def select_final_model(comparison: pd.DataFrame) -> pd.Series:
    candidates = comparison[
        (comparison["dataset"] == "Data2")
        & (comparison["variant"] == "tuned")
        & (comparison["model"] != "MajorityBaseline")
    ].copy()
    precision_filtered = candidates[candidates["cv_precision_mean"] >= PRIMARY_THRESHOLD_PRECISION_FLOOR].copy()
    if not precision_filtered.empty:
        candidates = precision_filtered
    candidates = candidates.sort_values(
        ["cv_recall_mean", "cv_f1_mean", "cv_roc_auc_mean"],
        ascending=[False, False, False],
    )
    return candidates.iloc[0]


def plot_confusion_matrix(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_curves(curve_payloads: list[dict[str, Any]]) -> None:
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))

    for payload in curve_payloads:
        y_test = payload["y_test"]
        proba = payload["proba"]
        label = payload["label"]
        fpr, tpr, _ = roc_curve(y_test, proba)
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ax_roc.plot(fpr, tpr, label=f"{label} AUC={roc_auc_score(y_test, proba):.3f}")
        ax_pr.plot(recall, precision, label=f"{label} PR-AUC={auc(recall, precision):.3f}")

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax_roc.set_title("Data2 ROC Curve")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_roc.grid(alpha=0.25)
    fig_roc.tight_layout()
    fig_roc.savefig(FIGURES_DIR / "roc_curve_data2.png", dpi=160)
    plt.close(fig_roc)

    ax_pr.set_title("Data2 Precision-Recall Curve")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left", fontsize=8)
    ax_pr.grid(alpha=0.25)
    fig_pr.tight_layout()
    fig_pr.savefig(FIGURES_DIR / "precision_recall_curve_data2.png", dpi=160)
    plt.close(fig_pr)


def plot_feature_importance(importance_df: pd.DataFrame) -> None:
    data = importance_df.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(data["feature"], data["importance_mean"], xerr=data["importance_std"], color="#3b6ea8")
    ax.set_title("Final Model Permutation Importance")
    ax.set_xlabel("Mean F1 decrease")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance_final.png", dpi=160)
    plt.close(fig)


def plot_threshold_analysis(threshold_df: pd.DataFrame, selected_threshold: float) -> None:
    data = threshold_df[threshold_df["split"] == "train_oof"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for metric in ["precision", "recall", "f1"]:
        ax.plot(data["threshold"], data[metric], marker="o", label=metric)
    ax.axvline(selected_threshold, color="black", linestyle="--", label=f"selected {selected_threshold:.2f}")
    ax.set_title("Threshold Analysis on Training OOF Predictions")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_analysis_final.png", dpi=160)
    plt.close(fig)


def leakage_checks(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset_name, df in datasets.items():
        config = DATASET_CONFIGS[dataset_name]
        for feature in config["features"]:
            corr = df[[feature, config["target"]]].corr(numeric_only=True).iloc[0, 1]
            rows.append(
                {
                    "dataset": dataset_name,
                    "feature": feature,
                    "target_correlation": corr,
                    "abs_target_correlation": abs(corr),
                    "flag": "review" if abs(corr) >= 0.75 else "",
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "abs_target_correlation"], ascending=[True, False])


def run_training() -> None:
    ensure_output_dirs()
    specs = build_model_specs()
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    all_comparison_rows: list[dict[str, Any]] = []
    tuning_frames: list[pd.DataFrame] = []
    fitted_models: dict[tuple[str, str, str], Pipeline] = {}
    split_payloads: dict[str, dict[str, Any]] = {}
    datasets = {name: load_dataset(name) for name in DATASET_CONFIGS}
    best_params: dict[str, dict[str, Any]] = {}
    data2_curve_payloads: list[dict[str, Any]] = []

    for dataset_name, df in datasets.items():
        config = DATASET_CONFIGS[dataset_name]
        X, y, ids = split_xy(df, dataset_name)
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X,
            y,
            ids,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        split_payloads[dataset_name] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "ids_train": ids_train,
            "ids_test": ids_test,
        }

        for model_name, spec in specs.items():
            default_estimator = spec["estimator"]
            default_cv = cv_summary(default_estimator, X_train, y_train, cv)
            fitted_default, test_metrics = evaluate_fit_model(
                default_estimator, X_train, y_train, X_test, y_test, threshold=0.5
            )
            fitted_models[(dataset_name, model_name, "default")] = fitted_default
            row = {
                "dataset": dataset_name,
                "model": model_name,
                "variant": "default",
                "params": "{}",
                "feature_count": len(config["features"]),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "test_threshold": 0.5,
                "notes": spec["notes"],
                **default_cv,
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            all_comparison_rows.append(row)

            if spec["param_grid"]:
                grid, tuning_df = tune_model(model_name, default_estimator, spec["param_grid"], X_train, y_train, cv)
                tuning_df.insert(0, "dataset", dataset_name)
                tuning_frames.append(tuning_df)
                tuned_estimator = grid.best_estimator_
                tuned_proba = tuned_estimator.predict_proba(X_test)[:, 1]
                tuned_test_metrics = metric_dict(y_test, tuned_proba, threshold=0.5)
                tuned_cv = best_grid_cv_metrics(grid)
                fitted_models[(dataset_name, model_name, "tuned")] = tuned_estimator
                best_params[f"{dataset_name}_{model_name}"] = to_jsonable(grid.best_params_)
                all_comparison_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "variant": "tuned",
                        "params": json.dumps(to_jsonable(grid.best_params_), sort_keys=True),
                        "feature_count": len(config["features"]),
                        "train_rows": len(X_train),
                        "test_rows": len(X_test),
                        "test_threshold": 0.5,
                        "notes": f"Tuned with GridSearchCV refit on recall. {spec['notes']}",
                        **tuned_cv,
                        **{f"test_{k}": v for k, v in tuned_test_metrics.items()},
                    }
                )
                if dataset_name == "Data2":
                    data2_curve_payloads.append(
                        {
                            "label": f"{model_name} tuned",
                            "y_test": y_test,
                            "proba": tuned_proba,
                        }
                    )

    comparison = pd.DataFrame(all_comparison_rows)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    pd.concat(tuning_frames, ignore_index=True).to_csv(REPORTS_DIR / "tuning_results.csv", index=False)

    leakage_df = leakage_checks(datasets)
    leakage_df.to_csv(REPORTS_DIR / "leakage_checks.csv", index=False)

    final_row = select_final_model(comparison)
    final_key = (str(final_row["dataset"]), str(final_row["model"]), str(final_row["variant"]))
    holdout_model = fitted_models[final_key]
    final_split = split_payloads["Data2"]

    oof_proba = cross_val_predict(
        clone(holdout_model),
        final_split["X_train"],
        final_split["y_train"],
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    train_threshold_df = threshold_analysis(final_split["y_train"], oof_proba, "train_oof")
    selected_threshold = select_threshold(train_threshold_df)
    test_proba = holdout_model.predict_proba(final_split["X_test"])[:, 1]
    test_threshold_df = threshold_analysis(final_split["y_test"], test_proba, "holdout_test")
    threshold_df = pd.concat([train_threshold_df, test_threshold_df], ignore_index=True)
    threshold_df.to_csv(REPORTS_DIR / "threshold_analysis_final_model.csv", index=False)

    final_metrics = metric_dict(final_split["y_test"], test_proba, threshold=selected_threshold)
    write_json(REPORTS_DIR / "final_test_metrics.json", to_jsonable(final_metrics))

    cm = confusion_matrix(final_split["y_test"], (test_proba >= selected_threshold).astype(int), labels=[0, 1])
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        REPORTS_DIR / "confusion_matrix_final.csv"
    )
    plot_confusion_matrix(cm, FIGURES_DIR / "confusion_matrix_final.png", "Final Model Confusion Matrix")
    plot_curves(data2_curve_payloads)
    plot_threshold_analysis(threshold_df, selected_threshold)

    perm = permutation_importance(
        holdout_model,
        final_split["X_test"],
        final_split["y_test"],
        scoring="f1",
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": DATASET_CONFIGS["Data2"]["features"],
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(REPORTS_DIR / "feature_importance_final.csv", index=False)
    plot_feature_importance(importance_df)

    full_data2 = datasets["Data2"]
    full_X, full_y, _ = split_xy(full_data2, "Data2")
    deploy_model = clone(holdout_model)
    deploy_model.fit(full_X, full_y)
    joblib.dump(deploy_model, MODELS_DIR / "final_model.pkl")
    joblib.dump(holdout_model, MODELS_DIR / "final_holdout_model.pkl")

    metadata = {
        "project": "High-Risk Pregnancy Prediction Using Maternal and Clinical Data",
        "model_purpose": "Predict whether a pregnancy is likely to be classified as high-risk.",
        "not_a_medical_device": True,
        "dataset": "Data2",
        "model_name": str(final_row["model"]),
        "variant": str(final_row["variant"]),
        "selected_threshold": selected_threshold,
        "feature_names": DATASET_CONFIGS["Data2"]["features"],
        "excluded_features": DATASET_CONFIGS["Data2"].get("excluded_features", {}),
        "target_name": DATASET_CONFIGS["Data2"]["target"],
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_splits": CV_SPLITS,
        "selection_rule": (
            "Select tuned Data2 model by cross-validated recall, using precision and F1 as tie-breakers; "
            "select threshold from training out-of-fold predictions by maximizing recall while meeting "
            "the precision floor, then using F1 as a tie-breaker."
        ),
        "threshold_recall_floor": PRIMARY_THRESHOLD_RECALL_FLOOR,
        "threshold_precision_floor": PRIMARY_THRESHOLD_PRECISION_FLOOR,
        "best_params": best_params.get(f"Data2_{final_row['model']}", {}),
        "holdout_metrics_at_selected_threshold": to_jsonable(final_metrics),
        "holdout_test_pregnancy_ids": [int(x) for x in final_split["ids_test"].tolist()],
        "artifacts": {
            "deployment_model": "models/final_model.pkl",
            "holdout_model": "models/final_holdout_model.pkl",
            "model_comparison": "reports/model_comparison.csv",
            "tuning_results": "reports/tuning_results.csv",
            "threshold_analysis": "reports/threshold_analysis_final_model.csv",
            "feature_importance": "reports/feature_importance_final.csv",
        },
        "limitations": [
            "Predictions are statistical risk estimates, not medical diagnoses.",
            "Data1 was not selected for deployment because unusually strong results may indicate leakage risk.",
            "Dataset size and representativeness are limited.",
            "Real-world performance depends on input data quality.",
        ],
    }
    write_json(MODELS_DIR / "model_metadata.json", to_jsonable(metadata))


if __name__ == "__main__":
    run_training()
