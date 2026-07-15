"""Re-evaluate the saved holdout model on its holdout split."""

from __future__ import annotations

import json

import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from modeling_common import MODELS_DIR, REPORTS_DIR, load_dataset, split_xy, to_jsonable, write_json


def main() -> None:
    metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    dataset_name = metadata["dataset"]
    threshold = float(metadata["selected_threshold"])
    test_ids = set(int(x) for x in metadata["holdout_test_pregnancy_ids"])

    df = load_dataset(dataset_name)
    X, y, ids = split_xy(df, dataset_name)
    mask = ids.isin(test_ids)
    X_test = X.loc[mask]
    y_test = y.loc[mask]

    model = joblib.load(MODELS_DIR / "final_holdout_model.pkl")
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    metrics = {
        "dataset": dataset_name,
        "model_name": metadata["model_name"],
        "threshold": threshold,
        "test_rows": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    write_json(REPORTS_DIR / "final_model_recheck_metrics.json", to_jsonable(metrics))


if __name__ == "__main__":
    main()
