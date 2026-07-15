"""Compare random under/over-sampling against the imbalanced baseline on Data2
using 5-fold out-of-fold scores (resampling applied to the training folds only)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from modeling_common import (
    CV_SPLITS,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    load_dataset,
    split_xy,
)

DATASET = "Data2"
THRESHOLD = 0.5


def build_model() -> Pipeline:
    best_params = json.loads(
        (MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8")
    ).get("best_params", {})
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )
    if best_params:
        pipe.set_params(**best_params)
    return pipe


def balance(X, y, mode):
    if mode is None:
        return X, y
    frame = X.copy()
    frame["_y"] = y.values
    counts = frame["_y"].value_counts()
    major = frame[frame["_y"] == counts.idxmax()]
    minor = frame[frame["_y"] == counts.idxmin()]
    if mode == "under":
        out = pd.concat([major.sample(len(minor), random_state=RANDOM_STATE), minor])
    else:
        out = pd.concat([major, minor.sample(len(major), replace=True, random_state=RANDOM_STATE)])
    out = out.sample(frac=1, random_state=RANDOM_STATE)
    return out.drop(columns="_y"), out["_y"]


def cv_oof(X, y, mode):
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.full(len(y), np.nan)
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = balance(X.iloc[train_idx], y.iloc[train_idx], mode)
        oof[val_idx] = build_model().fit(X_train, y_train).predict_proba(X.iloc[val_idx])[:, 1]
    pred = (oof >= THRESHOLD).astype(int)
    return {
        "roc_auc": roc_auc_score(y, oof),
        "pr_auc": average_precision_score(y, oof),
        "recall": recall_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
    }


def main():
    X, y, _ = split_xy(load_dataset(DATASET), DATASET)
    strategies = {"baseline": None, "undersample": "under", "oversample": "over"}
    rows = [{"strategy": name, **{k: round(v, 4) for k, v in cv_oof(X, y, mode).items()}}
            for name, mode in strategies.items()]
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(REPORTS_DIR / "balancing_experiment.csv", index=False)


if __name__ == "__main__":
    main()
