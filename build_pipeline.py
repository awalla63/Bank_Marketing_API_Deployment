"""Fit the bank-marketing subscription-propensity pipeline and dump a bundle.

Usage:
    python build_pipeline.py [--data data/bankmarketing.csv] [--out pipeline.joblib]

The dumped .joblib is a dict bundle (NOT a bare pipeline):
    {
        "pipeline": <fitted sklearn Pipeline>,
        "cat_columns": [...],
        "num_columns": [...],
        "target_rate": float,          # overall positive rate in training data
        "metadata": {
            "steps": [...],
            "built_at": "<ISO timestamp>",
            "sklearn_version": "<version pipeline was fit with>",
            "n_train_rows": int,
            "n_test_rows": int,
            "test_roc_auc": float,
            "model_type": "LogisticRegression",
            "custom_transformer": "MeanTargetEncoder",
            "target_column": "y",
            "description": "...",
        },
    }

This module exposes reusable functions (load_data, build_pipeline, evaluate) so it
can be imported from a notebook for interactive development, not just run as a
script.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline_def import MeanTargetEncoder

CAT_COLUMNS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

NUM_COLUMNS = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

# `duration` is intentionally excluded: it is only known *after* a call completes
# (duration=0 almost always implies y=no), so it leaks the outcome and is not a
# legitimate input for a pre-call propensity estimate.
FEATURE_COLUMNS = CAT_COLUMNS + NUM_COLUMNS
TARGET_COLUMN = "y"


def load_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["y_binary"] = (df[TARGET_COLUMN].str.strip().str.lower() == "yes").astype(int)
    return df


def build_pipeline() -> Pipeline:
    """Construct an *unfit* Pipeline. Calling this alone produces no learned
    state -- it must be .fit() on real data before it can be used, which is the
    whole point of shipping a fitted .joblib artifact rather than rebuilding at
    boot."""
    features = ColumnTransformer(
        transformers=[
            ("cat", MeanTargetEncoder(columns=CAT_COLUMNS, smoothing=10.0), CAT_COLUMNS),
            ("num", StandardScaler(), NUM_COLUMNS),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("features", features),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ]
    )
    return pipeline


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds, output_dict=True)
    return {"roc_auc": auc, "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/bankmarketing.csv")
    parser.add_argument("--out", default="pipeline.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = load_data(args.data)
    X = df[FEATURE_COLUMNS]
    y = df["y_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)
    print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Test accuracy: {metrics['report']['accuracy']:.4f}")

    bundle = {
        "pipeline": pipeline,
        "cat_columns": CAT_COLUMNS,
        "num_columns": NUM_COLUMNS,
        "target_rate": float(y_train.mean()),
        "metadata": {
            "steps": [name for name, _ in pipeline.steps],
            "built_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "n_train_rows": int(len(X_train)),
            "n_test_rows": int(len(X_test)),
            "test_roc_auc": float(metrics["roc_auc"]),
            "model_type": "LogisticRegression",
            "custom_transformer": "MeanTargetEncoder",
            "target_column": TARGET_COLUMN,
            "description": (
                "Bank telemarketing subscription-propensity pipeline: smoothed "
                "mean-target encoding (custom transformer) on categorical client "
                "fields + standardized macroeconomic/contact features, fed into "
                "a logistic regression that outputs a subscription probability."
            ),
        },
    }

    out_path = Path(args.out)
    joblib.dump(bundle, out_path)
    print(f"Wrote bundle to {out_path.resolve()} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
