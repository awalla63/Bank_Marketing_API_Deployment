"""Generates notebooks/model_development.ipynb programmatically via nbformat.
Run once: python scripts/make_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md("""# Bank Marketing Subscription-Propensity Pipeline — Model Development

Interactive companion to `build_pipeline.py`. Explores the data, builds the
`ColumnTransformer(MeanTargetEncoder + StandardScaler) -> LogisticRegression`
pipeline using the shared functions in `build_pipeline.py` / `pipeline_def.py`,
fits it, evaluates it, and inspects the learned state that makes this a *real*
fitted pipeline (not something that could be rebuilt from scratch at boot).
""")

code("""import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from build_pipeline import (
    load_data, build_pipeline, evaluate,
    CAT_COLUMNS, NUM_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN,
)
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", 50)
""")

md("## 1. Load & inspect raw data")

code("""df = load_data("../data/bankmarketing.csv")
print(df.shape)
df.head()
""")

code("""df[TARGET_COLUMN].value_counts(normalize=True)
""")

code("""df[NUM_COLUMNS].describe()
""")

md("""`duration` is deliberately **excluded** from `FEATURE_COLUMNS`: it's only known
*after* a call ends (duration≈0 almost always implies `y=no`), so it leaks the
outcome and isn't a legitimate input for a pre-call propensity estimate.
""")

code("""FEATURE_COLUMNS
""")

md("## 2. Quick EDA — subscription rate by category")

code("""fig, axes = plt.subplots(2, 5, figsize=(20, 7))
for ax, col in zip(axes.ravel(), CAT_COLUMNS):
    rates = df.groupby(col)[TARGET_COLUMN].apply(lambda s: (s == "yes").mean()).sort_values()
    rates.plot(kind="barh", ax=ax, title=col)
plt.tight_layout()
plt.show()
""")

md("## 3. Train / test split")

code("""X = df[FEATURE_COLUMNS]
y = df["y_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train.shape, X_test.shape, y_train.mean(), y_test.mean()
""")

md("""## 4. Build & fit the pipeline

`build_pipeline()` returns an **unfit** `sklearn.Pipeline`. Calling `.fit()` is
what produces the learned state (the `MeanTargetEncoder.mapping_` dict and the
`LogisticRegression` coefficients) that gets serialized into the `.joblib`
bundle. An unfit pipeline (or one rebuilt fresh at API boot without re-fitting)
has none of that state.
""")

code("""pipeline = build_pipeline()
pipeline
""")

code("""pipeline.fit(X_train, y_train)
""")

md("## 5. Evaluate on held-out test data")

code("""metrics = evaluate(pipeline, X_test, y_test)
print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
print(pd.DataFrame(metrics["report"]).T)
""")

code("""from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
plt.show()
""")

md("""## 6. Inspect the custom transformer's learned state

This is the "wrong if rebuilt from scratch at boot" part: `mapping_` is a
per-category smoothed target rate learned from the *training* split. A freshly
constructed `MeanTargetEncoder` has no `mapping_` at all.
""")

code("""encoder = pipeline.named_steps["features"].named_transformers_["cat"]
print("global_mean_:", encoder.global_mean_)
encoder.mapping_["poutcome"]
""")

code("""encoder.mapping_["month"]
""")

code("""clf = pipeline.named_steps["clf"]
feature_names = list(encoder.get_feature_names_out()) + NUM_COLUMNS
coefs = pd.Series(clf.coef_[0], index=feature_names).sort_values()
coefs.plot(kind="barh", figsize=(6, 6), title="Logistic regression coefficients")
plt.tight_layout()
plt.show()
""")

md("## 7. Sanity check a single prediction (mirrors what the API will do)")

code("""sample = X_test.iloc[[0]]
proba = pipeline.predict_proba(sample)[0, 1]
print(sample.to_dict(orient="records")[0])
print(f"Predicted subscribe probability: {proba:.4f} (actual y={y_test.iloc[0]})")
""")

md("""## 8. Dump the bundle

Equivalent to running `python build_pipeline.py` from the project root — this
cell exists so you can iterate on the pipeline here and re-dump without leaving
the notebook.
""")

code("""import joblib, sklearn, datetime as dt

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

# Uncomment to overwrite the shipped artifact from the notebook:
# joblib.dump(bundle, "../pipeline.joblib")
bundle["metadata"]
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

import pathlib
out = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "model_development.ipynb"
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Wrote {out}")
