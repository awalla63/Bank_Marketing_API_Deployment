"""Custom transformer(s) used by the bank-marketing subscription-propensity pipeline.

This module must be importable from both the FastAPI service (serve.py) and the
Modal deployment (modal_serve.py) using the exact same class definitions that were
present when the pipeline was fit and pickled with joblib. If this file changes
shape (renamed attributes, different __init__ signature, etc.) the joblib artifact
will fail to unpickle correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed mean-target (likelihood) encoder for categorical columns.

    For every category value seen during ``fit``, this learns the smoothed mean of
    the training target ``y`` conditioned on that category. That learned mapping
    (``self.mapping_``) plus the overall training target rate (``self.global_mean_``)
    is exactly the kind of "learned state" that only exists after fitting on real
    data -- a freshly-constructed, unfit instance of this class has no mapping and
    cannot reproduce the pipeline's behavior, which is the point of shipping a
    fitted ``.joblib`` artifact instead of rebuilding the pipeline at boot.

    Unseen categories at transform time fall back to ``self.global_mean_``.
    """

    def __init__(self, columns=None, smoothing: float = 10.0):
        # __init__ only assigns its arguments (scikit-learn convention).
        self.columns = columns
        self.smoothing = smoothing

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("MeanTargetEncoder requires y at fit time.")

        columns = self.columns
        if columns is None:
            columns = list(X.columns) if isinstance(X, pd.DataFrame) else list(range(np.asarray(X).shape[1]))

        if isinstance(X, pd.DataFrame):
            X_df = X.reset_index(drop=True)
        else:
            X_df = pd.DataFrame(np.asarray(X), columns=columns)

        y_series = pd.Series(np.asarray(y).ravel()).reset_index(drop=True)

        self.global_mean_ = float(y_series.mean())
        self.mapping_ = {}
        for col in columns:
            stats = pd.DataFrame({col: X_df[col].values, "_y": y_series.values}).groupby(col)["_y"].agg(
                ["mean", "count"]
            )
            smoothed = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (
                stats["count"] + self.smoothing
            )
            self.mapping_[col] = smoothed.to_dict()

        self.columns_ = list(columns)
        self.n_features_in_ = len(self.columns_)
        return self

    def transform(self, X):
        if not hasattr(self, "mapping_"):
            raise RuntimeError("MeanTargetEncoder.transform called before fit().")

        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(np.asarray(X), columns=self.columns_)

        out = np.empty((len(X_df), len(self.columns_)), dtype=float)
        for i, col in enumerate(self.columns_):
            mapping = self.mapping_[col]
            out[:, i] = X_df[col].map(mapping).fillna(self.global_mean_).to_numpy(dtype=float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_target_enc" for c in self.columns_])
