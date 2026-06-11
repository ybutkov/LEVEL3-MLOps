"""Custom sklearn transformers used inside the recipe preprocessors."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicEncoder(BaseEstimator, TransformerMixin):
    """Encode each configured column as a sin/cos pair (learns nothing)."""

    def __init__(self, periods: dict[str, int]):
        self.periods = periods

    def fit(self, X, y=None):
        """Stateless — return self (nothing to learn)."""
        return self

    def transform(self, X):
        """Return the sin/cos encoding of each configured column."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=list(self.periods))
        out = {}
        for col, period in self.periods.items():
            angle = 2 * np.pi * X[col].to_numpy() / period
            out[f"{col}_sin"] = np.sin(angle)
            out[f"{col}_cos"] = np.cos(angle)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        """Output column names: ``{col}_sin`` / ``{col}_cos`` per configured column."""
        names = [f"{c}_{p}" for c in self.periods for p in ("sin", "cos")]
        return np.asarray(names, dtype=object)
