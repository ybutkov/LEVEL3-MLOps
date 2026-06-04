"""Stateless feature engineering shared by the per-model dataset assets.

These transforms are row-wise and learn nothing from the data, so they are safe
to compute *before* the train/val/test split — no leakage. Keeping them in one
module means training (the dataset assets) and any future serving path compute
identical features from the same code, avoiding train/serve skew. Stateful
transforms (scaling, encoders that learn categories) deliberately do NOT live
here — they must be fit on train only, inside each model's sklearn Pipeline.
"""

import numpy as np
import pandas as pd


def add_cyclic_features(df: pd.DataFrame, periods: dict[str, int]) -> pd.DataFrame:
    """Encode each cyclic column as sin/cos and drop the raw source column.

    `periods` maps a column to its cycle length, e.g. ``{"hour_of_day": 24}``.
    For column ``c`` this adds ``c_sin`` / ``c_cos`` and removes ``c`` — the
    resulting frame shows exactly the engineered features the model consumes.
    """
    out = df.copy()
    for col, period in periods.items():
        angle = 2 * np.pi * out[col] / period
        out[f"{col}_sin"] = np.sin(angle)
        out[f"{col}_cos"] = np.cos(angle)
        out = out.drop(columns=col)
    return out
