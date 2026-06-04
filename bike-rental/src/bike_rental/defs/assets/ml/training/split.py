"""Chronological train/validation/test split shared by model assets.

Split fractions come from the global `split` setting (recipe_loader.load_split),
so every model uses the same train/val/test boundaries — keeping their validation
metrics comparable. When train_frac + val_frac == 1 there is no test set: val runs
to the end and the test split is empty.
"""

import dagster as dg
import numpy as np
import pandas as pd

from bike_rental.defs.assets.ml.recipe.loader import load_split

_SPLIT = load_split()
TRAIN_FRAC = _SPLIT["train_frac"]
VAL_FRAC = _SPLIT["val_frac"]


def _cut_points(timestamps: np.ndarray) -> tuple:
    """Boundary timestamps for train|val and val|test.

    The val|test boundary is None when train_frac + val_frac == 1 (no test set):
    its index would be `len(timestamps)`, i.e. past the last timestamp.
    """
    n = len(timestamps)
    cut1 = timestamps[int(n * TRAIN_FRAC)]
    i2 = int(n * (TRAIN_FRAC + VAL_FRAC))
    cut2 = timestamps[i2] if i2 < n else None
    return cut1, cut2


def train_validate_test_time_split(
    df: pd.DataFrame, time_feature: str, features: list[str], target: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split chronologically into train/val/test by unique timestamps.

    With no test boundary (train+val == 1), val extends to the end and test is empty.
    """
    ds = df.sort_values(time_feature).reset_index(drop=True)
    timestamps = np.sort(ds[time_feature].unique())
    cut1, cut2 = _cut_points(timestamps)

    train = ds[ds[time_feature] < cut1]
    if cut2 is None:
        val = ds[ds[time_feature] >= cut1]
        test = ds.iloc[0:0]
    else:
        val = ds[(ds[time_feature] >= cut1) & (ds[time_feature] < cut2)]
        test = ds[ds[time_feature] >= cut2]

    X_train, y_train = train[features], train[target]
    X_val, y_val = val[features], val[target]
    X_test, y_test = test[features], test[target]

    return X_train, y_train, X_val, y_val, X_test, y_test


def describe_time_split(df: pd.DataFrame, time_feature: str) -> dict:
    """Dagster metadata describing how the chronological split was made.

    Logs the strategy, boundary timestamps, and per-split row counts so the split
    is auditable from the asset's materialization in the UI.
    """
    ds = df.sort_values(time_feature)
    timestamps = np.sort(ds[time_feature].unique())
    cut1, cut2 = _cut_points(timestamps)

    n_train = int((ds[time_feature] < cut1).sum())
    if cut2 is None:
        n_val = int((ds[time_feature] >= cut1).sum())
        n_test = 0
        val_end = "— (no test set)"
    else:
        n_val = int(((ds[time_feature] >= cut1) & (ds[time_feature] < cut2)).sum())
        n_test = int((ds[time_feature] >= cut2).sum())
        val_end = str(cut2)
    test_frac = max(0.0, round(1 - TRAIN_FRAC - VAL_FRAC, 4))

    return {
        "split_strategy": dg.MetadataValue.text(
            f"chronological {TRAIN_FRAC:.0%}/{VAL_FRAC:.0%}/{test_frac:.0%} "
            f"by unique {time_feature}"
        ),
        "split_train_end": dg.MetadataValue.text(str(cut1)),
        "split_val_end": dg.MetadataValue.text(val_end),
        "split_rows": dg.MetadataValue.text(
            f"train={n_train}, val={n_val}, test={n_test}"
        ),
    }
