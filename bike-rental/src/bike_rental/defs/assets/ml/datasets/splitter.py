"""Chronological train/validation/test split shared by model assets.

Split fractions come from the global `split` setting (recipe_loader.load_split),
so every model uses the same train/val/test boundaries — keeping their validation
metrics comparable. When train_frac + val_frac == 1 there is no test set: val runs
to the end and the test split is empty.
"""

import dagster as dg
import numpy as np
import pandas as pd

from bike_rental.defs.assets.ml.recipes.recipe_config import RecipeConfig


class DatasetSplitter():
    def __init__(self, recipe_config: RecipeConfig):
        self.recipe_config = recipe_config

        self.train_border = float(self.recipe_config.get_recipe("split")["train_frac"])
        self.val_border = float(self.recipe_config.get_recipe("split")["val_frac"])
        valid = (
            isinstance(self.train_border, (int, float))
            and isinstance(self.val_border, (int, float))
            and 0 < self.train_border < 1
            and 0 < self.val_border
            and round(self.train_border + self.val_border, 6) <= 1
        )
        if not valid:
            raise Exception(
                "split: need 0<train_frac<1, 0<val_frac, train_frac+val_frac<=1; "
                f"got {self.train_border, self.val_border}"
            )

    def _cut_points(self, timestamps: np.ndarray) -> tuple:
        """Boundary timestamps for train|val and val|test.

        The val|test boundary is None when train_frac + val_frac == 1 (no test set):
        its index would be `len(timestamps)`, i.e. past the last timestamp.
        """
        n = len(timestamps)
        cut1 = timestamps[int(n * self.train_border)]
        i2 = int(n * (self.train_border + self.val_border))
        cut2 = timestamps[i2] if i2 < n else None
        return cut1, cut2

    def split_frames(
        self, df: pd.DataFrame, time_feature: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        ds = df.sort_values(time_feature).reset_index(drop=True)
        timestamps = np.sort(ds[time_feature].unique())
        cut1, cut2 = self._cut_points(timestamps)

        train = ds[ds[time_feature] < cut1]
        if cut2 is None:
            val = ds[ds[time_feature] >= cut1]
            test = ds.iloc[0:0]
        else:
            val = ds[(ds[time_feature] >= cut1) & (ds[time_feature] < cut2)]
            test = ds[ds[time_feature] >= cut2]
        return train, val, test

    def describe_time_split(self, df: pd.DataFrame, time_feature: str) -> dict:
        """Build Dagster metadata describing how the chronological split was made.

        Records the strategy, boundary timestamps and per-split row counts so the
        split is auditable from the asset's materialization in the UI.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataset being split.
        time_feature : str
            Column holding the timestamp used for the split.

        Returns
        -------
        dict
            Mapping of metadata keys to ``dagster.MetadataValue`` (strategy, train/val
            boundaries, per-split row counts).
        """
        ds = df.sort_values(time_feature)
        timestamps = np.sort(ds[time_feature].unique())
        cut1, cut2 = self._cut_points(timestamps)

        n_train = int((ds[time_feature] < cut1).sum())
        if cut2 is None:
            n_val = int((ds[time_feature] >= cut1).sum())
            n_test = 0
            val_end = "— (no test set)"
        else:
            n_val = int(((ds[time_feature] >= cut1) & (ds[time_feature] < cut2)).sum())
            n_test = int((ds[time_feature] >= cut2).sum())
            val_end = str(cut2)
        test_frac = max(0.0, round(1 - self.train_border - self.val_border, 4))

        return {
            "split_strategy": dg.MetadataValue.text(
                f"chronological {self.train_border:.0%}/{self.val_border:.0%}/{test_frac:.0%} "
                f"by unique {time_feature}"
            ),
            "split_train_end": dg.MetadataValue.text(str(cut1)),
            "split_val_end": dg.MetadataValue.text(val_end),
            "split_rows": dg.MetadataValue.text(
                f"train={n_train}, val={n_val}, test={n_test}"
            ),
        }
