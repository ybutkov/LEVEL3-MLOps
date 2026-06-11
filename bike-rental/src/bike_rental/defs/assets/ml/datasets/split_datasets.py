"""Chronological train/val/test split of the shared feature table."""

import dagster as dg
import pandas as pd

from bike_rental.defs.assets.ml.datasets.splitter import DatasetSplitter
from bike_rental.defs.assets.ml.recipes.recipe_config import RecipeConfig

TIME_KEY = "datetime_hourly"


@dg.multi_asset(
    group_name="dataset_splits",
    outs={
        "feature_rentals_hourly_train": dg.AssetOut(io_manager_key="csv_io"),
        "feature_rentals_hourly_val":   dg.AssetOut(io_manager_key="csv_io"),
        "feature_rentals_hourly_test":  dg.AssetOut(io_manager_key="csv_io"),
    },
)
def feature_rentals_hourly_splits(
    feature_rentals_hourly: pd.DataFrame, recipe_config: RecipeConfig
):
    """Split the feature table into chronological train/val/test frames."""
    splitter = DatasetSplitter(recipe_config)
    train, val, test = splitter.split_frames(feature_rentals_hourly, TIME_KEY)
    return train, val, test
