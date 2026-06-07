import dagster as dg
import pandas as pd

from bike_rental.defs.assets.ml.recipe.recipe_config import RecipeConfig
from bike_rental.defs.assets.ml.training.split import DatasetSplitter

TIME_KEY = "datetime_hourly"


@dg.multi_asset(
    group_name="dataset_splits",
    outs={
        "linear_dataset_hourly_train": dg.AssetOut(io_manager_key="csv_io"),
        "linear_dataset_hourly_val":   dg.AssetOut(io_manager_key="csv_io"),
        "linear_dataset_hourly_test":  dg.AssetOut(io_manager_key="csv_io"),
    },
)
def linear_dataset_splits(linear_dataset_hourly: pd.DataFrame, recipe_config: RecipeConfig):
    splitter = DatasetSplitter(recipe_config)
    train, val, test = splitter.split_frames(linear_dataset_hourly, TIME_KEY)
    return train, val, test

@dg.multi_asset(
    group_name="dataset_splits",
    outs={
        "tree_dataset_hourly_train": dg.AssetOut(io_manager_key="csv_io"),
        "tree_dataset_hourly_val":   dg.AssetOut(io_manager_key="csv_io"),
        "tree_dataset_hourly_test":  dg.AssetOut(io_manager_key="csv_io"),
    },
)
def tree_dataset_splits(tree_dataset_hourly: pd.DataFrame, recipe_config: RecipeConfig):
    splitter = DatasetSplitter(recipe_config)
    train, val, test = splitter.split_frames(tree_dataset_hourly, TIME_KEY)
    return train, val, test