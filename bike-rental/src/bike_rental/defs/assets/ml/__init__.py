"""ML feature assets: the final dataset."""

from bike_rental.defs.assets.ml.dataset_linear_hourly import linear_dataset_hourly
from bike_rental.defs.assets.ml.dataset_tree_hourly import tree_dataset_hourly
from bike_rental.defs.assets.ml.feature import hourly_by_location, hourly_total
from bike_rental.defs.assets.ml.model_hgb_hourly import hgb_hourly
from bike_rental.defs.assets.ml.model_linear_hourly import linear_hourly
from bike_rental.defs.assets.ml.model_rf_hourly import rf_hourly

__all__ = [
    "hourly_by_location",
    "hourly_total",
    "linear_dataset_hourly",
    "tree_dataset_hourly",
    "linear_hourly",
    "rf_hourly",
    "hgb_hourly",
]
