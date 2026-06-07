"""ML assets: base datasets, per-model datasets/splits, and trained models."""

from bike_rental.defs.assets.ml.base_dataset import hourly_by_location, hourly_total
from bike_rental.defs.assets.ml.datasets.linear_hourly import linear_dataset_hourly
from bike_rental.defs.assets.ml.datasets.tree_hourly import tree_dataset_hourly
from bike_rental.defs.assets.ml.models.hgb_hourly import hgb_hourly
from bike_rental.defs.assets.ml.models.linear_hourly import linear_hourly
from bike_rental.defs.assets.ml.models.rf_hourly import rf_hourly

__all__ = [
    "hourly_by_location",
    "hourly_total",
    "linear_dataset_hourly",
    "tree_dataset_hourly",
    "linear_hourly",
    "rf_hourly",
    "hgb_hourly",
]
