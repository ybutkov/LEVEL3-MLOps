"""ML assets: base datasets, per-model datasets/splits, and trained models."""

from bike_rental.defs.assets.ml.base_dataset import hourly_by_location, hourly_total
from bike_rental.defs.assets.ml.datasets.feature_rentals import feature_rentals_hourly
from bike_rental.defs.assets.ml.models.hourly_models import hgb_hourly, linear_hourly, rf_hourly

__all__ = [
    "hourly_by_location",
    "hourly_total",
    "feature_rentals_hourly",
    "linear_hourly",
    "rf_hourly",
    "hgb_hourly",
]
