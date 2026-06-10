"""Dagster Definitions: wires assets, checks, resources, and IO managers."""

from dagster import Definitions, FilesystemIOManager, definitions

from bike_rental.config import AppConfig
from bike_rental.defs.asset_checks.contracts import (
    direct_pickups_raw_contract,
    holidays_raw_contract,
    hourly_by_location_no_nulls,
    registered_rentals_raw_contract,
    weather_raw_contract,
)
from bike_rental.defs.assets.data.holidays import holidays_split
from bike_rental.defs.assets.data.raw import (
    direct_pickups_raw,
    holidays_raw,
    registered_rentals_raw,
    weather_raw,
)
from bike_rental.defs.assets.data.rentals import hourly_rentals, rentals_split
from bike_rental.defs.assets.data.weather import clean_weather, weather_split
from bike_rental.defs.assets.ml.base_dataset import hourly_by_location, hourly_total
from bike_rental.defs.assets.ml.datasets.linear_hourly import linear_dataset_hourly
from bike_rental.defs.assets.ml.datasets.tree_hourly import tree_dataset_hourly
from bike_rental.defs.assets.ml.datasets.split_datasets import linear_dataset_splits, tree_dataset_splits
from bike_rental.defs.assets.ml.models.hgb_hourly import hgb_hourly
from bike_rental.defs.assets.ml.models.linear_hourly import linear_hourly
from bike_rental.defs.assets.ml.models.rf_hourly import rf_hourly
from bike_rental.defs.assets.ml.models.promotion import champion
from bike_rental.defs.io_managers.csv_io import CSVIOManager
from bike_rental.defs.resources.source import SourceDirResource
from bike_rental.defs.resources.experiment import ExperimentConfig
from bike_rental.defs.resources.experiment_tracker import MlflowExperimentTracker
from bike_rental.defs.assets.ml.recipes.recipe_config import RecipeConfig


@definitions
def defs() -> Definitions:
    """Build the Dagster ``Definitions`` for the bike-rental pipeline.

    Returns
    -------
    dagster.Definitions
        All assets, asset checks, resources and IO managers wired together.
    """
    cfg = AppConfig.load()
    return Definitions(
        assets=[
            registered_rentals_raw,
            direct_pickups_raw,
            weather_raw,
            holidays_raw,
            rentals_split,
            weather_split,
            holidays_split,
            hourly_rentals,
            clean_weather,
            hourly_by_location,
            hourly_total,
            linear_dataset_hourly,
            tree_dataset_hourly,
            linear_dataset_splits,
            tree_dataset_splits,
            linear_hourly,
            rf_hourly,
            hgb_hourly,
            champion,
        ],
        asset_checks=[
            registered_rentals_raw_contract,
            direct_pickups_raw_contract,
            weather_raw_contract,
            holidays_raw_contract,
            hourly_by_location_no_nulls,
        ],
        resources={
            "source": SourceDirResource(base_path=cfg.source_dir),
            "io_manager": FilesystemIOManager(base_dir=cfg.dagster_storage_dir),
            "csv_io": CSVIOManager(base_dir=cfg.processed_dir),
            "quarantine_io": CSVIOManager(base_dir=cfg.quarantine_dir),
            "base_config": AppConfig.load(),
            "recipe_config": RecipeConfig(),
            "experiment_config": ExperimentConfig(),
            "experiment_tracker": MlflowExperimentTracker(
                tracking_uri=cfg.mlflow.tracking_uri,
                experiment_name=cfg.mlflow.experiment_name,
                registered_model=cfg.mlflow.registered_model,
            ),
        },
    )
