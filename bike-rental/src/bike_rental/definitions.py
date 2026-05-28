"""Dagster Definitions: wires assets, checks, resources, and IO managers."""

from dagster import Definitions, FilesystemIOManager, definitions

from bike_rental.config import AppConfig
from bike_rental.defs.asset_checks.contracts import (
    direct_pickups_raw_contract,
    final_dataset_no_nulls,
    holidays_raw_contract,
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
from bike_rental.defs.assets.data.rentals import (
    hourly_rentals,
    rentals_split,
)
from bike_rental.defs.assets.data.weather import (
    clean_weather,
    weather_split,
)
from bike_rental.defs.assets.ml import final_dataset
from bike_rental.defs.io_managers.csv_io import CSVIOManager
from bike_rental.defs.resources.bike_data import BikeDataDirResource


@definitions
def defs() -> Definitions:
    """Build the Dagster Definitions for the bike rental pipeline."""
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
            final_dataset,
        ],
        asset_checks=[
            registered_rentals_raw_contract,
            direct_pickups_raw_contract,
            weather_raw_contract,
            holidays_raw_contract,
            final_dataset_no_nulls,
        ],
        resources={
            "bike_data": BikeDataDirResource(base_path=cfg.source_dir),
            "io_manager": FilesystemIOManager(base_dir=cfg.dagster_storage_dir),
            "csv_io": CSVIOManager(base_dir=cfg.processed_dir),
            "quarantine_io": CSVIOManager(base_dir=cfg.quarantine_dir),
        },
    )
