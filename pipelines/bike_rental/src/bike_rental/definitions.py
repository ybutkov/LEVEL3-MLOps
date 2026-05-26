from dagster import Definitions, definitions, FilesystemIOManager

from bike_rental.defs.assets.raw_data import (
    registered_rentals_raw,
    direct_pickups_raw,
    weather_raw,
    holidays_raw,
)
from bike_rental.defs.assets.rentals import hourly_rentals
from bike_rental.defs.assets.weather import clean_weather
from bike_rental.defs.assets.final_asset import final_dataset
from bike_rental.defs.assets.holidays import clean_holidays
from bike_rental.defs.resources.bike_data import BikeDataDirResource
from bike_rental.defs.io_managers.csv_io import CSVIOManager



@definitions
def defs() -> Definitions:
    return Definitions(
        assets=[
            registered_rentals_raw,
            direct_pickups_raw,
            weather_raw,
            holidays_raw,
            hourly_rentals,
            clean_weather,
            clean_holidays,
            final_dataset,
        ],
        resources={
            "bike_data":  BikeDataDirResource(base_path="../../data/raw/bike_rental"),
            "io_manager": FilesystemIOManager(base_dir=".dagster_storage"),
            "csv_io":     CSVIOManager(base_dir="../../data/processed/bike_rental"),
        },
    )
