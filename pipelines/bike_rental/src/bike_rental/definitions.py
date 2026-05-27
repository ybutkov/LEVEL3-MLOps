from dagster import Definitions, definitions, FilesystemIOManager

from bike_rental.defs.assets.data.raw import (
    registered_rentals_raw,
    direct_pickups_raw,
    weather_raw,
    holidays_raw,
)
from bike_rental.defs.assets.data.rentals import (
    rentals_split,
    hourly_rentals,
)
from bike_rental.defs.assets.data.weather import (
    clean_weather,
    weather_split,
)
from bike_rental.defs.assets.data.holidays import holidays_split
from bike_rental.defs.assets.ml import final_dataset
from bike_rental.defs.asset_checks.contracts import (
    registered_rentals_raw_contract,
    direct_pickups_raw_contract,
    weather_raw_contract,
    holidays_raw_contract,
    final_dataset_no_nulls,
)
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
            "bike_data":     BikeDataDirResource(base_path="../../data/raw/bike_rental"),
            "io_manager":    FilesystemIOManager(base_dir=".dagster_storage"),
            "csv_io":        CSVIOManager(base_dir="../../data/processed/bike_rental"),
            "quarantine_io": CSVIOManager(base_dir="../../data/quarantine/bike_rental"),
        },
    )
