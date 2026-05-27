"""Raw layer — thin loaders for every source CSV.

No domain logic here, just plumbing: which file is read into which asset.
Domain-specific transformations start at the `intermediate` layer inside each
domain package (weather/, rentals/, holidays/).
"""
import dagster as dg
import pandas as pd

from bike_rental.defs.resources.bike_data import BikeDataDirResource


@dg.asset(group_name="raw", kinds={"pandas"})
def registered_rentals_raw(bike_data: BikeDataDirResource) -> pd.DataFrame:
    return bike_data.load_csv("registered_bike_rentals.csv")


@dg.asset(group_name="raw", kinds={"pandas"})
def direct_pickups_raw(bike_data: BikeDataDirResource) -> pd.DataFrame:
    return bike_data.load_csv("direct_pickup_bike_rentals.csv")


@dg.asset(group_name="raw", kinds={"pandas"})
def weather_raw(bike_data: BikeDataDirResource) -> pd.DataFrame:
    return bike_data.load_csv("weather.csv")


@dg.asset(group_name="raw", kinds={"pandas"})
def holidays_raw(bike_data: BikeDataDirResource) -> pd.DataFrame:
    return bike_data.load_csv("holidays.csv")
