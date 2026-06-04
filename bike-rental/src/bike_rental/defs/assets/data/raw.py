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
    """Load the registered (pre-booked) rentals CSV.

    Parameters
    ----------
    bike_data : BikeDataDirResource
        Resource pointing at the raw data directory.

    Returns
    -------
    pandas.DataFrame
        Raw registered-rentals rows, unmodified.
    """
    return bike_data.load_csv("registered_bike_rentals.csv")


@dg.asset(group_name="raw", kinds={"pandas"})
def direct_pickups_raw(bike_data: BikeDataDirResource) -> pd.DataFrame:
    """Load the direct-pickup rentals CSV.

    Parameters
    ----------
    bike_data : BikeDataDirResource
        Resource pointing at the raw data directory.

    Returns
    -------
    pandas.DataFrame
        Raw direct-pickup rows, unmodified.
    """
    return bike_data.load_csv("direct_pickup_bike_rentals.csv")


@dg.asset(group_name="raw", kinds={"pandas"})
def weather_raw(bike_data: BikeDataDirResource) -> pd.DataFrame:
    """Load the weather CSV.

    Parameters
    ----------
    bike_data : BikeDataDirResource
        Resource pointing at the raw data directory.

    Returns
    -------
    pandas.DataFrame
        Raw weather rows, unmodified.
    """
    return bike_data.load_csv("weather.csv")


@dg.asset(group_name="raw", kinds={"pandas"})
def holidays_raw(bike_data: BikeDataDirResource) -> pd.DataFrame:
    """Load the holiday calendar CSV.

    Parameters
    ----------
    bike_data : BikeDataDirResource
        Resource pointing at the raw data directory.

    Returns
    -------
    pandas.DataFrame
        Raw holiday-calendar rows, unmodified.
    """
    return bike_data.load_csv("holidays.csv")
