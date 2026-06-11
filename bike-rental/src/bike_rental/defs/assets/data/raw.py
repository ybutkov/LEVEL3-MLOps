"""Raw layer — thin loaders for every source CSV, generated from a factory.

No domain logic here, just plumbing: which file is read into which asset.
Domain-specific transformations start at the `intermediate` layer inside each
domain package (weather/, rentals/, holidays/).
"""

import dagster as dg
import pandas as pd

from bike_rental.defs.resources.source import SourceResource


def _raw_asset(asset_name: str, filename: str):
    """Build a raw-loader asset that reads ``filename`` via the source resource."""

    @dg.asset(name=asset_name, group_name="raw", kinds={"pandas"})
    def _asset(source: SourceResource) -> pd.DataFrame:
        """Load a source CSV unmodified."""
        return source.load_csv(filename)

    return _asset


registered_rentals_raw = _raw_asset(
    asset_name="registered_rentals_raw", filename="registered_bike_rentals.csv"
)
direct_pickups_raw = _raw_asset(
    asset_name="direct_pickups_raw", filename="direct_pickup_bike_rentals.csv"
)
weather_raw = _raw_asset(asset_name="weather_raw", filename="weather.csv")
holidays_raw = _raw_asset(asset_name="holidays_raw", filename="holidays.csv")
