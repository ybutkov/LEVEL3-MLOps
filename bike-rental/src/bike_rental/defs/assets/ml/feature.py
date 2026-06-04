"""Feature layer: the final ML-ready dataset."""

import dagster as dg
import pandas as pd

from bike_rental.defs import schemas


@dg.asset(group_name="feature", io_manager_key="csv_io", kinds={"pandas"})
def hourly_by_location(
    hourly_rentals: pd.DataFrame,
    clean_weather: pd.DataFrame,
    holidays_typed: pd.DataFrame,
) -> dg.MaterializeResult:
    """Join hourly rentals with weather and per-row holiday flags.

    Weather is merged on the hourly timestamp; each row is flagged as a holiday
    when its date appears in the holiday calendar. Produces the per-(hour,
    location) feature table written to CSV by the IO manager.

    Parameters
    ----------
    hourly_rentals : pandas.DataFrame
        Hourly rental counts per location with calendar features.
    clean_weather : pandas.DataFrame
        Cleaned weather, one row per ``datetime_hourly``.
    holidays_typed : pandas.DataFrame
        Validated holiday calendar with a ``date`` column.

    Returns
    -------
    dagster.MaterializeResult
        Joined per-(hour, location) table with row-count and preview metadata.
    """
    df = hourly_rentals.merge(clean_weather, on="datetime_hourly", how="left")

    holiday_set = set(holidays_typed["date"])
    df["is_holiday"] = df["datetime_hourly"].dt.date.isin(holiday_set).astype(int)

    return dg.MaterializeResult(
        value=df,
        metadata={
            "row_count": dg.MetadataValue.int(len(df)),
            "preview": dg.MetadataValue.md(df.head().to_markdown()),
        },
    )


@dg.asset(group_name="feature", io_manager_key="csv_io", kinds={"pandas"})
def hourly_total(hourly_by_location: pd.DataFrame,) -> dg.MaterializeResult:
    """Aggregate the per-location feature table to city-wide hourly totals.

    Rentals are summed over all locations within each hour; the per-hour
    features (calendar, weather, holiday) are identical across locations and
    taken with ``first``. This is the base dataset all models train on.

    Parameters
    ----------
    hourly_by_location : pandas.DataFrame
        Per-(hour, location) feature table from :func:`hourly_by_location`.

    Returns
    -------
    dagster.MaterializeResult
        One row per hour (city-wide totals + per-hour features) with row-count
        and preview metadata.
    """
    df = (
        hourly_by_location
        .groupby("datetime_hourly")
        .agg(total_rentals=("total_rentals", "sum"),
             registered_rentals=("registered_rentals", "sum"),
             direct_pickups=("direct_pickups", "sum"),
             **{f: (f, "first") for f in schemas.HOURLY_FEATURES},
             )
        .reset_index()
    )

    return dg.MaterializeResult(
        value=df,
        metadata={
            "row_count": dg.MetadataValue.int(len(df)),
            "preview": dg.MetadataValue.md(df.head().to_markdown()),
        },
    )
