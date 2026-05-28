"""Feature layer: the final ML-ready dataset."""

import dagster as dg
import pandas as pd


@dg.asset(group_name="feature", io_manager_key="csv_io", kinds={"pandas"})
def final_dataset(
    hourly_rentals: pd.DataFrame,
    clean_weather: pd.DataFrame,
    holidays_typed: pd.DataFrame,
) -> dg.MaterializeResult:
    """Join hourly rentals with weather and holiday flags.

    Weather is merged per hour, and a row is marked as a holiday when its date
    is in the holiday calendar. The IO manager writes the result to CSV.
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
