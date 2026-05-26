import dagster as dg
import pandas as pd


@dg.asset(group_name="output", io_manager_key="csv_io", kinds={"pandas"})
def final_dataset(
    hourly_rentals: pd.DataFrame,
    clean_weather: pd.DataFrame,
    clean_holidays: pd.DataFrame,
) -> dg.MaterializeResult:
    
    df = hourly_rentals.merge(clean_weather, on="datetime_hourly", how="left")

    holiday_set = set(clean_holidays["date"])
    df["is_holiday"] = df["datetime_hourly"].dt.date.isin(holiday_set).astype(int)

    return dg.MaterializeResult(
        value=df,
        metadata={
            "row_count": dg.MetadataValue.int(len(df)),
            "preview":   dg.MetadataValue.md(df.head().to_markdown()),
        },
    )
