import dagster as dg
import pandas as pd


@dg.asset(group_name="primary", kinds={"pandas"})
def hourly_rentals(
    context: dg.AssetExecutionContext,
    rentals_typed: pd.DataFrame,
) -> pd.DataFrame:


    reg_counts = (
        rentals_typed[rentals_typed["is_registered"]]
        .groupby(["datetime_hourly", "location_id"]).size()
        .reset_index(name="registered_rentals")
    )
    direct_counts = (
        rentals_typed[~rentals_typed["is_registered"]]
        .groupby(["datetime_hourly", "location_id"]).size()
        .reset_index(name="direct_pickups")
    )

    all_hours = sorted(rentals_typed["datetime_hourly"].unique())
    all_locations = sorted(rentals_typed["location_id"].unique())
    full_grid = pd.merge(
        pd.DataFrame({"datetime_hourly": all_hours}),
        pd.DataFrame({"location_id": all_locations}),
        how="cross",
    )

    hourly_rentals = (
        full_grid
        .merge(reg_counts, on=["datetime_hourly", "location_id"], how="left")
        .merge(direct_counts, on=["datetime_hourly", "location_id"], how="left")
        .fillna(0)
        .astype({"registered_rentals": int, "direct_pickups": int})
    )
    hourly_rentals["total_rentals"] = (
        hourly_rentals["registered_rentals"] + hourly_rentals["direct_pickups"]
    )

    context.log.info(f"Shape: {hourly_rentals.shape}")
    context.log.info(hourly_rentals.head().to_string(index=False))
    context.log.info(hourly_rentals.info())

    dt = hourly_rentals["datetime_hourly"]
    hourly_rentals["month"]       = dt.dt.month
    hourly_rentals["hour_of_day"] = dt.dt.hour
    hourly_rentals["day_of_week"] = dt.dt.dayofweek
    hourly_rentals["is_weekend"]  = (dt.dt.dayofweek >= 5).astype(int)

    return hourly_rentals
