"""Data contract checks on raw sources.

Each check validates the raw DataFrame against its Pandera schema defined in
`defs/schemas.py`. `blocking=True` stops downstream materialization if the
contract is violated.
"""
import dagster as dg
import pandas as pd
import pandera as pa

from bike_rental.defs.assets.data.raw import (
    direct_pickups_raw,
    holidays_raw,
    registered_rentals_raw,
    weather_raw,
)
from bike_rental.defs.assets.ml import final_dataset
from bike_rental.defs.schemas import HolidaysRaw, RentalsRaw, WeatherRaw


def _validate_dataset(df: pd.DataFrame, schema) -> dg.AssetCheckResult:
    try:
        schema.validate(df, lazy=True)
        return dg.AssetCheckResult(
            passed=True,
            description="all checks passed",
            metadata={"row_count": len(df)},
        )
    except pa.errors.SchemaErrors as e:
        cases = e.failure_cases
        return dg.AssetCheckResult(
            passed=False,
            description=f"{len(cases)} failure(s); first: {cases.iloc[0].to_dict()}",
            metadata={
                "row_count":     len(df),
                "failure_count": len(cases),
                "failures":      dg.MetadataValue.md(cases.head(20).to_markdown()),
            },
        )


@dg.asset_check(asset=registered_rentals_raw, blocking=True)
def registered_rentals_raw_contract(registered_rentals_raw: pd.DataFrame) -> dg.AssetCheckResult:
    return _validate_dataset(registered_rentals_raw, RentalsRaw)


@dg.asset_check(asset=direct_pickups_raw, blocking=True)
def direct_pickups_raw_contract(direct_pickups_raw: pd.DataFrame) -> dg.AssetCheckResult:
    return _validate_dataset(direct_pickups_raw, RentalsRaw)


@dg.asset_check(asset=weather_raw, blocking=True)
def weather_raw_contract(weather_raw: pd.DataFrame) -> dg.AssetCheckResult:
    return _validate_dataset(weather_raw, WeatherRaw)


@dg.asset_check(asset=holidays_raw, blocking=True)
def holidays_raw_contract(holidays_raw: pd.DataFrame) -> dg.AssetCheckResult:
    return _validate_dataset(holidays_raw, HolidaysRaw)


@dg.asset_check(asset=final_dataset, blocking=False)
def final_dataset_no_nulls(final_dataset: pd.DataFrame) -> dg.AssetCheckResult:
    """Canary: no NaN anywhere in the published dataset.

    Currently safe because EDA §5.2 confirmed weather covers every rental
    hour. If a future weather drop has gaps during active hours, the left
    join would surface NaN here — this check fires before training consumes
    bad data. Non-blocking: dataset still publishes, but the alert is loud.
    """
    nulls = final_dataset.isnull().sum()
    bad_cols = nulls[nulls > 0]
    passed = bad_cols.empty
    return dg.AssetCheckResult(
        passed=passed,
        description=("all columns non-null" if passed
                     else f"nulls in: {bad_cols.to_dict()}"),
        metadata={
            "row_count":   len(final_dataset),
            "null_totals": dg.MetadataValue.md(
                bad_cols.to_markdown() if not bad_cols.empty else "_none_"
            ),
        },
    )
