import dagster as dg
import pandas as pd

from bike_rental.defs import schemas
from bike_rental.defs.schemas import ParsedDatetime


TYPED_OUT      = "rentals_typed"
QUARANTINE_OUT = "rentals_quarantine"

@dg.multi_asset(
    outs={
        TYPED_OUT: dg.AssetOut(
            group_name="intermediate",
            kinds={"pandas"},
        ),
        QUARANTINE_OUT: dg.AssetOut(
            group_name="quarantine",
            io_manager_key="quarantine_io",
            kinds={"pandas"},
        ),
    }
)
def rentals_split(
    registered_rentals_raw: pd.DataFrame,
    direct_pickups_raw: pd.DataFrame):

    raw_cols = list(registered_rentals_raw.columns) + ["is_registered"]

    df = pd.concat([
        registered_rentals_raw.assign(is_registered=True),
        direct_pickups_raw.assign(is_registered=False),
    ], ignore_index=True)
    df["parsed_dt"] = schemas.parse_datetime(df["datetime"])

    typed, quarantine = schemas.validate_and_split(df, ParsedDatetime, raw_cols)

    typed["datetime_hourly"] = typed["parsed_dt"].dt.floor("h")
    typed = typed.drop(columns=["id", "datetime", "parsed_dt"])

    yield dg.Output(typed,      output_name=TYPED_OUT,      metadata={"row_count": len(typed)})
    yield dg.Output(quarantine, output_name=QUARANTINE_OUT, metadata={"row_count": len(quarantine)})
