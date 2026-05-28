"""Holidays intermediate layer: type and split the holiday calendar."""

import dagster as dg
import pandas as pd

from bike_rental.defs import schemas
from bike_rental.defs.schemas import ParsedDatetime

TYPED_OUT = "holidays_typed"
QUARANTINE_OUT = "holidays_quarantine"


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
def holidays_split(holidays_raw: pd.DataFrame):
    """Type-check the holiday calendar and keep one date per holiday.

    Rows with an unparseable date go to the quarantine output.
    """
    log = dg.get_dagster_logger()

    raw_cols = list(holidays_raw.columns)

    df = holidays_raw.copy()
    df["parsed_dt"] = schemas.parse_datetime(df["date"])

    clean, quarantine = schemas.validate_and_split(df, ParsedDatetime, raw_cols)
    if not quarantine.empty:
        log.warning("Quarantined %d of %d rows (unparseable datetime)", len(quarantine), len(df))

    typed = clean[["parsed_dt", "holiday"]].rename(columns={"parsed_dt": "date"})
    typed["date"] = typed["date"].dt.date

    yield dg.Output(typed, output_name=TYPED_OUT, metadata={"row_count": len(typed)})
    yield dg.Output(quarantine, output_name=QUARANTINE_OUT, metadata={"row_count": len(quarantine)})
