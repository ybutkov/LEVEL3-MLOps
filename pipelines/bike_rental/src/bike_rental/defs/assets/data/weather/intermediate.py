import dagster as dg
import numpy as np
import pandas as pd

from bike_rental.defs.schemas import WEATHER_CONDITIONS

from bike_rental.defs import schemas
from bike_rental.defs.schemas import ParsedDatetime


TYPED_OUT      = "weather_typed"
QUARANTINE_OUT = "weather_quarantine"

WEATHER_CONDITIONS_DTYPE = pd.CategoricalDtype(
    categories=WEATHER_CONDITIONS,
    ordered=True,
)

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
def weather_split(weather_raw: pd.DataFrame):
    raw_cols = list(weather_raw.columns)

    df = weather_raw.copy()
    df["parsed_dt"] = schemas.parse_datetime(df["datetime"])

    typed, quarantine = schemas.validate_and_split(df, ParsedDatetime, raw_cols)

    typed["datetime_hourly"] = typed["parsed_dt"].dt.floor("h")
    typed = typed.drop(columns=["id", "datetime", "parsed_dt"])

    # known sensor outage: humidity sometimes reports 0% (physically impossible).
    humidity_zeros = int((typed["humidity"] == 0).sum())
    # log humidity_zeros

    typed["humidity"] = typed["humidity"].replace(0, np.nan).interpolate("linear")

    normalized = typed["conditions"].astype(str).str.lower().str.strip()
    typed["conditions"] = normalized.astype(WEATHER_CONDITIONS_DTYPE)

    yield dg.Output(typed,      output_name=TYPED_OUT,      metadata={"row_count": len(typed), "humidity_zeros": humidity_zeros})
    yield dg.Output(quarantine, output_name=QUARANTINE_OUT, metadata={"row_count": len(quarantine)})
