"""Weather intermediate layer: type, clean, and split weather rows."""

import dagster as dg
import numpy as np
import pandas as pd

from bike_rental.defs import schemas
from bike_rental.defs.schemas import WEATHER_CONDITIONS, ParsedDatetime

TYPED_OUT = "weather_typed"
QUARANTINE_OUT = "weather_quarantine"

WEATHER_CONDITIONS_DTYPE = pd.CategoricalDtype(
    categories=WEATHER_CONDITIONS,
    ordered=True,
)

# Feels-like temperature legitimately differs from the dry-bulb reading (wind
# chill / heat index), but only so far — the real spread in this data tops out
# at ~13°C. A larger gap is a sensor glitch (e.g. perceived ~0°C on a +24°C day,
# 2012-08-17), not a real feels-like value.
PERCEIVED_TEMP_MAX_GAP_C = 15


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
    """Type-check weather rows and clean known sensor issues.

    Unparseable timestamps are routed to the quarantine output. Two sensor
    glitches are repaired by interpolation: zero-humidity readings, and
    perceived-temperature values that decouple from the actual reading by more
    than ``PERCEIVED_TEMP_MAX_GAP_C``.

    Parameters
    ----------
    weather_raw : pandas.DataFrame
        Raw weather rows with a ``datetime`` column.

    Yields
    ------
    dagster.Output
        ``weather_typed`` — validated, cleaned weather keyed by
        ``datetime_hourly``; then ``weather_quarantine`` — rows dropped for an
        unparseable timestamp.
    """
    raw_cols = list(weather_raw.columns)

    log = dg.get_dagster_logger()

    df = weather_raw.copy()
    df["parsed_dt"] = schemas.parse_datetime(df["datetime"])

    typed, quarantine = schemas.validate_and_split(df, ParsedDatetime, raw_cols)
    if not quarantine.empty:
        log.warning("Quarantined %d of %d rows (unparseable datetime)", len(quarantine), len(df))

    typed["datetime_hourly"] = typed["parsed_dt"].dt.floor("h")
    typed = typed.drop(columns=["id", "datetime", "parsed_dt"])

    # known sensor outage: humidity sometimes reports 0% (physically impossible).
    humidity_zeros = int((typed["humidity"] == 0).sum())
    if humidity_zeros:
        log.warning("Humidity sensor reported 0%% in %d rows — interpolating", humidity_zeros)

    typed["humidity"] = typed["humidity"].replace(0, np.nan).interpolate("linear")

    # known sensor glitch: perceived temperature decouples from the actual reading
    # (e.g. ~0°C on a +24°C day, 2012-08-17). Blank the implausible gap and interpolate.
    perceived_gap = (typed["perceived_temperature_c"] - typed["temperature_c"]).abs()
    perceived_bad = int((perceived_gap > PERCEIVED_TEMP_MAX_GAP_C).sum())
    if perceived_bad:
        log.warning("Perceived temperature decoupled from actual in %d rows — interpolating", perceived_bad)

    typed.loc[perceived_gap > PERCEIVED_TEMP_MAX_GAP_C, "perceived_temperature_c"] = np.nan
    typed["perceived_temperature_c"] = typed["perceived_temperature_c"].interpolate("linear")

    normalized = typed["conditions"].astype(str).str.lower().str.strip()
    typed["conditions"] = normalized.astype(WEATHER_CONDITIONS_DTYPE)

    yield dg.Output(
        typed,
        output_name=TYPED_OUT,
        metadata={
            "row_count": len(typed),
            "humidity_zeros": humidity_zeros,
            "perceived_temp_fixed": perceived_bad,
        },
    )
    yield dg.Output(quarantine, output_name=QUARANTINE_OUT, metadata={"row_count": len(quarantine)})
