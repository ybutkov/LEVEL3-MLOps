"""Schema-as-code for raw and intermediate sources.

Single source of truth for what every CSV looks like: columns, dtypes,
nullable, uniqueness, value ranges, vocabularies.

Two patterns live here:
- `*Raw` schemas: validated by asset_checks on raw assets — structural gate
  at ingest (columns, types, ranges, vocab). No parseability checks: parsing
  is done once inside intermediate.
- `ParsedDatetime`: shared row-level check used by every `*_split` multi_asset
  after a parse helper column (`parsed_dt`) has been added. NaT in `parsed_dt`
  → row goes to quarantine (see `validate_and_split`).
"""
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors
from pandera.typing import Series


USER_ID_MIN,     USER_ID_MAX     = 0, 300
LOCATION_ID_MIN, LOCATION_ID_MAX = 0, 20
WEATHER_CONDITIONS = ["clear", "clouds", "light_rain", "heavy_rain"]
DATETIME_FORMAT = "ISO8601"


def parse_datetime(s: pd.Series) -> pd.Series:
    """Parse string column to Timestamp using project DATETIME_FORMAT.

    Returns NaT for unparseable rows (caller decides whether to keep or
    route to quarantine).
    """
    return pd.to_datetime(s, format=DATETIME_FORMAT, errors="coerce")


def validate_and_split(
    df: pd.DataFrame,
    schema,
    raw_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate `df` against `schema`, return (clean, quarantine).

    Schema-level errors (missing columns, wrong dtypes) → RuntimeError —
    partial processing is unsafe.
    Row-level errors → those rows go to quarantine with only `raw_cols`
    columns (any helper columns added before validation are stripped).
    """
    try:
        schema.validate(df, lazy=True)
        return df, pd.DataFrame(columns=raw_cols)
    except SchemaErrors as e:
        cases = e.failure_cases
        schema_lvl = cases[cases["index"].isnull()]
        if not schema_lvl.empty:
            raise RuntimeError(
                f"schema-level validation errors: {schema_lvl.to_dict('records')}"
            )
        bad_idx = pd.Index(cases["index"].dropna().unique())
        quarantine = df.loc[bad_idx, raw_cols].copy()
        clean      = df.drop(index=bad_idx)
        return clean, quarantine


class RentalsRaw(pa.DataFrameModel):
    """Schema shared by registered_rentals_raw and direct_pickups_raw."""

    id:          Series[int] = pa.Field(unique=True, ge=1, nullable=False)
    datetime:    Series[str] = pa.Field(nullable=False)
    user_id:     Series[int] = pa.Field(ge=USER_ID_MIN,     le=USER_ID_MAX,     nullable=False)
    location_id: Series[int] = pa.Field(ge=LOCATION_ID_MIN, le=LOCATION_ID_MAX, nullable=False)

    class Config:
        strict = True
        coerce = False



class WeatherRaw(pa.DataFrameModel):
    id:                      Series[int]   = pa.Field(unique=True, ge=1, nullable=False)
    datetime:                Series[str]   = pa.Field(unique=True, nullable=False)
    conditions:              Series[str]   = pa.Field(nullable=False)
    temperature_c:           Series[float] = pa.Field(ge=-40, le=50,  nullable=False)
    perceived_temperature_c: Series[float] = pa.Field(ge=-50, le=60,  nullable=False)
    humidity:                Series[float] = pa.Field(ge=0,   le=100, nullable=False)
    windspeed_kmh:           Series[float] = pa.Field(ge=0,   le=200, nullable=False)

    class Config:
        strict = True
        coerce = False
   
    @pa.check("conditions", name="known_weather_condition")
    def _conditions_known(cls, s: pd.Series) -> pd.Series:
        return s.astype(str).str.lower().str.strip().isin(WEATHER_CONDITIONS)


class HolidaysRaw(pa.DataFrameModel):
    """Structural contract for holidays_raw — validated by asset_check at ingest.

    No date-parseability check here: parsing happens once inside holidays_split
    (intermediate), and ParsedDatetime catches parse failures via NaT detection.
    """

    id:      Series[int] = pa.Field(unique=True, ge=1, nullable=False)
    date:    Series[str] = pa.Field(unique=True, nullable=False)
    holiday: Series[str] = pa.Field(nullable=False)

    class Config:
        strict = True
        coerce = False


class ParsedDatetime(pa.DataFrameModel):
    """Shared row-level parse check used by every `*_split` multi_asset.

    Each intermediate asset adds a temporary `parsed_dt` column via
    `schemas.parse_datetime(...)`, then runs this schema. `strict = False`
    ignores all other columns (their structural contracts are enforced by
    `*Raw` schemas at the ingest gate). NaT in `parsed_dt` (unparseable
    string) becomes a row-level failure → row goes to quarantine.
    """

    parsed_dt: Series[pd.Timestamp] = pa.Field(nullable=False)

    class Config:
        strict = False
        coerce = False
