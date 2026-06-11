"""Assemble the model's feature row from an API request — matching the pipeline.

The served Pipeline expects the same raw feature columns the training dataset
had. Calendar/trend features are derived from the request timestamp and the
holiday lookup exactly as the upstream assets compute them (same ``LAUNCH_DATE``;
``conditions`` mapped to the same ordinal code as ``weather/primary``'s
``cat.codes``), so there is no train/serve skew. Only the columns the model's
signature lists are returned, in signature order.
"""

import pandas as pd

from bike_rental.defs.schemas import LAUNCH_DATE, WEATHER_CONDITIONS
from bike_rental.serving.holidays import HolidayRepository
from bike_rental.serving.schemas import PredictionRequest


def build_feature_row(
    request: PredictionRequest,
    holidays: HolidayRepository,
    feature_columns: list[str],
) -> pd.DataFrame:
    """One-row feature frame for the model, restricted to ``feature_columns`` order."""
    ts = pd.Timestamp(request.timestamp)
    if ts.tz is not None:
        ts = ts.tz_localize(None)

    row = {
        "month": ts.month,
        "hour_of_day": ts.hour,
        "day_of_week": ts.dayofweek,
        "is_weekend": int(ts.dayofweek >= 5),
        "is_holiday": int(holidays.is_holiday(ts.date())),
        # ordinal code = index in WEATHER_CONDITIONS (matches weather/primary cat.codes)
        "conditions": WEATHER_CONDITIONS.index(request.conditions),
        "temperature_c": request.temperature_c,
        "perceived_temperature_c": request.perceived_temperature_c,
        "humidity": request.humidity,
        "windspeed_kmh": request.windspeed_kmh,
        "days_since_launch": (ts.normalize() - LAUNCH_DATE).days,
    }
    return pd.DataFrame([row])[feature_columns]
