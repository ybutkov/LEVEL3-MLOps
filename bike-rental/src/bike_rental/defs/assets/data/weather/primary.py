"""Weather primary layer: encode weather conditions for ML."""

import dagster as dg
import pandas as pd


@dg.asset(group_name="primary", kinds={"pandas"})
def clean_weather(weather_typed: pd.DataFrame) -> pd.DataFrame:
    """Encode the weather condition category as small integer codes."""
    df = weather_typed.copy()
    df["conditions"] = df["conditions"].cat.codes.astype("int8")
    return df
