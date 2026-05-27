import dagster as dg
import pandas as pd


@dg.asset(group_name="primary", kinds={"pandas"})
def clean_weather(weather_typed: pd.DataFrame) -> pd.DataFrame:
    df = weather_typed.copy()
    # ordered Categorical → int codes (NaN → -1)
    df["conditions"] = df["conditions"].cat.codes.astype("int8")
    return df
