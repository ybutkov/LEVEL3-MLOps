# src/bike_rental/defs/assets/weather.py
import dagster as dg
import numpy as np
import pandas as pd

@dg.asset(group_name="processing", kinds={"pandas"})
def clean_holidays(holidays_raw: pd.DataFrame) -> pd.DataFrame:
    
    df = holidays_raw[["date", "holiday"]].copy()
    df["date"] = pd.to_datetime(df["date"], format="ISO8601").dt.date
    return df
