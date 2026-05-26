# src/bike_rental/defs/assets/weather.py
import dagster as dg
import numpy as np
import pandas as pd


@dg.asset(group_name="processing", kinds={"pandas"})
def clean_weather(weather_raw: pd.DataFrame) -> pd.DataFrame:
    
    WEATHER_SEVERITY = {"clear": 0, "clouds": 1, "light_rain": 2, "heavy_rain": 3}

    clean_weather = weather_raw.copy()
    clean_weather["datetime_hourly"] = pd.to_datetime(clean_weather["datetime"], format="ISO8601").dt.floor("h")

    clean_weather["humidity"] = clean_weather["humidity"].replace(0, np.nan)
    clean_weather["humidity"] = clean_weather["humidity"].interpolate("linear")

    # Ordinal encoding для conditions: clear < clouds < light_rain < heavy_rain.
    clean_weather["conditions"] = clean_weather["conditions"].map(WEATHER_SEVERITY)

    clean_weather = clean_weather.drop(columns=["datetime", "id"])

    return clean_weather
