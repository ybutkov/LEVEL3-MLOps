"""Scratch: day-of-week / weekend effect for time-trends notebook. Run from bike-rental/."""
from pathlib import Path

import numpy as np
import pandas as pd

DATETIME_FORMAT = "ISO8601"
DATA_RAW = Path("data/raw")

reg = pd.read_csv(DATA_RAW / "registered_bike_rentals.csv")
direct = pd.read_csv(DATA_RAW / "direct_pickup_bike_rentals.csv")
holidays = pd.read_csv(DATA_RAW / "holidays.csv")

rentals = pd.concat([reg, direct], ignore_index=True)
rentals["datetime"] = pd.to_datetime(rentals["datetime"], format=DATETIME_FORMAT)
rentals["date"] = rentals["datetime"].dt.date

hol_dates = set(pd.to_datetime(holidays["date"], format=DATETIME_FORMAT).dt.date)

# ---- daily level, regular days only ----
daily = rentals.groupby("date").size().reset_index(name="daily_rentals")
daily["date"] = pd.to_datetime(daily["date"])
daily["day_of_week"] = daily["date"].dt.dayofweek
daily["is_holiday"] = daily["date"].dt.date.isin(hol_dates).astype(int)
regular = daily[daily["is_holiday"] == 0]

dow_names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
by_dow = regular.groupby("day_of_week")["daily_rentals"].agg(["mean", "std", "count"]).round(0)
by_dow.index = [dow_names[i] for i in by_dow.index]
print("Среднедневные аренды по дню недели (обычные дни):")
print(by_dow.to_string())

wd = regular[regular["day_of_week"] < 5]["daily_rentals"].mean()
we = regular[regular["day_of_week"] >= 5]["daily_rentals"].mean()
print(f"\nБудни:    {wd:,.0f}")
print(f"Выходные: {we:,.0f}")
print(f"Разница:  {(we / wd - 1) * 100:+.1f}%")

# ---- hourly profile: weekday vs weekend ----
hourly = rentals.copy()
hourly["hour_floor"] = hourly["datetime"].dt.floor("h")
hourly_counts = hourly.groupby("hour_floor").size().reset_index(name="rentals")
hourly_counts["date"] = hourly_counts["hour_floor"].dt.date
hourly_counts["hour"] = hourly_counts["hour_floor"].dt.hour
hourly_counts["dow"] = hourly_counts["hour_floor"].dt.dayofweek
hourly_counts["is_holiday"] = hourly_counts["date"].isin(hol_dates).astype(int)
hc = hourly_counts[hourly_counts["is_holiday"] == 0]
hc = hc.assign(is_weekend=(hc["dow"] >= 5).astype(int))

prof = hc.groupby(["is_weekend", "hour"])["rentals"].mean().unstack("is_weekend")
prof.columns = ["будни", "выходные"]
print("\nСредние аренды по часам (будни vs выходные):")
print(prof.round(1).to_string())

print("\nПиковые часы:")
print(" будни:    топ-3 ->", prof["будни"].sort_values(ascending=False).head(3).index.tolist())
print(" выходные: топ-3 ->", prof["выходные"].sort_values(ascending=False).head(3).index.tolist())
