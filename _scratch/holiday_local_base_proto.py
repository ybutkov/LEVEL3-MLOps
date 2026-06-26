"""Scratch: ±15-day local base for holiday effect (%). Run from bike-rental/."""
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

daily = rentals.groupby("date").size().reset_index(name="daily_rentals")
daily["date"] = pd.to_datetime(daily["date"])
hol = holidays.copy()
hol["date"] = pd.to_datetime(hol["date"], format=DATETIME_FORMAT)
daily = daily.merge(hol[["date", "holiday"]], on="date", how="left")
daily["is_holiday"] = daily["holiday"].notna().astype(int)

regular = daily[daily["is_holiday"] == 0]
holidays_df = daily[daily["is_holiday"] == 1].copy()

WINDOW = pd.Timedelta(days=15)


def local_base(hdate):
    mask = (regular["date"] - hdate).abs() <= WINDOW
    return regular.loc[mask, "daily_rentals"].mean(), int(mask.sum())


bases, ns = zip(*[local_base(d) for d in holidays_df["date"]])
holidays_df["base_local"] = bases
holidays_df["n_base"] = ns
holidays_df["dev_abs"] = holidays_df["daily_rentals"] - holidays_df["base_local"]
holidays_df["dev_pct"] = holidays_df["dev_abs"] / holidays_df["base_local"] * 100

out = holidays_df.sort_values("dev_pct")[
    ["date", "holiday", "daily_rentals", "base_local", "n_base", "dev_abs", "dev_pct"]
]
print(f"глобальное среднее обычного дня: {regular['daily_rentals'].mean():,.0f}")
print(f"окно ±15 дней, дней в базе: min={min(ns)} max={max(ns)}\n")
with pd.option_context("display.width", 160, "display.max_columns", None):
    print(out.round(1).to_string(index=False))

print(f"\nдиапазон dev_pct: {holidays_df['dev_pct'].min():.0f}%..{holidays_df['dev_pct'].max():.0f}%")
print(f"медиана dev_pct: {holidays_df['dev_pct'].median():.0f}%")
print(f"праздников с падением (>5%): {(holidays_df['dev_pct'] < -5).sum()}, "
      f"рост (>5%): {(holidays_df['dev_pct'] > 5).sum()}, "
      f"около нормы (±5%): {(holidays_df['dev_pct'].abs() <= 5).sum()}")

# same holiday across years: does % base stabilize the 2011/2012 gap vs global?
print("\nОдин праздник, два года — dev_pct:")
piv = holidays_df.pivot_table(index="holiday", columns=holidays_df["date"].dt.year, values="dev_pct").dropna()
print(piv.round(0).to_string())
