"""Scratch prototype: verify Ridge baseline runs and get real numbers.

Run from bike-rental/ dir with the project venv:
    .venv/bin/python ../_scratch/ridge_baseline_proto.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

DATETIME_FORMAT = "ISO8601"
DATA_RAW = Path("data/raw")

# ---- rebuild dataset (mirrors notebook) ----
registered = pd.read_csv(DATA_RAW / "registered_bike_rentals.csv")
direct = pd.read_csv(DATA_RAW / "direct_pickup_bike_rentals.csv")
weather = pd.read_csv(DATA_RAW / "weather.csv")
holidays = pd.read_csv(DATA_RAW / "holidays.csv")

rentals = pd.concat([
    registered.assign(is_registered=True),
    direct.assign(is_registered=False),
], ignore_index=True)
rentals["datetime_hourly"] = pd.to_datetime(rentals["datetime"], format=DATETIME_FORMAT).dt.floor("h")

reg_counts = (rentals[rentals["is_registered"]]
              .groupby(["datetime_hourly", "location_id"]).size().reset_index(name="registered_rentals"))
direct_counts = (rentals[~rentals["is_registered"]]
                 .groupby(["datetime_hourly", "location_id"]).size().reset_index(name="direct_pickups"))

all_hours = sorted(rentals["datetime_hourly"].unique())
all_locations = sorted(rentals["location_id"].unique())
full_grid = pd.merge(pd.DataFrame({"datetime_hourly": all_hours}),
                     pd.DataFrame({"location_id": all_locations}), how="cross")
hourly = (full_grid
          .merge(reg_counts, on=["datetime_hourly", "location_id"], how="left")
          .merge(direct_counts, on=["datetime_hourly", "location_id"], how="left")
          .fillna(0).astype({"registered_rentals": int, "direct_pickups": int}))
hourly["total_rentals"] = hourly["registered_rentals"] + hourly["direct_pickups"]

dt = hourly["datetime_hourly"]
hourly["month"] = dt.dt.month
hourly["hour_of_day"] = dt.dt.hour
hourly["day_of_week"] = dt.dt.dayofweek
hourly["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
# NEW: days_since_launch
launch = pd.Timestamp("2011-01-01")
hourly["days_since_launch"] = (dt.dt.normalize() - launch).dt.days

WEATHER_SEVERITY = {"clear": 0, "clouds": 1, "light_rain": 2, "heavy_rain": 3}
cw = weather.copy()
cw["datetime_hourly"] = pd.to_datetime(cw["datetime"], format=DATETIME_FORMAT).dt.floor("h")
cw["humidity"] = cw["humidity"].replace(0, np.nan).interpolate("linear")
cw["conditions"] = cw["conditions"].map(WEATHER_SEVERITY)
cw = cw.drop(columns=["datetime", "id"])

dataset = hourly.merge(cw, on="datetime_hourly", how="left")
holiday_dates = set(pd.to_datetime(holidays["date"], format=DATETIME_FORMAT).dt.date)
dataset["is_holiday"] = dataset["datetime_hourly"].dt.date.isin(holiday_dates).astype(int)

print("dataset shape:", dataset.shape)
print("days_since_launch range:", dataset["days_since_launch"].min(), "..", dataset["days_since_launch"].max())

# ---- target distribution ----
tr = dataset["total_rentals"]
print("\ntarget: mean=%.2f median=%.0f max=%d zeros=%.1f%% skew=%.2f"
      % (tr.mean(), tr.median(), tr.max(), (tr == 0).mean() * 100, tr.skew()))

# ---- chronological 70/30 split on unique timestamps ----
ds = dataset.sort_values(["datetime_hourly", "location_id"]).reset_index(drop=True)
ts = np.sort(ds["datetime_hourly"].unique())
cut = ts[int(len(ts) * 0.70)]
train = ds[ds["datetime_hourly"] < cut]
test = ds[ds["datetime_hourly"] >= cut]
print("\nsplit cut at:", cut)
print("train rows=%d (%.1f%%) test rows=%d (%.1f%%)"
      % (len(train), len(train) / len(ds) * 100, len(test), len(test) / len(ds) * 100))

FEATURES = ["month", "hour_of_day", "day_of_week", "is_weekend", "is_holiday",
            "conditions", "temperature_c", "perceived_temperature_c", "humidity",
            "windspeed_kmh", "days_since_launch", "location_id"]
X_train, y_train = train[FEATURES], train["total_rentals"]
X_test, y_test = test[FEATURES], test["total_rentals"]


def cyclic(period):
    return FunctionTransformer(
        lambda x, p=period: np.column_stack([np.sin(2 * np.pi * x / p), np.cos(2 * np.pi * x / p)]),
        feature_names_out="one-to-one",
    )


pre = ColumnTransformer([
    ("hour", cyclic(24), ["hour_of_day"]),
    ("month", cyclic(12), ["month"]),
    ("dow", cyclic(7), ["day_of_week"]),
    ("loc", OneHotEncoder(handle_unknown="ignore"), ["location_id"]),
    ("num", StandardScaler(), ["conditions", "temperature_c", "perceived_temperature_c",
                               "humidity", "windspeed_kmh", "days_since_launch"]),
    ("bin", "passthrough", ["is_weekend", "is_holiday"]),
])

model = TransformedTargetRegressor(
    regressor=Pipeline([("pre", pre), ("ridge", Ridge(alpha=1.0))]),
    func=np.log1p, inverse_func=np.expm1,
)
model.fit(X_train, y_train)
pred = np.clip(model.predict(X_test), 0, None)


def report(y, p, label):
    mae = mean_absolute_error(y, p)
    rmse = np.sqrt(mean_squared_error(y, p))
    r2 = r2_score(y, p)
    print("%-12s MAE=%.3f RMSE=%.3f R2=%.4f RMSE/MAE=%.2f" % (label, mae, rmse, r2, rmse / mae))


print("\n--- Ridge (log1p target) ---")
report(y_test, pred, "test")
# naive baseline: predict train mean
report(y_test, np.full(len(y_test), y_train.mean()), "mean-base")

# breakdown by hour
print("\nMAE by hour_of_day (top 5 worst):")
bd = test.assign(ae=np.abs(y_test.values - pred)).groupby("hour_of_day")["ae"].mean().sort_values(ascending=False)
print(bd.head().round(2).to_string())

# ---- log1p vs raw target comparison ----
print("\n--- log1p vs raw target (Ridge) ---")
for name, ttr in [("raw", None), ("log1p", (np.log1p, np.expm1))]:
    pipe = Pipeline([("pre", pre), ("ridge", Ridge(alpha=1.0))])
    m = pipe if ttr is None else TransformedTargetRegressor(regressor=pipe, func=ttr[0], inverse_func=ttr[1])
    m.fit(X_train, y_train)
    p = np.clip(m.predict(X_test), 0, None)
    report(y_test, p, name)
