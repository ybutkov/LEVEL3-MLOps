"""Scratch: 70/15/15 chronological split, Ridge baseline on validation. Run from bike-rental/."""
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
registered = pd.read_csv(DATA_RAW / "registered_bike_rentals.csv")
direct = pd.read_csv(DATA_RAW / "direct_pickup_bike_rentals.csv")
weather = pd.read_csv(DATA_RAW / "weather.csv")
holidays = pd.read_csv(DATA_RAW / "holidays.csv")

rentals = pd.concat([registered.assign(is_registered=True), direct.assign(is_registered=False)], ignore_index=True)
rentals["datetime_hourly"] = pd.to_datetime(rentals["datetime"], format=DATETIME_FORMAT).dt.floor("h")
reg_counts = rentals[rentals["is_registered"]].groupby(["datetime_hourly", "location_id"]).size().reset_index(name="registered_rentals")
direct_counts = rentals[~rentals["is_registered"]].groupby(["datetime_hourly", "location_id"]).size().reset_index(name="direct_pickups")
full_grid = pd.merge(pd.DataFrame({"datetime_hourly": sorted(rentals["datetime_hourly"].unique())}),
                     pd.DataFrame({"location_id": sorted(rentals["location_id"].unique())}), how="cross")
hourly = (full_grid.merge(reg_counts, on=["datetime_hourly", "location_id"], how="left")
          .merge(direct_counts, on=["datetime_hourly", "location_id"], how="left")
          .fillna(0).astype({"registered_rentals": int, "direct_pickups": int}))
hourly["total_rentals"] = hourly["registered_rentals"] + hourly["direct_pickups"]
dt = hourly["datetime_hourly"]
hourly["month"], hourly["hour_of_day"], hourly["day_of_week"] = dt.dt.month, dt.dt.hour, dt.dt.dayofweek
hourly["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
hourly["days_since_launch"] = (dt.dt.normalize() - pd.Timestamp("2011-01-01")).dt.days
WEATHER_SEVERITY = {"clear": 0, "clouds": 1, "light_rain": 2, "heavy_rain": 3}
cw = weather.copy()
cw["datetime_hourly"] = pd.to_datetime(cw["datetime"], format=DATETIME_FORMAT).dt.floor("h")
cw["humidity"] = cw["humidity"].replace(0, np.nan).interpolate("linear")
cw["conditions"] = cw["conditions"].map(WEATHER_SEVERITY)
cw = cw.drop(columns=["datetime", "id"])
dataset = hourly.merge(cw, on="datetime_hourly", how="left")
hol_dates = set(pd.to_datetime(holidays["date"], format=DATETIME_FORMAT).dt.date)
dataset["is_holiday"] = dataset["datetime_hourly"].dt.date.isin(hol_dates).astype(int)

TARGET = "total_rentals"
FEATURES = ["month", "hour_of_day", "day_of_week", "is_weekend", "is_holiday", "conditions",
            "temperature_c", "perceived_temperature_c", "humidity", "windspeed_kmh",
            "days_since_launch", "location_id"]
final_dataset = dataset[["datetime_hourly", *FEATURES, TARGET]].copy()

# ---- 70/15/15 chronological split on unique timestamps ----
ds = final_dataset.sort_values(["datetime_hourly", "location_id"]).reset_index(drop=True)
ts = np.sort(ds["datetime_hourly"].unique())
cut1, cut2 = ts[int(len(ts) * 0.70)], ts[int(len(ts) * 0.85)]
train = ds[ds["datetime_hourly"] < cut1]
val = ds[(ds["datetime_hourly"] >= cut1) & (ds["datetime_hourly"] < cut2)]
test = ds[ds["datetime_hourly"] >= cut2]
for name, part in [("train", train), ("val", val), ("test", test)]:
    print(f"{name:5} {len(part):>7,} ({len(part)/len(ds):.0%})  "
          f"[{part['datetime_hourly'].min():%Y-%m-%d} … {part['datetime_hourly'].max():%Y-%m-%d}]")

X_train, y_train = train[FEATURES], train[TARGET]
X_val, y_val = val[FEATURES], val[TARGET]


def cyclic(period):
    return FunctionTransformer(lambda x, p=period: np.column_stack([np.sin(2*np.pi*x/p), np.cos(2*np.pi*x/p)]),
                              feature_names_out="one-to-one")


pre = ColumnTransformer([
    ("hour", cyclic(24), ["hour_of_day"]), ("month", cyclic(12), ["month"]),
    ("dow", cyclic(7), ["day_of_week"]), ("loc", OneHotEncoder(handle_unknown="ignore"), ["location_id"]),
    ("num", StandardScaler(), ["conditions", "temperature_c", "perceived_temperature_c", "humidity", "windspeed_kmh", "days_since_launch"]),
    ("bin", "passthrough", ["is_weekend", "is_holiday"]),
])


def evaluate(y, p):
    rmse = np.sqrt(mean_squared_error(y, p))
    mae = mean_absolute_error(y, p)
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2_score(y, p), 3), "RMSE/MAE": round(rmse/mae, 3)}


print("\n--- Ridge on VALIDATION ---")
for name, log in [("raw", False), ("log1p", True)]:
    pipe = Pipeline([("pre", pre), ("ridge", Ridge(alpha=1.0))])
    m = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1) if log else pipe
    m.fit(X_train, y_train)
    p = np.clip(m.predict(X_val), 0, None)
    print(f"{name:6}", evaluate(y_val, p))
print("mean-base", evaluate(y_val, np.full(len(y_val), y_train.mean())))

# breakdown on validation (raw model)
pipe = Pipeline([("pre", pre), ("ridge", Ridge(alpha=1.0))]); pipe.fit(X_train, y_train)
pv = np.clip(pipe.predict(X_val), 0, None)
bh = val.assign(ae=np.abs(y_val.values - pv)).groupby("hour_of_day")["ae"].mean().sort_values(ascending=False)
print("\nworst hours (val):", bh.head(4).round(2).to_dict())
bl = val.assign(ae=np.abs(y_val.values - pv)).groupby("location_id")["ae"].mean()
print("kiosk MAE (val): min=%.2f max=%.2f spread=%.2f" % (bl.min(), bl.max(), bl.max()-bl.min()))
