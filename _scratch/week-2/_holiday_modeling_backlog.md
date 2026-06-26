# Holiday modeling backlog (parked from week-2 exploration)

These three subsections were originally drafted as 5.4-5.6 of `data_exploration.ipynb` during week-2.
They explore **how to encode holidays in the dataset** by asking whether the binary `is_holiday`
flag is enough, or whether a richer encoding (type/name/group) is needed.

**Why parked:** the week-2 handout asks for `is_holiday` as a binary flag plus standard
time-derived features — no holiday-type analysis is required at the preprocessing stage.
This analysis belongs to feature engineering for the modeling stage (week-3+), when we'd
return here if the baseline model struggles with holidays.

**Known issues to fix before reviving:**

- The hourly aggregate is rebuilt three times — should be computed once and reused
- Subsection 5.5 code has stray `Loading` / `daily Loading_rentals` tokens — looks like editor leftover, will not execute
- 5.5 observation claims "12 of 21 holidays outside ±1 std" — actual is 8/21 (verified during week-2 review)
- 5.6's final recommendation uses `year` as a numeric feature. This won't generalize to 2013+
  (model would only have seen year=2011 and year=2012). For learning purposes it's fine;
  for production a trend feature like `days_since_launch` or a rolling baseline is safer.
- Observations are in Russian; headers in English — pick one before reviving.

---

### 5.4 Holiday Impact on Rentals

```python
registered_rentals_data["hour"] = pd.to_datetime(registered_rentals_data["datetime"]).dt.floor("h")
direct_rentals_data["hour"] = pd.to_datetime(direct_rentals_data["datetime"]).dt.floor("h")

reg_by_hour = registered_rentals_data.groupby("hour").size().rename("registered_count")
direct_by_hour = direct_rentals_data.groupby("hour").size().rename("direct_count")

hourly = (
    weather_data.copy()
    .assign(datetime=lambda d: pd.to_datetime(d["datetime"]))
    .set_index("datetime")
    .join(reg_by_hour)
    .join(direct_by_hour)
    .fillna(0)
    .reset_index()
)
hourly["total_rentals"] = hourly["registered_count"] + hourly["direct_count"]
hourly["date"] = hourly["datetime"].dt.date

holiday_calendar_data["date"] = pd.to_datetime(holiday_calendar_data["date"]).dt.date
hourly = hourly.merge(holiday_calendar_data[["date", "holiday"]], on="date", how="left")
hourly["is_holiday"] = hourly["holiday"].notna().astype(int)

regular_day_mean = hourly[hourly["is_holiday"] == 0].groupby("date")["total_rentals"].sum().mean()

holiday_daily = (
    hourly[hourly["is_holiday"] == 1]
    .groupby(["date", "holiday"])["total_rentals"]
    .sum()
    .reset_index()
    .rename(columns={"total_rentals": "daily_rentals"})
    .sort_values("daily_rentals")
)

print(f"Regular day average: {regular_day_mean:.0f} rentals")
print(f"\nPer-holiday daily totals:\n")
print(holiday_daily.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#d62728" if v < regular_day_mean else "#2ca02c" for v in holiday_daily["daily_rentals"]]
ax.barh(holiday_daily["holiday"], holiday_daily["daily_rentals"], color=colors)
ax.axvline(regular_day_mean, color="black", linestyle="--", linewidth=1.2, label=f"Regular day avg ({regular_day_mean:.0f})")
ax.set_xlabel("Total daily rentals")
ax.set_title("Daily rentals on holidays vs regular day average")
ax.legend()
plt.tight_layout()
plt.show()
```

### Observations

- Разброс между праздниками огромный: от ~1,000 (MLK Day, Christmas) до ~7,400 (Independence Day) при среднем обычного дня 4,527.
- Праздники чётко делятся на два кластера:
  - **Зимние/семейные** (Christmas, Thanksgiving, Washington's Birthday, New Year's) — аренды в 2–4x ниже нормы. Люди остаются дома или уезжают из города.
  - **Летние/активные** (Independence Day, Labor Day, D.C. Emancipation Day, Columbus Day) — аренды на уровне нормы или выше. Люди выходят на улицу.
- Бинарный флаг `is_holiday` не различает эти два поведения и может вводить модель в заблуждение.
- **Гипотеза**: часть разброса объясняется погодой (зимой холоднее), а не характером праздника. Стоит проверить, остаётся ли разница после контроля по температуре.
- **Варианты для эксперимента**: добавить `holiday_type` (winter / summer), или оставить `is_holiday` и доверить погодным признакам объяснить остаток.

---

### 5.5 Hypothesis: does temperature explain the holiday spread?

```python
import numpy as np

# Build daily aggregation
_reg = registered_rentals_data.copy()
_dir = direct_rentals_data.copy()
_reg["hour"] = pd.to_datetime(_reg["datetime"]).dt.floor("h")
_dir["hour"] = pd.to_datetime(_dir["datetime"]).dt.floor("h")

_reg_by_hour = _reg.groupby("hour").size().rename("registered_count")
_dir_by_hour = _dir.groupby("hour").size().rename("direct_count")

_hourly = (
    weather_data.copy()
    .assign(datetime=lambda d: pd.to_datetime(d["datetime"]))
    .set_index("datetime")
    .join(_reg_by_hour).join(_dir_by_hour)
    .fillna(0).reset_index()
)
_hourly["total_rentals"] = _hourly["registered_count"] + _hourly["direct_count"]
_hourly["date"] = _hourly["datetime"].dt.date

_hol = holiday_calendar_data.copy()
_hol["date"] = pd.to_datetime(_hol["date"]).dt.date
_hourly = _hourly.merge(_hol[["date", "holiday"]], on="date", how="left")
_hourly["is_holiday"] = _hourly["holiday"].notna().astype(int)

daily = _hourly.groupby("date").agg(
    daily_rentals=("total_rentals", "sum"),
    avg_temp=("temperature_c", "mean"),
    is_holiday=("is_holiday", "max"),
    holiday=("holiday", "first"),
).reset_index()
 Loading
# Fit linear trend on regular days only
regular = daily[daily["is_holiday"] == 0].copy()
holidays_df = daily[daily["is_holiday"] == 1].copy()

coef = np.polyfit(regular["avg_temp"], regular["daily_rentals"], 1)
pred_fn = np.poly1d(coef)

regular["residual"] = regular["daily_rentals"] - pred_fn(regular["avg_temp"])
holidays_df["residual"] = holidays_df["daily_rentals"] - pred_fn(holidays_df["avg_temp"])

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: scatter temp vs rentals
temp_range = np.linspace(daily["avg_temp"].min(), daily["avg_temp"].max(), 100)
axes[0].scatter(regular["avg_temp"], regular["daily Loading_rentals"], alpha=0.3, s=10, label="Regular days")
axes[0].scatter(holidays_df["avg_temp"], holidays_df["daily_rentals"], color="red", s=60, zorder=5, label="Holidays")
axes[0].plot(temp_range, pred_fn(temp_range), "k--", linewidth=1.2, label="Trend (regular days)")
axes[0].set_xlabel("Avg temperature (°C)")
axes[0].set_ylabel("Daily rentals")
axes[0].set_title("Temperature vs daily rentals")
axes[0].legend()

# Right: holiday residuals
colors = ["#d62728" if r < 0 else "#2ca02c" for r in holidays_df["residual"]]
axes[1].barh(holidays_df.sort_values("residual")["holiday"],
             holidays_df.sort_values("residual")["residual"], color=colors)
axes[1].axvline(0, color="black", linewidth=1)
axes[1].axvline(regular["residual"].std(), color="grey", linestyle="--", linewidth=0.8, label="+1 std regular")
axes[1].axvline(-regular["residual"].std(), color="grey", linestyle="--", linewidth=0.8, label="-1 std regular")
axes[1].set_xlabel("Residual (actual − temperature-predicted)")
axes[1].set_title("Holiday effect after controlling for temperature")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()

print(f"Regular day residual std: {regular['residual'].std():.0f}")
print(f"Holiday mean residual:    {holidays_df['residual'].mean():.0f}")
print(f"Holidays outside ±1 std:  {(holidays_df['residual'].abs() > regular['residual'].std()).sum()} / {len(holidays_df)}")
```

### Observations

**Гипотеза опровергнута: температура не объясняет разброс по праздникам.**

- Std остатков обычных дней ≈ 1,512 — это "нормальная" вариация.
- 12 из 21 праздничных точек выходят за пределы ±1 std. Среди них — целые кластеры с остатком -2,000 и ниже.
- Паттерн после контроля на температуру:
  - **Стабильно ниже нормы**: Thanksgiving, Christmas, Washington's Birthday, Memorial Day (2011). Это праздники, когда люди уезжают из города или остаются дома независимо от погоды.
  - **Около нуля или выше**: Independence Day, Columbus Day, Labor Day (2012), Veterans Day. Активные праздники, когда люди выходят на улицу.
- Интересный эффект: один и тот же праздник в разные годы может вести себя по-разному (Labor Day 2011: -2,340 vs 2012: +120). Это может быть связано с днём недели, на который он выпал, или с локальными событиями.

**Вывод для pipeline**: бинарного `is_holiday` недостаточно. Разумный следующий шаг — добавить `holiday_name` (закодированный) или `holiday_type` (civic / family / federal). Взаимодействие `is_holiday × month` тоже может помочь модели уловить этот паттерн без явной типизации.

---

### 5.6 Why holiday grouping won't help: year growth effect

```python
_hourly2 = hourly.copy() if "hourly" in dir() else None

# Rebuild daily with year
_reg2 = registered_rentals_data.copy()
_dir2 = direct_rentals_data.copy()
_reg2["hour"] = pd.to_datetime(_reg2["datetime"]).dt.floor("h")
_dir2["hour"] = pd.to_datetime(_dir2["datetime"]).dt.floor("h")
_reg_h2 = _reg2.groupby("hour").size().rename("registered_count")
_dir_h2 = _dir2.groupby("hour").size().rename("direct_count")

_hourly2 = (
    weather_data.copy()
    .assign(datetime=lambda d: pd.to_datetime(d["datetime"]))
    .set_index("datetime")
    .join(_reg_h2).join(_dir_h2).fillna(0).reset_index()
)
_hourly2["total_rentals"] = _hourly2["registered_count"] + _hourly2["direct_count"]
_hourly2["date"] = _hourly2["datetime"].dt.date
_hourly2["year"] = _hourly2["datetime"].dt.year
_hourly2["month"] = _hourly2["datetime"].dt.month

_hol2 = holiday_calendar_data.copy()
_hol2["date"] = pd.to_datetime(_hol2["date"]).dt.date
_hourly2 = _hourly2.merge(_hol2[["date", "holiday"]], on="date", how="left")
_hourly2["is_holiday"] = _hourly2["holiday"].notna().astype(int)

_daily2 = _hourly2.groupby(["date", "year", "month"]).agg(
    daily_rentals=("total_rentals", "sum"),
    avg_temp=("temperature_c", "mean"),
    is_holiday=("is_holiday", "max"),
).reset_index()

regular2 = _daily2[_daily2["is_holiday"] == 0]

# Year summary
year_stats = regular2.groupby("year")["daily_rentals"].agg(["mean", "median"]).round(0)
print("Regular day average by year:")
print(year_stats)
print(f"\n2012 vs 2011 growth: +{(year_stats.loc[2012,'mean'] / year_stats.loc[2011,'mean'] - 1)*100:.0f}%")

# Monthly heatmap
pivot = regular2.groupby(["year", "month"])["daily_rentals"].mean().unstack("year").round(0)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Left: monthly pivot
im = axes[0].imshow(pivot.values, aspect="auto", cmap="YlOrRd")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["2011", "2012"])
axes[0].set_yticks(range(12))
axes[0].set_yticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
for i in range(12):
    for j in range(2):
        axes[0].text(j, i, f"{pivot.values[i,j]:.0f}", ha="center", va="center", fontsize=8)
axes[0].set_title("Mean daily rentals by month and year\n(regular days only)")
plt.colorbar(im, ax=axes[0])

# Right: year-over-year growth per month
growth = ((pivot[2012] / pivot[2011]) - 1) * 100
axes[1].barh(range(12), growth.values, color="#2ca02c")
axes[1].set_yticks(range(12))
axes[1].set_yticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
axes[1].set_xlabel("YoY growth (%)")
axes[1].set_title("Year-over-year growth by month\n(regular days only)")
axes[1].axvline(growth.mean(), color="black", linestyle="--", linewidth=1, label=f"mean {growth.mean():.0f}%")
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Observations

- 2012 в среднем на **+65% выше** 2011 на обычных днях — и это верно для каждого месяца без исключений.
- Именно это объясняет, почему один и тот же праздник в 2011 выглядит "провальным", а в 2012 — нормальным: фон сильно вырос.
- Вариация внутри одного праздника между годами (до 2,460 rentals) сопоставима со стд обычных дней (1,512) — это шум, а не сигнал о типе праздника.

**Почему больше групп праздников не поможет:**
- В датасете всего 21 праздничный день, из них у большинства по 2 наблюдения (2011 + 2012).
- Std внутри одной группы сопоставим со std обычных дней — группы статистически не разделимы.
- Любая схема с 3+ группами будет фактически заучивать год, а не тип праздника.

**Что использовать вместо:**
- `is_holiday` (бинарный) — факт праздника
- `year` — улавливает рост сервиса между 2011 и 2012
- `month` — улавливает сезонность, которая косвенно разделяет зимние и летние праздники

Модель сама научится взаимодействию `is_holiday × year × month` без ручной типизации праздников.

---

