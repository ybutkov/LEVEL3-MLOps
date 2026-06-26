# Замечания к `bike-rental/notebooks/eda_and_preprocessing.ipynb`

> Временный файл. Список того, к чему вернуться при следующем проходе по ноутбуку.
> Ревью сделано на коммите `cc0e019` (post-restructure cleanup).

В целом ноутбук зрелый: чёткая структура (objectives → audit → quality checks → build → summary), все решения объяснены, в §6.5 есть row-count sanity check. Ниже — места, которые стоит почистить.

---

## 🟡 Реальные замечания

### 1. §6.1 — concat-then-split (главное)

Сейчас:

```python
rentals = pd.concat([
    registered_rentals_data.assign(is_registered=True),
    direct_rentals_data.assign(is_registered=False),
], ignore_index=True)
rentals["datetime_hourly"] = pd.to_datetime(rentals["datetime"], format="ISO8601").dt.floor("h")

reg_counts = (
    rentals[rentals["is_registered"]]
    .groupby(["datetime_hourly", "location_id"]).size()
    .reset_index(name="registered_rentals")
)
direct_counts = (
    rentals[~rentals["is_registered"]]
    .groupby(["datetime_hourly", "location_id"]).size()
    .reset_index(name="direct_pickups")
)
```

Проблема: код **сначала склеивает 3.3M строк, а потом сразу же разделяет их обратно фильтром**. Худший из обоих миров — память на `concat` + два `groupby`. В заметках `_hourly_aggregation_approaches.md` ты выбрал Approach B (concat + один groupby + unstack), но в коде получился гибрид Approach A с лишним concat.

**Вариант A — без concat, два groupby:**

```python
reg_counts = (
    registered_rentals_data.assign(
        datetime_hourly=pd.to_datetime(registered_rentals_data["datetime"], format="ISO8601").dt.floor("h"),
    )
    .groupby(["datetime_hourly", "location_id"]).size()
    .reset_index(name="registered_rentals")
)
# то же для direct_counts
```

**Вариант B — concat + один groupby + unstack:**

```python
rentals = pd.concat([
    registered_rentals_data.assign(is_registered=True),
    direct_rentals_data.assign(is_registered=False),
], ignore_index=True)
rentals["datetime_hourly"] = pd.to_datetime(rentals["datetime"], format="ISO8601").dt.floor("h")

counts = (
    rentals.groupby(["datetime_hourly", "location_id", "is_registered"])
    .size()
    .unstack("is_registered", fill_value=0)
    .rename(columns={True: "registered_rentals", False: "direct_pickups"})
    .rename_axis(columns=None)
    .reset_index()
)
```

Рекомендация: вариант A. Меньше памяти, меньше «магии» с unstack, два независимых конвейера читаются проще. Заметка `_hourly_aggregation_approaches.md` про вариант B — это история размышлений, а не финальное решение.

---

### 2. §6.1 — `full_grid` через `MultiIndex.from_product`

Сейчас:

```python
all_hours = sorted(rentals["datetime_hourly"].unique())
all_locations = sorted(rentals["location_id"].unique())
full_grid = pd.merge(
    pd.DataFrame({"datetime_hourly": all_hours}),
    pd.DataFrame({"location_id": all_locations}),
    how="cross",
)
```

Лучше:

```python
full_grid = pd.MultiIndex.from_product(
    [sorted(rentals["datetime_hourly"].unique()),
     sorted(rentals["location_id"].unique())],
    names=["datetime_hourly", "location_id"],
).to_frame(index=False)
```

«Декартово произведение двух осей» — это ровно то, для чего `from_product`. Идиоматичнее.

---

### 3. §6.1 — `fillna(0)` глобальный, лучше явно по колонкам

Сейчас:

```python
.fillna(0)
.astype({"registered_rentals": int, "direct_pickups": int})
```

`.fillna(0)` без указания колонок заполнит **любую** колонку с NaN. Сейчас других колонок с NaN нет, поэтому работает — но если кто-то добавит ещё один merge выше, NaN в новых колонках молча превратятся в 0.

Безопаснее:

```python
.fillna({"registered_rentals": 0, "direct_pickups": 0})
.astype({"registered_rentals": int, "direct_pickups": int})
```

---

### 4. §6.3 — `.map(WEATHER_SEVERITY)` молча даёт NaN на неизвестных значениях

Сейчас:

```python
clean_weather["conditions"] = clean_weather["conditions"].map(WEATHER_SEVERITY)
```

Если в данных появится новая категория (например, `snow`), `.map()` сделает её NaN, и узнаешь об этом только когда модель упадёт. По данным §4.3 категорий ровно 4, и они зашиты в `WEATHER_SEVERITY` — но защитная проверка стоит копейки:

```python
mapped = clean_weather["conditions"].map(WEATHER_SEVERITY)
assert mapped.notna().all(), (
    f"Unknown conditions: {clean_weather.loc[mapped.isna(), 'conditions'].unique()}"
)
clean_weather["conditions"] = mapped
```

---

### 5. §6.5 — `print(... expect ...)` вместо `assert`

Сейчас:

```python
print(f"\nsum(total_rentals): {dataset['total_rentals'].sum():,}  (expect {raw_events_total:,} = raw events)")
```

Это «проверка глазами»: если в будущем число разойдётся, придётся это заметить в выводе. Лучше:

```python
assert dataset["total_rentals"].sum() == raw_events_total, (
    f"Rental count mismatch: {dataset['total_rentals'].sum():,} vs {raw_events_total:,}"
)
```

Регрессия упадёт громко, а не молча.

---

## 🟢 Что нравится (не трогать)

- `basic_audit_dataframe(...)` с опциональными `date_cols` / `categorical_cols` — переиспользуемая и в меру универсальная функция.
- §5 (data quality) — это лучшая часть. Каждое подозрение сформулировано, проверено и закрыто конкретным решением: «keep as-is», «inner-join safe», «replace with NaN + interpolate».
- §6 — каждая ступенька печатает shape и head, легко следить за инвариантами.
- `format="ISO8601"` везде, `.dt.floor("h")` везде — консистентно.
- §6.5 row-count sanity check (`sum(total_rentals) == raw events`) — это именно та проверка, которая ловит самые опасные баги агрегации.

---

## Минорное / опционально

- В §6.1 переменная `rentals` после построения `reg_counts` / `direct_counts` больше не используется, кроме как для `all_hours` / `all_locations`. Если перейти на вариант A из пункта 1, она уйдёт сама.
- §6.4: `holiday_dates` — `set` для `.isin(...)`. `.isin()` сам построит set, можно передать просто list/array. Не критично.

---

## План применения

Пункты 2–5 независимы друг от друга, их можно применять по одному.
Пункт 1 (§6.1) — отдельный заход; сначала решить A vs B, потом применять.
