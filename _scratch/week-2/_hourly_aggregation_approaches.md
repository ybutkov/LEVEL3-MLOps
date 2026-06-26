# Hourly rental aggregation: two approaches

Both approaches produce the same result: a DataFrame with one row per `(datetime_hourly, location_id)` pair, containing `registered_rentals`, `direct_pickups`, and `total_rentals` counts.

We compared two implementation styles. Both are valid; we picked **approach B** (tag + unstack) for the final 6.1 cell because of slightly better readability and extensibility.

---

## Approach A — two separate groupbys, then outer-join

```python
reg = registered_rentals_data.copy()
reg["datetime_hourly"] = pd.to_datetime(reg["datetime"]).dt.floor("h")
reg_count = reg.groupby(["datetime_hourly", "location_id"]).size().rename("registered_count")

dir_ = direct_rentals_data.copy()
dir_["datetime_hourly"] = pd.to_datetime(dir_["datetime"]).dt.floor("h")
dir_count = dir_.groupby(["datetime_hourly", "location_id"]).size().rename("direct_count")

hourly_rentals = (
    pd.concat([reg_count, dir_count], axis=1)
      .fillna(0).astype(int)
      .reset_index()
)
hourly_rentals["total_rentals"] = hourly_rentals["registered_count"] + hourly_rentals["direct_count"]
```

**Conceptually:** "process each source separately, then combine the counts side-by-side."

### Pros

- Each source is handled in its own block — easy to follow line-by-line
- Explicit column naming at the `.rename()` step — no positional surprises
- Works well if the two sources have different schemas or need different preprocessing
- No magic from `unstack` to mentally trace

### Cons

- Two `groupby` operations instead of one (slightly more compute on large data)
- Adding a third rental source means adding a third groupby block — code duplication
- The `pd.concat(..., axis=1)` on different indices needs `fillna(0)` (because not every (hour, location) pair appears in both source counts)

---

## Approach B — tag with source flag, unstack on the tag (chosen)

```python
rentals = pd.concat([
    registered_rentals_data.assign(is_registered=True),
    direct_rentals_data.assign(is_registered=False),
], ignore_index=True)
rentals["datetime_hourly"] = pd.to_datetime(rentals["datetime"]).dt.floor("h")

hourly_rentals = (
    rentals.groupby(["datetime_hourly", "location_id", "is_registered"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={True: "registered_rentals", False: "direct_pickups"})
    .rename_axis(columns=None)
    .reset_index()
    [["datetime_hourly", "location_id", "registered_rentals", "direct_pickups"]]
)
hourly_rentals["total_rentals"] = hourly_rentals["registered_rentals"] + hourly_rentals["direct_pickups"]
```

**Conceptually:** "unite all rental events into one table, tag each with its source, then group and pivot on the source."

### Pros

- Single source of truth (`rentals`) — all events live in one DataFrame before aggregation
- Single `groupby` — one pass over the data
- `unstack(fill_value=0)` handles the "no rentals from one source at this hour/location" case automatically — no explicit `fillna(0)`
- Extends easily to a third rental source: add `.assign(is_third=...)` to the concat list and one more rename mapping
- Tag-and-pivot pattern is a common idiom for handling segmented data

### Cons

- Concatenates all 3.3M events into one DataFrame temporarily (more peak memory)
- `unstack` is a bit magic — reader needs to know what it does
- Column order after `unstack` is alphabetical on the tag values (False, True for bool) — we explicitly reorder at the end to control this
- The `.rename_axis(columns=None)` is needed to drop the lingering `"is_registered"` name on the columns axis after `unstack` — easy to forget

---

## Safer column renaming after unstack

A subtle pitfall in the original prototype was renaming columns by **position**:

```python
hourly_rentals_df.columns = ["datetime_hour", "location_id", "direct_pickups", "registered_rentals"]
```

This relies on `unstack` returning columns in alphabetical order (`False` before `True`). If pandas ever changes that order, or if we change `is_registered` to a different type (e.g., int 0/1), the columns would silently swap.

The safer pattern is to rename by **name**:

```python
.rename(columns={True: "registered_rentals", False: "direct_pickups"})
```

This works regardless of column order and makes the intent explicit.

---

## Why B was chosen

- Slightly fewer lines
- Single `groupby` (cleaner mental model: "all events → group → pivot by source")
- Naming aligns with the source CSV files: `registered_bike_rentals.csv` → `registered_rentals`, `direct_pickup_bike_rentals.csv` → `direct_pickups`
- Adding a third source later is a one-line change

Approach A is still a perfectly valid alternative — slightly more verbose, slightly more memory-efficient. If we ever needed per-source preprocessing (e.g., different cleaning rules for `registered` vs `direct`), A would be the better starting point.
