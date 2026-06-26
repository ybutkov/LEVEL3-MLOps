"""Insert §4 (day-of-week / weekend) into exploration_time_trends.ipynb."""
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

NB = "notebooks/exploration_time_trends.ipynb"
nb = nbformat.read(NB, as_version=4)

# 1) add day_of_week to the shared daily aggregation
for c in nb.cells:
    if c.get("id") == "daily-agg" and "day_of_week" not in c["source"]:
        c["source"] = c["source"].replace(
            'daily["month"] = daily["date"].dt.month',
            'daily["month"] = daily["date"].dt.month\n'
            'daily["day_of_week"] = daily["date"].dt.dayofweek',
        )

# 2) build §4 cells
new_cells = []


def md(cid, src):
    c = new_markdown_cell(src); c["id"] = cid; new_cells.append(c)


def code(cid, src):
    c = new_code_cell(src); c["id"] = cid; new_cells.append(c)


md("q4-hdr", """## 4. День недели и выходные

---

На обычных днях (без праздников): зависит ли спрос от дня недели? Смотрим два уровня —
дневной объём и профиль внутри суток.

Confound'ы здесь слабые: каждый день недели встречается ~100 раз, равномерно по всему периоду,
так что рост и сезон в среднем гасятся → сырые средние допустимы (в отличие от праздников).""")

code("q4-daily", """\
dow_names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]

by_dow = regular.groupby("day_of_week")["daily_rentals"].agg(["mean", "std", "count"]).round(0)
by_dow.index = [dow_names[i] for i in by_dow.index]

weekday_mean = regular[regular["day_of_week"] < 5]["daily_rentals"].mean()
weekend_mean = regular[regular["day_of_week"] >= 5]["daily_rentals"].mean()

print("Среднедневные аренды по дню недели (обычные дни):")
print(by_dow.to_string())
print(f"\\nБудни: {weekday_mean:,.0f}   выходные: {weekend_mean:,.0f}   "
      f"разница: {(weekend_mean / weekday_mean - 1) * 100:+.1f}%")

fig, ax = plt.subplots(figsize=(8, 3.5))
colors = ["#4c72b0"] * 5 + ["#dd8452"] * 2
ax.bar(by_dow.index, by_dow["mean"], color=colors)
ax.axhline(regular["daily_rentals"].mean(), color="grey", linestyle="--", linewidth=1,
           label="среднее по всем дням")
ax.set_ylabel("Среднедневные аренды"); ax.set_title("Спрос по дню недели (обычные дни)")
ax.legend()
plt.tight_layout(); plt.show()""")

md("q4-daily-obs", """### Наблюдения

- Дневной объём **почти не зависит от дня недели**: Пн–Вс в диапазоне ~4,230–4,720, будни vs выходные — всего **−4.3%**.
- Как фича *объёма* `is_weekend` слабая — общее число аренд за день примерно одинаковое.
- Но это про суммарный объём. Совпадение объёма не значит совпадение распределения внутри дня → проверяем профиль по часам.""")

code("q4-hourly", """\
# Системные аренды по часам (все локации), обычные дни.
hourly = rentals.assign(hour_floor=rentals["datetime"].dt.floor("h"))
hourly_counts = hourly.groupby("hour_floor").size().reset_index(name="rentals")
hourly_counts["date"] = hourly_counts["hour_floor"].dt.date
hourly_counts["hour"] = hourly_counts["hour_floor"].dt.hour
hourly_counts["dow"] = hourly_counts["hour_floor"].dt.dayofweek
hourly_counts = hourly_counts[~hourly_counts["date"].isin(hol_dates)]
hourly_counts["is_weekend"] = (hourly_counts["dow"] >= 5).astype(int)

profile = hourly_counts.groupby(["is_weekend", "hour"])["rentals"].mean().unstack("is_weekend")
profile.columns = ["будни", "выходные"]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(profile.index, profile["будни"], marker="o", label="будни")
ax.plot(profile.index, profile["выходные"], marker="o", label="выходные")
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("Час суток"); ax.set_ylabel("Средние аренды/час (система)")
ax.set_title("Профиль спроса по часам: будни vs выходные"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print("Пиковые часы — будни:   ", profile["будни"].sort_values(ascending=False).head(3).index.tolist())
print("Пиковые часы — выходные:", profile["выходные"].sort_values(ascending=False).head(3).index.tolist())""")

md("q4-hourly-obs", """### Наблюдения

При почти равном дневном объёме **форма внутри суток разная**:

- **Будни** — двугорбый коммьют-профиль: пики в **8:00** и **17:00–18:00** (дорога на работу и обратно).
- **Выходные** — один пологий дневной горб **12:00–14:00**, и более живые ночи (0–2ч: ~90 против ~30 в будни — досуг/ночная жизнь).

**Главный вывод:** эффект дня недели — это **взаимодействие `is_weekend` × `hour_of_day`**, а не сдвиг уровня. `is_weekend` сам по себе сдвигает дневной объём лишь на 4%, но полностью переставляет, *когда* люди катаются.

Прямое следствие для модели: это ровно та структура, которую недооценивал Ridge-baseline (систематический промах на пиковых часах, см. [eda_and_modeling.ipynb](eda_and_modeling.ipynb)). Линейной модели нужен явный член взаимодействия час×выходной; деревья поймают его бесплатно.""")

# 3) insert §4 before "## Выводы"
idx = next(i for i, c in enumerate(nb.cells) if c["source"].lstrip().startswith("## Выводы"))
nb.cells[idx:idx] = new_cells

# 4) add a bullet to Выводы
concl = nb.cells[idx + len(new_cells)]
if "День недели" not in concl["source"]:
    concl["source"] = concl["source"].rstrip() + (
        "\n- **День недели** влияет не на объём (будни vs выходные всего −4%), а на **форму** внутри"
        " суток: будни — двойной коммьют-пик (8/17–18ч), выходные — дневной горб (12–14ч). Значит"
        " ценность не в `is_weekend`, а во взаимодействии `is_weekend × hour_of_day` (§4)."
    )

nbformat.write(nb, NB)
print(f"inserted {len(new_cells)} cells before Выводы at idx {idx}; total {len(nb.cells)}")
