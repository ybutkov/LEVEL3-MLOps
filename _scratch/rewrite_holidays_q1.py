"""Rewrite §1 of exploration_holidays.ipynb to use a local ±15-day % base."""
import nbformat

NB = "notebooks/exploration_holidays.ipynb"
nb = nbformat.read(NB, as_version=4)

sources = {
    "q1-hdr": """## 1. Эффект праздника: локальная база ±15 дней

---

Сравниваем дневные аренды каждого праздника со средним **простых дней в пределах ±15 дней** от него.
Локальная база убирает оба confound'а сразу — рост сервиса и сезон — потому что берёт «текущие условия»
вокруг даты, а не среднее по всему периоду (который сильно меняется во времени).

Отклонение считаем и в абсолюте, и в **процентах**: % сопоставим между сезонами (летом и база, и праздник крупнее).
День недели **не** матчим — [trends §4](exploration_time_trends.ipynb) показал, что на дневном объёме его эффект ~3%
(он сидит во внутридневной форме, а не в сумме за день). Из базы исключаем все праздники.""",

    "q1-table": """\
WINDOW = pd.Timedelta(days=15)
reg_dates = pd.to_datetime(regular["date"])  # date -> Timestamp для арифметики окна

def local_base(holiday_date):
    \"\"\"Среднее простых дней в ±15 дней от праздника + размер базы.\"\"\"
    near = regular.loc[(reg_dates - pd.Timestamp(holiday_date)).abs() <= WINDOW, "daily_rentals"]
    return near.mean(), int(near.shape[0])

holidays_df = holidays_df.copy()
base = holidays_df["date"].map(local_base)
holidays_df["base_local"] = [b[0] for b in base]
holidays_df["n_base"] = [b[1] for b in base]
holidays_df["dev_abs"] = holidays_df["daily_rentals"] - holidays_df["base_local"]
holidays_df["dev_pct"] = holidays_df["dev_abs"] / holidays_df["base_local"] * 100

holiday_effect = holidays_df.sort_values("dev_pct")[
    ["date", "holiday", "year", "daily_rentals", "base_local", "dev_abs", "dev_pct"]
]
print(f"дней в локальной базе: min={holidays_df['n_base'].min()}  max={holidays_df['n_base'].max()}\\n")
print(holiday_effect.round(1).to_string(index=False))
print(f"\\nмедиана: {holidays_df['dev_pct'].median():.0f}%   "
      f"спад >5%: {(holidays_df['dev_pct'] < -5).sum()}   "
      f"рост >5%: {(holidays_df['dev_pct'] > 5).sum()}   "
      f"норма ±5%: {(holidays_df['dev_pct'].abs() <= 5).sum()}  (из {len(holidays_df)})")""",

    "q1-plot": """\
he = holidays_df.sort_values("dev_pct")
labels = he["holiday"] + " (" + he["year"].astype(str) + ")"
colors = ["#d62728" if v < 0 else "#2ca02c" for v in he["dev_pct"]]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(labels, he["dev_pct"], color=colors)
ax.axvline(0, color="black", linewidth=1)
ax.axvline(he["dev_pct"].median(), color="grey", linestyle="--", linewidth=1.2,
           label=f"медиана {he['dev_pct'].median():.0f}%")
ax.set_xlabel("Отклонение от локальной базы ±15 дней, %")
ax.set_title("Эффект праздника относительно соседних обычных дней")
ax.legend()
plt.tight_layout()
plt.show()""",

    "q1-obs": """### Наблюдения

- Локальная база меняет картину: эффект праздника — преимущественно **спад**. Медиана **−13%**, падают (>5%) **14 из 21**, растут — **5**, около нормы — **2**. Глобальное среднее это маскировало (симметричный ±разброс был артефактом смешения сезонов и роста).
- Два полюса сохраняются, но теперь чистые — без примеси роста/сезона:
  - **Сильный спад:** Christmas (−72%), Thanksgiving (−54%/−53%), Washington's Birthday, MLK, New Year's — семейные/зимние, люди уезжают.
  - **Рост:** Independence Day (+25%/+15%), Veterans Day observed (+31%), Columbus Day 2011 (+20%) — активные/уличные.
- **Год к году теперь устойчиво** — ключевое: тот же праздник в 2011 и 2012 даёт близкий %: Thanksgiving −54/−53, Independence +25/+15, MLK −20/−28, Memorial −12/−8. То «межгодовое расхождение», которое §3 ниже списал на шум роста, было **в основном самим ростом** — локальная база его убирает, и у праздника проявляется устойчивая идентичность. Исключение — **Columbus Day** (+20%/−23%): единственный меняет знак.

**Что это значит для модели:**
- Бинарный `is_holiday` усреднит спад −13% и потеряет полюса (Christmas −72% и July 4 +25% в одну кучу).
- Устойчивость год-к-году — это мягкий аргумент, что богатая кодировка праздника *могла бы* нести сигнал (в отличие от вывода §3). Но на 21 дне (~2 наблюдения на праздник) это всё равно ненадёжно → решает не глаз, а **ablation на тесте** (модель с фичей и без, см. [eda_and_modeling.ipynb](eda_and_modeling.ipynb)).

*📌 Методическое: локальная база ±15 дней уже контролирует и рост, и сезон. Поэтому §2 (температура) и §3 (рост) ниже — теперь скорее «как мы сюда пришли», чем необходимый шаг; их стоит ужать при чистовой сборке.*""",
}

found = set()
for c in nb.cells:
    if c.get("id") in sources:
        c["source"] = sources[c["id"]]
        found.add(c["id"])

missing = set(sources) - found
nbformat.write(nb, NB)
print("rewrote:", sorted(found), "| missing:", sorted(missing))
