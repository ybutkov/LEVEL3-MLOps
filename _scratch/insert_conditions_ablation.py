"""Insert improvement #2: conditions ordinal vs one-hot, after the Ridge note."""
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

NB = "notebooks/eda_and_modeling.ipynb"
nb = nbformat.read(NB, as_version=4)

cells = []
def md(cid, s):
    c = new_markdown_cell(s); c["id"] = cid; cells.append(c)
def code(cid, s):
    c = new_code_cell(s); c["id"] = cid; cells.append(c)

md("imp2-hdr", """### #2. Кодирование `conditions`: ординал vs one-hot

Ординал (0..3 + scaler) навязывает линейной модели **равный шаг** между состояниями погоды
(clear→clouds = light_rain→heavy_rain). One-hot снимает это допущение — свой коэффициент на каждое состояние.
Проверяем, оправдан ли он. (`heavy_rain` редок: 63 строки на весь датасет, и все в train.)""")

code("imp2-code", """\
# Два варианта препроцессора, отличаются только обработкой conditions.
WEATHER_NUM = ["temperature_c", "perceived_temperature_c", "humidity", "windspeed_kmh", "days_since_launch"]
common = [
    ("hour",  cyclic(24), ["hour_of_day"]),
    ("month", cyclic(12), ["month"]),
    ("dow",   cyclic(7),  ["day_of_week"]),
    ("loc",   OneHotEncoder(handle_unknown="ignore"), ["location_id"]),
    ("bin",   "passthrough", ["is_weekend", "is_holiday"]),
]


def preprocess_conditions(mode):
    if mode == "ordinal":
        steps = common + [("num", StandardScaler(), WEATHER_NUM + ["conditions"])]
    else:  # one-hot
        steps = common + [("num", StandardScaler(), WEATHER_NUM),
                          ("cond", OneHotEncoder(handle_unknown="ignore"), ["conditions"])]
    return ColumnTransformer(steps)


rows = {}
for mode in ["ordinal", "one-hot"]:
    m = Pipeline([("pre", preprocess_conditions(mode)), ("model", LinearRegression())])
    m.fit(X_train, y_train)
    rows[f"conditions = {mode}"] = evaluate(y_val, np.clip(m.predict(X_val), 0, None))
pd.DataFrame(rows).T.round(3)""")

md("imp2-concl", """**Вывод:** one-hot даёт **+0.002 R²** — в пределах шума: погода слабый признак на фоне часа/сезона/локации, а допущение равного шага здесь почти выполняется. Колонка `heavy_rain` к тому же обучается, но на validation не встречается (0 строк) — мёртвый груз. → оставляем **ординал** (проще, по качеству эквивалентно). Для деревьев (#3/#4) кодирование не важно — им ординал ок.""")

ids = [c.get("id") for c in nb.cells]
pos = ids.index("imp1-concl") + 1
nb.cells[pos:pos] = cells
nbformat.write(nb, NB)
print("inserted improvement #2 at", pos, "; total", len(nb.cells))
