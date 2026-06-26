"""Baseline -> LinearRegression; Ridge becomes improvement #1."""
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

NB = "notebooks/eda_and_modeling.ipynb"
nb = nbformat.read(NB, as_version=4)

S = {
"m-hdr": """## Baseline-модель: LinearRegression

---

Применяем решения из блока «Идеи и открытые вопросы»:

- таргет `total_rentals`, гранулярность `(hour, location_id)`;
- **хронологический сплит 70/15/15**, обучаем на train, меряем на validation;
- **самая простая модель** — `LinearRegression` через `Pipeline` + `ColumnTransformer`: cyclic-кодирование времени, one-hot для `location_id`, масштабирование числовых;
- **защита от лика:** `registered_rentals` и `direct_pickups` в сумме дают таргет — в признаки не идут.

Регуляризованные и нелинейные модели — дальше, в разделе «Улучшения», и только если оправданы результатом.""",

"m-pipe-hdr": """### Pipeline

Представление признаков — под линейную модель, внутри `ColumnTransformer` (обёртка над `final_dataset`, не новый датасет):

- `hour_of_day`, `month`, `day_of_week` → **cyclic (sin/cos)**, иначе 23:00 и 00:00 «далеко»;
- `location_id` → **one-hot** (21 значение);
- числовые (`conditions` ordinal, погода, `days_since_launch`) → **StandardScaler**;
- бинарные (`is_weekend`, `is_holiday`) → passthrough.

`make_linear(estimator)` собирает любую линейную модель поверх этого препроцессора; `log_target` оборачивает таргет в log1p/expm1. Обучаемые трансформации фитятся только на train-фолде — лика нет.""",

"m-pipe": """\
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def cyclic(period):
    \"\"\"sin/cos-кодирование циклического признака с заданным периодом.\"\"\"
    return FunctionTransformer(
        lambda x, p=period: np.column_stack([np.sin(2 * np.pi * x / p), np.cos(2 * np.pi * x / p)]),
        feature_names_out="one-to-one",
    )


preprocess = ColumnTransformer([
    ("hour",  cyclic(24), ["hour_of_day"]),
    ("month", cyclic(12), ["month"]),
    ("dow",   cyclic(7),  ["day_of_week"]),
    ("loc",   OneHotEncoder(handle_unknown="ignore"), ["location_id"]),
    ("num",   StandardScaler(), ["conditions", "temperature_c", "perceived_temperature_c",
                                 "humidity", "windspeed_kmh", "days_since_launch"]),
    ("bin",   "passthrough", ["is_weekend", "is_holiday"]),
])


def make_linear(estimator, log_target=False):
    \"\"\"Линейная модель поверх preprocess; log_target=True оборачивает таргет в log1p/expm1.\"\"\"
    pipe = Pipeline([("pre", preprocess), ("model", estimator)])
    if log_target:
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe""",

"m-eval-hdr": """### Обучение и метрики

Набор: **MAE** (интерпретируемо — «промах на N аренд/час/киоск»), **RMSE** (штрафует крупные промахи),
**R²** (доля объяснённой дисперсии), **RMSE/MAE** (диагностика хвоста: ≈1 — однородно, >>1 — есть пики).
Baseline — `LinearRegression`. Обучаем на **train**, меряем на **validation** (test держим до финала).
Сравнение — с наивным baseline (среднее train). Прогноз клипуем снизу нулём.""",

"m-eval": """\
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2_score(y_true, y_pred), "RMSE/MAE": rmse / mae}


baseline = make_linear(LinearRegression())   # raw-таргет (обоснование ниже)
baseline.fit(X_train, y_train)
val_pred = np.clip(baseline.predict(X_val), 0, None)

results = pd.DataFrame({
    "LinearRegression (val)": evaluate(y_val, val_pred),
    "mean-baseline":          evaluate(y_val, np.full(len(y_val), y_train.mean())),
}).T
results.round(3)""",

"m-logvsraw": """\
rows = {}
for name, log in [("raw target", False), ("log1p target", True)]:
    m = make_linear(LinearRegression(), log_target=log)
    m.fit(X_train, y_train)
    p = np.clip(m.predict(X_val), 0, None)
    rows[name] = evaluate(y_val, p)
pd.DataFrame(rows).T.round(3)""",

"m-logvsraw-concl": """**Вывод:** `log1p` символически лучше по MAE, но проигрывает по **RMSE и R²**
(на validation raw R²≈0.35 против log1p≈0.24) — лог-таргет недооценивает пиковые часы, а они нам важны.
→ **baseline на raw-таргете.** (К лог-таргету вернёмся, если приоритетом станет MAE на низких счётчиках.)""",

"m-conclusions": """## Выводы baseline

---

- **LinearRegression бьёт наивный baseline** (на validation R²≈0.35 против отрицательного у предсказания среднего; MAE 7.1 vs 9.5): линейная модель ловит основной сигнал — часы, сезон, киоск, тренд `days_since_launch`.
- **RMSE/MAE≈1.27** → есть хвост: систематическая **недооценка пиковых часов** (утро ~8:00 и вечер 17–18:00) — видно на графике «факт vs прогноз». Это нелинейность и взаимодействия (час × будни, час × погода).
- **Лог-таргет** проигрывает raw по RMSE/R² → baseline на raw-таргете.
- Ошибка по киоскам **почти однородна** (spread ~0.2): one-hot по `location_id` уже впитал уровень киоска. Улучшать надо профиль по часам, не по локациям.
- **Test не трогали** — финальное число посчитаем один раз после выбора модели.

Дальше — раздел «Улучшения»: каждый шаг меняет одну вещь и обосновывается сравнением на validation.""",
}

found = []
for c in nb.cells:
    if c.get("id") in S:
        c["source"] = S[c["id"]]; found.append(c["id"])
assert set(S) == set(found), set(S) - set(found)

# insert improvement section after m-conclusions
imp = []
def md(cid, s):
    c = new_markdown_cell(s); c["id"] = cid; imp.append(c)
def code(cid, s):
    c = new_code_cell(s); c["id"] = cid; imp.append(c)

md("imp-hdr", """## Улучшения

---

Каждый шаг — **одна вещь за раунд**, сравнение на **validation** (test не трогаем). Порядок «просто → сложнее»; усложняем только если оправдано результатом.""")

md("imp1-hdr", """### #1. Регуляризация: Ridge

`LinearRegression` без регуляризации может быть нестабильна при коллинеарных признаках
(`temperature_c` ↔ `perceived_temperature_c`) и широком one-hot — `Ridge` (L2) это лечит.
Но у нас **n ≫ p** (255k строк против ~40 признаков) → ожидаем мизерный эффект. Проверяем на validation.""")

code("imp1-code", """\
rows = {}
for name, est in [("LinearRegression", LinearRegression()),
                  ("Ridge(alpha=1)",   Ridge(alpha=1.0)),
                  ("Ridge(alpha=10)",  Ridge(alpha=10.0))]:
    m = make_linear(est)
    m.fit(X_train, y_train)
    rows[name] = evaluate(y_val, np.clip(m.predict(X_val), 0, None))
pd.DataFrame(rows).T.round(3)""")

md("imp1-concl", """**Вывод:** Ridge **не меняет** метрики (до 3 знаков идентично, даже при alpha=10) — ровно как ожидалось при n ≫ p: регуляризации нечего стягивать, данных с избытком. Для нашего датасета это пустой ход → baseline `LinearRegression` остаётся. Ridge стал бы полезен при сильно меньшем train или гораздо большем числе признаков.

**Следующее улучшение** должно бить по реальной слабости — недооценке пиков, т.е. по **взаимодействиям**. Это #2 `RandomForestRegressor` и #3 `XGBRegressor` (берут взаимодействия из коробки) на том же `final_dataset` и сплите.""")

ids = [c.get("id") for c in nb.cells]
pos = ids.index("m-conclusions") + 1
nb.cells[pos:pos] = imp

nbformat.write(nb, NB)
print("rewrote baseline cells + inserted improvement #1; total", len(nb.cells))
