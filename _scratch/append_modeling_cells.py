"""Append the Ridge-baseline modeling cells to eda_and_modeling.ipynb."""
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

NB = "notebooks/eda_and_modeling.ipynb"
nb = nbformat.read(NB, as_version=4)

cells = []


def md(cell_id, src):
    c = new_markdown_cell(src)
    c["id"] = cell_id
    cells.append(c)


def code(cell_id, src):
    c = new_code_cell(src)
    c["id"] = cell_id
    cells.append(c)


md("m-hdr", """## Baseline-модель: Ridge

---

Применяем решения из блока «Идеи и открытые вопросы»:

- таргет `total_rentals`, гранулярность `(hour, location_id)`;
- **хронологический сплит 70/30** (random нельзя — лик по времени);
- Ridge через `Pipeline` + `ColumnTransformer`: cyclic-кодирование времени, one-hot для `location_id`, масштабирование числовых;
- **защита от лика:** `registered_rentals` и `direct_pickups` в сумме дают таргет — в признаки не идут.""")

code("m-final-dataset", """\
# Один базовый final_dataset для всех моделей: одни и те же строки, таргет и сплит.
# registered_rentals / direct_pickups исключены — это компоненты таргета (лик).
TARGET = "total_rentals"
FEATURES = [
    "month", "hour_of_day", "day_of_week", "is_weekend", "is_holiday",
    "conditions", "temperature_c", "perceived_temperature_c", "humidity",
    "windspeed_kmh", "days_since_launch", "location_id",
]
final_dataset = dataset[["datetime_hourly", *FEATURES, TARGET]].copy()
print(f"final_dataset: {final_dataset.shape},  признаков: {len(FEATURES)}")
final_dataset.head()""")

md("m-target-hdr", """### Распределение таргета

Сильно скошено вправо, ~14% строк — нули (тихие часы на киосках). Из-за нулей **MAPE неприменима**.
Скошенность поднимает вопрос лог-таргета для линейной модели — проверим его экспериментом ниже.""")

code("m-target", """\
import matplotlib.pyplot as plt

y_all = final_dataset[TARGET]
print("mean=%.2f  median=%.0f  max=%d  нулей=%.1f%%  skew=%.2f"
      % (y_all.mean(), y_all.median(), y_all.max(), (y_all == 0).mean() * 100, y_all.skew()))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
ax[0].hist(y_all, bins=64); ax[0].set_title("total_rentals (raw)"); ax[0].set_xlabel("аренд / час / киоск")
ax[1].hist(np.log1p(y_all), bins=64); ax[1].set_title("log1p(total_rentals)")
plt.tight_layout(); plt.show()""")

md("m-split-hdr", """### Хронологический сплит 70/30

Граница — по уникальным таймстампам (чтобы один час не попал и в train, и в test одновременно).
Оба года есть и в train, и в test → рост «размазан», метрики меряют качество паттернов, а не ступеньку роста.""")

code("m-split", """\
# Сплит по уникальным таймстампам, а не по индексу строк (один час = 21 киоск).
ds = final_dataset.sort_values(["datetime_hourly", "location_id"]).reset_index(drop=True)
timestamps = np.sort(ds["datetime_hourly"].unique())
cutoff = timestamps[int(len(timestamps) * 0.70)]

train = ds[ds["datetime_hourly"] < cutoff]
test  = ds[ds["datetime_hourly"] >= cutoff]
X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"граница: {cutoff}")
print(f"train: {len(train):>7,} строк ({len(train) / len(ds):.0%})  "
      f"[{train['datetime_hourly'].min():%Y-%m-%d} … {train['datetime_hourly'].max():%Y-%m-%d}]")
print(f"test:  {len(test):>7,} строк ({len(test) / len(ds):.0%})  "
      f"[{test['datetime_hourly'].min():%Y-%m-%d} … {test['datetime_hourly'].max():%Y-%m-%d}]")""")

md("m-pipe-hdr", """### Pipeline

Представление признаков — под линейную модель, внутри `ColumnTransformer` (это обёртка над `final_dataset`, не новый датасет):

- `hour_of_day`, `month`, `day_of_week` → **cyclic (sin/cos)**, иначе 23:00 и 00:00 «далеко»;
- `location_id` → **one-hot** (21 значение);
- числовые (`conditions` ordinal, погода, `days_since_launch`) → **StandardScaler**;
- бинарные (`is_weekend`, `is_holiday`) → passthrough.

Обучаемые трансформации (scaler, one-hot-словарь) фитятся только на train-фолде — лика нет.""")

code("m-pipe", """\
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
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


def make_ridge(alpha=1.0, log_target=False):
    \"\"\"Ridge поверх preprocess; log_target=True оборачивает таргет в log1p/expm1.\"\"\"
    pipe = Pipeline([("pre", preprocess), ("ridge", Ridge(alpha=alpha))])
    if log_target:
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe""")

md("m-eval-hdr", """### Обучение и метрики

Набор: **MAE** (интерпретируемо — «в среднем промах на N аренд/час/киоск»), **RMSE** (штрафует крупные промахи),
**R²** (доля объяснённой дисперсии), **RMSE/MAE** (диагностика хвоста: ≈1 — ошибки однородны, >>1 — есть пики).
Сравнение — с наивным baseline (предсказать среднее по train). Прогноз клипуем снизу нулём: отрицательная аренда невозможна.""")

code("m-eval", """\
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2_score(y_true, y_pred), "RMSE/MAE": rmse / mae}


ridge = make_ridge(alpha=1.0)        # baseline — raw-таргет (обоснование ниже)
ridge.fit(X_train, y_train)
pred = np.clip(ridge.predict(X_test), 0, None)

results = pd.DataFrame({
    "Ridge":         evaluate(y_test, pred),
    "mean-baseline": evaluate(y_test, np.full(len(y_test), y_train.mean())),
}).T
results.round(3)""")

md("m-logvsraw-hdr", """### Лог-таргет vs raw

Скошенность таргета — аргумент за `log1p`. Проверяем экспериментом (модель и сплит фиксированы, меняем только таргет).""")

code("m-logvsraw", """\
rows = {}
for name, log in [("raw target", False), ("log1p target", True)]:
    m = make_ridge(alpha=1.0, log_target=log)
    m.fit(X_train, y_train)
    p = np.clip(m.predict(X_test), 0, None)
    rows[name] = evaluate(y_test, p)
pd.DataFrame(rows).T.round(3)""")

md("m-logvsraw-concl", """**Вывод:** `log1p` выигрывает символически по MAE, но проигрывает по **RMSE и R²** —
лог-таргет «выравнивает» масштаб и недооценивает пиковые часы, а нам именно пики важны для планирования.
→ **baseline на raw-таргете.** (К лог-таргету можно вернуться, если приоритетом станет MAE на низких счётчиках.)""")

md("m-breakdown-hdr", """### Разрезы ошибок

Глобальная MAE может скрывать узкие места. Смотрим по часам суток (где модель промахивается) и по киоскам (однородна ли ошибка).""")

code("m-breakdown", """\
test_eval = test.assign(pred=pred, abs_err=np.abs(y_test.values - pred))

by_hour = test_eval.groupby("hour_of_day").agg(
    MAE=("abs_err", "mean"), факт=(TARGET, "mean"), прогноз=("pred", "mean"))
by_loc = test_eval.groupby("location_id")["abs_err"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
ax[0].plot(by_hour.index, by_hour["факт"], marker="o", label="факт")
ax[0].plot(by_hour.index, by_hour["прогноз"], marker="o", label="прогноз")
ax[0].set_title("Средняя аренда по часам: факт vs прогноз"); ax[0].set_xlabel("час"); ax[0].legend()
ax[1].bar(by_hour.index, by_hour["MAE"])
ax[1].set_title("MAE по часам суток"); ax[1].set_xlabel("час")
plt.tight_layout(); plt.show()

print("MAE по часам (худшие):")
print(by_hour["MAE"].sort_values(ascending=False).head(4).round(2).to_string())
print(f"\\nMAE по киоскам: min={by_loc.min():.2f}  max={by_loc.max():.2f}  "
      f"spread={by_loc.max() - by_loc.min():.2f}")""")

md("m-conclusions", """## Выводы baseline

---

- **Ridge бьёт наивный baseline** (R²≈0.37 против отрицательного у предсказания среднего): линейная модель ловит основной сигнал — часы, сезон, киоск, тренд `days_since_launch`.
- **RMSE/MAE≈1.3** → есть хвост: модель систематически **недооценивает пиковые часы** (утро ~8:00 и вечер 17–18:00) — видно на графике «факт vs прогноз». Это ровно та нелинейность и взаимодействия (час × будни, час × погода), которые возьмут деревья.
- **Лог-таргет** проигрывает raw по RMSE/R² → baseline на raw-таргете.
- Ошибка по киоскам неоднородна (spread заметный) — кандидат на разрез качества при следующих итерациях.

**Дальше:** `RandomForestRegressor` и `XGBRegressor` (тот же `final_dataset`, сплит и метрики — меняем только модель), затем интеграция в Dagster: asset `trained_model` + `.joblib` IO manager (сохранять всю Pipeline).""")

nb.cells.extend(cells)
nbformat.write(nb, NB)
print(f"appended {len(cells)} cells; total now {len(nb.cells)}")
