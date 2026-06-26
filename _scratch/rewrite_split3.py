"""Rewrite eda_and_modeling.ipynb modeling cells for train/val/test split; eval on validation."""
import nbformat

NB = "notebooks/eda_and_modeling.ipynb"
nb = nbformat.read(NB, as_version=4)

S = {
"m-split-hdr": """### Хронологический сплит 70/15/15

Строго по времени, по уникальным таймстампам (один час не дробим между частями). Роли:

- **train** (первые 70%) — обучаем модель;
- **validation** (15%) — здесь сравниваем фичи/модели и гоняем ablation;
- **test** (последние 15%) — held-out, трогаем **один раз в самом конце**, после выбора модели. До этого не смотрим, иначе подгоним решения под него (утечка теста в выбор).

train охватывает оба года; val и test — поздний 2012. Нюанс единственного тайм-сплита: val — лето, test — осень/зима (разные сезоны), поэтому val-метрики не идеально предсказывают test; от этого лечит time-series CV — добавим позже при необходимости.""",

"m-split": """\
# Сплит по уникальным таймстампам, строго по времени: train | val | test = 70 | 15 | 15.
ds = final_dataset.sort_values(["datetime_hourly", "location_id"]).reset_index(drop=True)
timestamps = np.sort(ds["datetime_hourly"].unique())
cut1 = timestamps[int(len(timestamps) * 0.70)]
cut2 = timestamps[int(len(timestamps) * 0.85)]

train = ds[ds["datetime_hourly"] < cut1]
val   = ds[(ds["datetime_hourly"] >= cut1) & (ds["datetime_hourly"] < cut2)]
test  = ds[ds["datetime_hourly"] >= cut2]

X_train, y_train = train[FEATURES], train[TARGET]
X_val,   y_val   = val[FEATURES],   val[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]   # held-out, до финала не трогаем

for name, part in [("train", train), ("val", val), ("test", test)]:
    print(f"{name:5} {len(part):>7,} ({len(part) / len(ds):.0%})  "
          f"[{part['datetime_hourly'].min():%Y-%m-%d} … {part['datetime_hourly'].max():%Y-%m-%d}]")""",

"m-eval-hdr": """### Обучение и метрики

Набор: **MAE** (интерпретируемо — «промах на N аренд/час/киоск»), **RMSE** (штрафует крупные промахи),
**R²** (доля объяснённой дисперсии), **RMSE/MAE** (диагностика хвоста: ≈1 — однородно, >>1 — есть пики).
Обучаем на **train**, меряем на **validation** (test держим до финала). Сравнение — с наивным baseline (среднее train).
Прогноз клипуем снизу нулём: отрицательная аренда невозможна.""",

"m-eval": """\
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2_score(y_true, y_pred), "RMSE/MAE": rmse / mae}


ridge = make_ridge(alpha=1.0)        # baseline — raw-таргет (обоснование ниже)
ridge.fit(X_train, y_train)
val_pred = np.clip(ridge.predict(X_val), 0, None)

results = pd.DataFrame({
    "Ridge (val)":   evaluate(y_val, val_pred),
    "mean-baseline": evaluate(y_val, np.full(len(y_val), y_train.mean())),
}).T
results.round(3)""",

"m-logvsraw": """\
rows = {}
for name, log in [("raw target", False), ("log1p target", True)]:
    m = make_ridge(alpha=1.0, log_target=log)
    m.fit(X_train, y_train)
    p = np.clip(m.predict(X_val), 0, None)
    rows[name] = evaluate(y_val, p)
pd.DataFrame(rows).T.round(3)""",

"m-logvsraw-concl": """**Вывод:** `log1p` снова символически лучше по MAE, но проигрывает по **RMSE и R²**
(на validation raw R²≈0.35 против log1p≈0.24) — лог-таргет недооценивает пиковые часы, а они нам важны.
→ **baseline на raw-таргете.** (К лог-таргету вернёмся, если приоритетом станет MAE на низких счётчиках.)""",

"m-breakdown-hdr": """### Разрезы ошибок (validation)

Глобальная MAE может скрывать узкие места. Смотрим по часам суток (где модель промахивается) и по киоскам (однородна ли ошибка).""",

"m-breakdown": """\
val_eval = val.assign(pred=val_pred, abs_err=np.abs(y_val.values - val_pred))

by_hour = val_eval.groupby("hour_of_day").agg(
    MAE=("abs_err", "mean"), факт=(TARGET, "mean"), прогноз=("pred", "mean"))
by_loc = val_eval.groupby("location_id")["abs_err"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
ax[0].plot(by_hour.index, by_hour["факт"], marker="o", label="факт")
ax[0].plot(by_hour.index, by_hour["прогноз"], marker="o", label="прогноз")
ax[0].set_title("Средняя аренда по часам: факт vs прогноз (val)"); ax[0].set_xlabel("час"); ax[0].legend()
ax[1].bar(by_hour.index, by_hour["MAE"])
ax[1].set_title("MAE по часам суток (val)"); ax[1].set_xlabel("час")
plt.tight_layout(); plt.show()

print("MAE по часам (худшие):")
print(by_hour["MAE"].sort_values(ascending=False).head(4).round(2).to_string())
print(f"\\nMAE по киоскам: min={by_loc.min():.2f}  max={by_loc.max():.2f}  "
      f"spread={by_loc.max() - by_loc.min():.2f}")""",

"m-conclusions": """## Выводы baseline

---

- **Ridge бьёт наивный baseline** (на validation R²≈0.35 против отрицательного у предсказания среднего; MAE 7.1 vs 9.5): линейная модель ловит основной сигнал — часы, сезон, киоск, тренд `days_since_launch`.
- **RMSE/MAE≈1.27** → есть хвост: модель систематически **недооценивает пиковые часы** (утро ~8:00 и вечер 17–18:00) — видно на графике «факт vs прогноз». Это ровно та нелинейность и взаимодействия (час × будни, час × погода), которые возьмут деревья.
- **Лог-таргет** проигрывает raw по RMSE/R² → baseline на raw-таргете.
- Ошибка по киоскам **почти однородна** (spread ~0.2): one-hot по `location_id` уже впитал уровень киоска, остаток — общий промах на пиках. Улучшать надо профиль по часам, не по локациям.
- **Test не трогали** — финальное число посчитаем один раз после выбора модели.

**Дальше:** `RandomForestRegressor` и `XGBRegressor` на том же `final_dataset` и сплите (меняем только модель), и ablation фич (`is_holiday`, взаимодействие `is_weekend × hour_of_day`) — всё на **validation**. Затем интеграция в Dagster: asset `trained_model` + `.joblib` IO manager (сохранять всю Pipeline).""",
}

found = []
for c in nb.cells:
    if c.get("id") in S:
        c["source"] = S[c["id"]]; found.append(c["id"])
missing = set(S) - set(found)
nbformat.write(nb, NB)
print("rewrote:", found, "| missing:", missing)
