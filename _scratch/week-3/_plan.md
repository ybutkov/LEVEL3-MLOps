# Week 3 — план

> Источник: `bike-rental/handout/week-3.md`. Сделано после ревью EDA-ноутбука недели 2.

## Цель

Расширить пайплайн ML-частью: от `final_dataset` (week-2) → обученная регрессионная модель, встроенная в Dagster как воспроизводимый шаг.

---

## Фазы

### 1. EDA на готовом датасете

Новый ноутбук, вход — `final_dataset` (~365k строк × 15 колонок).

- Распределение таргета `total_rentals` (много нулей? хвост?)
- Корреляции / визуализации vs таргета по: `hour_of_day`, `day_of_week`, `month`, `is_weekend`, `is_holiday`, `conditions`, `temperature_c`, `humidity`, `windspeed_kmh`, `location_id`
- Двумерные срезы: `hour_of_day × day_of_week` heatmap; `month × hour_of_day`; `location_id × hour_of_day`
- Стабильность во времени: 2011 vs 2012 (см. `_exploration_notes.md` — рост сервиса +65%)

Решение по выходу: какие фичи реально работают и какие нужны дополнительно.

### 2. Baseline-регрессия

Самая простая и быстрая модель. На выбор: `LinearRegression` или `Ridge` (с регуляризацией безопаснее).

- Train/test split, **time-aware** (см. [[_notes_time_series_split]])
- Метрики: MAE, RMSE, R² (см. [[_notes_regression_metrics]])
- Зафиксировать baseline-числа как точку отсчёта

### 3. Итеративное улучшение (time-boxed)

Time-box обязательно, иначе не хватит на интеграцию. Порядок:

1. **Feature engineering первого уровня:** cyclic encoding для `hour_of_day` / `month` / `day_of_week` (см. [[_notes_cyclic_encoding]])
2. **Tree-based baseline:** `RandomForestRegressor` — обычно сильный prior без тюнинга
3. **XGBoost:** `XGBRegressor` с early stopping (см. [[_notes_xgboost]])
4. **Feature engineering второго уровня (если останется время):** лаги (`total_rentals` за прошлый час / прошлую неделю), агрегаты по локации, year как фича для роста сервиса

Всё через `sklearn.Pipeline` + `ColumnTransformer` (см. [[_notes_sklearn_pipeline]]).

### 4. Интеграция в Dagster

Из ноутбука выкристаллизовать в assets:

- Если появились новые признаки → расширить существующие data-assets или создать новые feature-assets
- Новый asset `trained_model` — берёт `final_dataset`, train/test split, обучает Pipeline, возвращает сериализованный объект
- IO manager для `.joblib` (по аналогии с `csv_io.py`)
- Сохранение: вся Pipeline целиком, не только модель (см. [[_notes_persistence_joblib]])
- Метаданные: метрики на test, версии sklearn/xgboost, размер модели

---

## Что изучить — конспекты в этой папке

| Тема | Файл | Зачем |
|---|---|---|
| Регрессионные метрики | `_notes_regression_metrics.md` | Выбрать метрики, понять что они показывают и где врут |
| Time-series split | `_notes_time_series_split.md` | Не получить ложно-завышенный score из-за лика по времени |
| Cyclic encoding | `_notes_cyclic_encoding.md` | Правильно закодировать час/месяц/день недели для линейных моделей |
| XGBoost | `_notes_xgboost.md` | Что такое gradient boosting, основные гиперпараметры, early stopping |
| sklearn Pipeline / ColumnTransformer | `_notes_sklearn_pipeline.md` | Один объект fit/predict, защита от лика на препроцессинге |
| Persistence (joblib) | `_notes_persistence_joblib.md` | Как правильно сохранить и грузить обученные модели |

---

## Открытые вопросы (решить по ходу)

- [ ] **Один таргет или два?** `total_rentals` vs отдельно `registered_rentals` + `direct_pickups`. По `_exploration_notes.md` сегменты разные, стоит хотя бы проверить корреляцию остатков.
- [ ] **Гранулярность:** предсказывать по `(hour, location_id)` или агрегировать в `(hour)` и игнорировать локации? Если по `(hour, location_id)` — `location_id` категориальный (one-hot / target encoding) или ordinal?
- [ ] **Time-aware split:** train на 2011, test на 2012 — это "правда жизни", но рост +65% делает задачу искусственно сложной. Альтернатива: train на 70% хронологии, test на 30%. Выбор зависит от того, что хотим измерить.
- [ ] **Holiday-фичи:** оставить `is_holiday` или попробовать `holiday_name` с target encoding (см. `_holiday_modeling_backlog.md`)?

---

## Acceptance criteria (из handout)

- [ ] Notebook(ы) с EDA + baseline + эксперименты
- [ ] Расширенный Dagster pipeline с препроцессингом и тренировкой
- [ ] Pipeline сохраняет модель как сериализованный объект
- [ ] Notebook и pipeline запускаются end-to-end
- [ ] Кодовая база остаётся читаемой; новые признаки идут в upstream assets

---

## Перед началом работы

```bash
cd bike-rental && uv sync
```

(`.venv` был удалён при освобождении диска — нужно пересоздать.)
