# XGBoost

> Конспект для week-3. Что такое gradient boosting на деревьях, почему XGBoost — индустриальный стандарт, и какие гиперпараметры реально крутить.

## 1. Контекст — ансамбли деревьев

Одно дерево решений (decision tree) — слабая модель: либо underfit (мелкое дерево), либо overfit (глубокое). Идея ансамблей — объединить много слабых моделей в сильную. Два способа:

### Bagging — Random Forest (RF)

- Тренируем **N деревьев параллельно**.
- Каждое — на bootstrap-выборке (random sample с возвратом).
- Каждое — на случайном подмножестве фич (random feature subset) на каждом split-е.
- Финальное предсказание — **усреднение** (для регрессии) или **голосование** (для классификации).

Эффект: разные деревья ошибаются по-разному, ансамбль усредняет дисперсию (variance reduction). Bias почти не меняется — RF не делает каждое дерево «лучше», она их **независимыми**.

### Boosting — Gradient Boosting

- Тренируем **N деревьев последовательно** (sequentially).
- Каждое следующее дерево учится исправлять ошибки предыдущих.
- Финальное предсказание — **сумма** предсказаний всех деревьев (с весами).

Эффект: каждое дерево уменьшает bias (систематическую ошибку), которая осталась после предыдущих. Variance растёт с числом деревьев — отсюда нужна регуляризация.

## 2. Gradient boosting на пальцах

Шаг 1. Начальное предсказание `F₀(x)` — например, среднее по таргету:
```
F₀(x) = mean(y)
```

Шаг 2. Вычисляем остатки (residuals) — реальные минус предсказанные:
```
r_i = y_i - F₀(x_i)
```

Шаг 3. Обучаем дерево `h₁(x)` **на остатках** (не на исходном таргете). Дерево пытается выучить, где модель ошибается.

Шаг 4. Обновляем предсказание:
```
F₁(x) = F₀(x) + η · h₁(x)
```

где `η` (eta) — **learning rate**: насколько сильно учитываем новое дерево. Малое `η` (0.01..0.1) — медленное обучение, но более стабильное.

Шаг 5. Повторяем: считаем новые остатки относительно `F₁`, обучаем `h₂`, и так далее до `N` деревьев:
```
F_N(x) = F₀(x) + η · (h₁ + h₂ + ... + h_N)
```

**Почему «gradient»:** в общем случае вместо «остатков» используется **отрицательный градиент функции потерь** (negative gradient of loss). Для MSE этот градиент совпадает с остатками `(y - ŷ)`. Для других loss (Huber, Poisson, log-loss) — другие формулы. Boosting работает с любым дифференцируемым loss.

## 3. Что XGBoost добавляет к классическому GB

**XGBoost = eXtreme Gradient Boosting.** Главные улучшения:

### (a) Регуляризация на структуре дерева

В обычном GB дерево минимизирует только loss. XGBoost оптимизирует:
```
obj = loss + Ω(tree)
Ω(tree) = γ · T + (1/2) · λ · Σ w_j²
```
где `T` — число листьев, `w_j` — вес в j-м листе.

- `γ` (gamma) — штраф за каждый лист. Контролирует «можно ли вообще делать split». Если выигрыш от split-а меньше `γ` — split отбрасывается.
- `λ` (reg_lambda) — L2-регуляризация на веса листьев. Снижает уверенность отдельных предсказаний.
- `reg_alpha` — L1-регуляризация (sparse weights).

Это делает деревья «дисциплинированными» и снижает overfitting.

### (b) Второй порядок производных (Newton's method)

Классический GB использует только градиент (первая производная). XGBoost использует **градиент + гессиан** (Hessian, вторая производная). Это позволяет точнее найти оптимальный split — аналог метода Ньютона vs gradient descent.

### (c) Эффективная обработка пропусков (sparsity-aware split)

XGBoost умеет работать с NaN напрямую. На каждом split-е помимо порога `x_j > threshold` модель выбирает «куда отправлять NaN» — влево или вправо. Это решается из данных, оптимизируется как часть split-а.

В sklearn `RandomForestRegressor` NaN не поддерживаются — нужно явно импьютить. XGBoost — нет.

### (d) Параллелизация и эффективность

Классический GB — последовательный по деревьям, но **внутри одного дерева** XGBoost параллелит поиск split-а по фичам и сэмплам. На многоядерной машине это серьёзный буст.

### (e) Out-of-core / GPU / распределённые режимы

XGBoost умеет работать с данными, не помещающимися в RAM (out-of-core), а также на GPU (`tree_method="hist"` + `device="cuda"`). Для нашего объёма это неактуально, но полезно знать.

## 4. Ключевые гиперпараметры — что и зачем

Параметры разделены по группам.

### Структура дерева (контроль bias-variance отдельного дерева)

| Параметр | Смысл | Дефолт | Куда крутить |
|---|---|---|---|
| `max_depth` | Максимальная глубина дерева | 6 | 3-10. Большая глубина → variance, переобучение. Малая → underfit. |
| `min_child_weight` | Мин. сумма гессианов в листе (≈ мин. число сэмплов) | 1 | Увеличить (5-100) → меньше overfit |
| `gamma` (`min_split_loss`) | Мин. выигрыш для split-а | 0 | Увеличить (0.1-1) → больше консерватизма |
| `max_leaves` | Альтернатива `max_depth` (рост по листьям, не по уровням) | 0 (отключено) | Используется при `grow_policy="lossguide"` |

### Boosting-процесс

| Параметр | Смысл | Дефолт | Куда крутить |
|---|---|---|---|
| `n_estimators` | Число деревьев | 100 | Зависит от `learning_rate`. С early stopping (см. ниже) задавать большое (1000-5000), но реально использовано столько, сколько нужно. |
| `learning_rate` (`eta`) | Вес каждого нового дерева | 0.3 | Снижать до 0.05-0.1 и компенсировать `n_estimators`. Меньше eta + больше деревьев → стабильнее, но дольше. |

### Стохастичность (защита от overfit)

| Параметр | Смысл | Дефолт | Куда крутить |
|---|---|---|---|
| `subsample` | Доля row-sample для каждого дерева | 1.0 | 0.7-0.9 — bagging-эффект внутри boosting |
| `colsample_bytree` | Доля фич для каждого дерева | 1.0 | 0.7-0.9 — особенно полезно при много фичах |
| `colsample_bylevel`, `colsample_bynode` | То же на уровне split-а | 1.0 | Обычно дефолт |

### Регуляризация

| Параметр | Смысл | Дефолт | Куда крутить |
|---|---|---|---|
| `reg_lambda` | L2 на веса листьев | 1.0 | 1-10 |
| `reg_alpha` | L1 на веса листьев | 0.0 | 0-1; обычно reg_lambda хватает |

### Loss и задача

| Параметр | Смысл | Дефолт | Для регрессии |
|---|---|---|---|
| `objective` | loss-функция | `"reg:squarederror"` | По умолчанию MSE; альтернативы — `"reg:absoluteerror"` (MAE), `"reg:pseudohubererror"`, `"count:poisson"` (для count-данных, наш случай!), `"reg:gamma"` |
| `eval_metric` | Метрика для eval (early stopping) | По objective | `"rmse"`, `"mae"`, `"poisson-nloglik"` |

**Важно для bike-rental:** `total_rentals` — это **count** (натуральные числа, неотрицательные, есть нули). Чисто формально MSE-loss не идеален: он может предсказывать отрицательные значения, штрафовать симметрично «10 vs 9» и «10 vs 11». Альтернативы:
- `objective="count:poisson"` — Пуассоновская регрессия. Предсказание всегда ≥ 0, штраф асимметричный (как для count-данных). Полезно попробовать.
- `objective="reg:squarederror"` на `log1p(y)` (логарифмированном таргете), потом `expm1` обратно. Универсальный приём для длиннохвостых таргетов.

## 5. Early stopping — критично

Идея: не задавать `n_estimators` руками, а **остановиться, когда метрика на eval-выборке перестала улучшаться**.

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=5000,         # большой потолок
    learning_rate=0.05,
    max_depth=6,
    early_stopping_rounds=50,  # если 50 раундов нет улучшения — стоп
    eval_metric="rmse",
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],  # отдельный validation-сет
    verbose=False,
)

print(f"Использовано деревьев: {model.best_iteration}")
```

**Почему это лучше, чем CV для подбора `n_estimators`:**
- Early stopping использует один проход — гораздо быстрее, чем CV.
- Возвращает оптимальное число деревьев автоматически.
- Гарантирует, что модель не переобучилась на момент сохранения.

**Важно:** `eval_set` должен быть **отдельным от test**. Иначе early stopping подгоняет под test, и метрика на test становится оптимистичной. Схема: train → val (для early stopping) → test (финальный замер).

Для time-series: val и test — оба позже train по времени.

## 6. Практический порядок тюнинга

Тюнить все параметры одновременно — комбинаторный взрыв. Стандартный порядок (Aarshay Jain’s approach, базовая практика):

1. **Зафиксировать `learning_rate=0.1`**, найти оптимальный `n_estimators` через early stopping.
2. **Подкрутить `max_depth` и `min_child_weight`** (контроль bias-variance дерева).
3. **`gamma`** — отсечь шумные splits.
4. **`subsample` и `colsample_bytree`** — стохастичность.
5. **`reg_lambda` / `reg_alpha`** — финальная регуляризация.
6. **Снизить `learning_rate` до 0.01-0.05**, поднять `n_estimators` соответственно (с early stopping). Это финальный «полировальный» шаг.

Каждый шаг — Grid Search или Random Search на 2-3 значениях, с CV (TimeSeriesSplit для нас).

**На неделю-3:** ограничиться шагами 1-2. Полный тюнинг — отдельная работа.

## 7. Сравнение с альтернативами

| Метод | Скорость | Точность | Память | Когда выбирать |
|---|---|---|---|---|
| `LinearRegression` / `Ridge` | очень быстро | базовая | малая | baseline, интерпретация |
| `RandomForestRegressor` | средне | хорошая | большая (хранит все деревья целиком) | минимальный тюнинг, хороший дефолт |
| `XGBoost` | быстро | очень хорошая | средняя | главный workhorse |
| `LightGBM` | очень быстро | сопоставимо XGB | малая | для больших данных, leaf-wise growth |
| `CatBoost` | средне | сопоставимо XGB | средняя | если много категориальных фич без encoding |

Для bike-rental (~365k строк, ~15 фич) — XGBoost подходит идеально. Если будут проблемы со скоростью на тюнинге — посмотри LightGBM.

## 8. API: sklearn-style vs native

XGBoost имеет два API:

### Sklearn-style (`XGBRegressor`)

```python
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

# Работает в sklearn Pipeline / GridSearchCV напрямую
```

Рекомендуется для большинства случаев — интегрируется с sklearn-инфраструктурой (Pipeline, GridSearchCV, ColumnTransformer).

### Native API (`xgb.train` + `DMatrix`)

```python
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)

params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.05,
    "max_depth": 6,
}

booster = xgb.train(
    params, dtrain,
    num_boost_round=1000,
    evals=[(dval, "val")],
    early_stopping_rounds=50,
)
```

Используется когда нужен максимальный контроль или продвинутые фичи (callbacks, custom objective). Для week-3 — берём sklearn-style.

## 9. Feature importance — что выучила модель

```python
import pandas as pd

importances = pd.Series(
    model.feature_importances_,
    index=X_train.columns,
).sort_values(ascending=False)
print(importances.head(10))
```

XGBoost даёт три типа importance:
- `weight` — сколько раз фича использовалась в split-ах.
- `gain` — средний выигрыш loss при split-ах по этой фиче (важнее по сути).
- `cover` — средний охват (число сэмплов, попавших в split-ы по этой фиче).

По умолчанию `feature_importances_` это `gain`. Это лучший показатель «важности».

Для глубокого понимания — **SHAP values** (отдельная библиотека `shap`), показывает вклад каждой фичи в каждое индивидуальное предсказание. Для week-3 — необязательно, но полезно знать.

## 10. Резюме для bike-rental

**Минимум для week-3:**
```python
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",   # попробовать "count:poisson"
    eval_metric="rmse",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
y_pred = model.predict(X_test)

print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")
print(f"R²:   {r2_score(y_test, y_pred):.3f}")
```

**Что попробовать на week-3:**
- `objective="reg:squarederror"` vs `objective="count:poisson"` — две модели, сравнить.
- Подбор `max_depth` в [4, 6, 8].
- Feature importance — увидеть, что модель сама считает важным (мы ожидаем `hour_of_day`, `temperature_c`, `is_weekend` в топе).

**Что **не** делать на week-3:**
- Полный grid search по 10 параметрам.
- Stacking / blending моделей.
- SHAP — для understanding это полезно, но не для acceptance criteria.
