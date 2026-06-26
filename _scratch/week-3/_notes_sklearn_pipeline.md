# sklearn Pipeline & ColumnTransformer

> Конспект для week-3. Как собрать препроцессинг + модель в один объект, чтобы не получить лик и легко сериализовать.

## 1. Зачем pipeline — три причины

### (a) Защита от data leakage на препроцессинге

Самая частая ошибка новичков:

```python
# ПЛОХО — лик на препроцессинге
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # fit на ВСЕХ данных, включая будущий test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

model.fit(X_train, y_train)
```

Здесь `StandardScaler.fit(X)` использует среднее и стандартное отклонение **всего** датасета — включая test. Это значит test «протёк» в препроцессинг, и финальная оценка завышена.

Pipeline решает это автоматически:

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  Ridge()),
])

X_train, X_test, y_train, y_test = train_test_split(X, y)
pipeline.fit(X_train, y_train)   # fit_transform на train, fit на train
pipeline.score(X_test, y_test)   # transform на test (не fit!), predict на test
```

Pipeline гарантирует: на каждом этапе `fit_transform` зовётся только для train, для test/predict — только `transform`. **Это та же самая защита, что в `TimeSeriesSplit` — и она работает автоматически только если препроцессинг внутри pipeline.**

### (b) Один объект — один артефакт

Сериализуешь `pipeline` целиком, и в продакшене делаешь:
```python
loaded.predict(new_raw_data)   # сырые данные → препроцессинг → модель → ответ
```

Без pipeline: пришлось бы сохранять отдельно scaler, encoder, model — и в продакшене руками их в правильном порядке применять. Каждый шаг — место для ошибки.

### (c) GridSearchCV видит весь конвейер

Можешь тюнить гиперпараметры **препроцессинга и модели вместе**:
```python
param_grid = {
    "scaler__with_mean": [True, False],   # параметр шага scaler
    "model__alpha":      [0.1, 1.0, 10],  # параметр шага model
}
GridSearchCV(pipeline, param_grid, ...).fit(X_train, y_train)
```

Двойное подчёркивание `step__param` — стандартный sklearn-синтаксис для адресации параметра внутри шага.

## 2. `Pipeline` — синтаксис

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  Ridge(alpha=1.0)),
])
```

- Список **(name, transformer)** пар.
- Все шаги, кроме последнего, должны быть **transformers** (имеют `fit`+`transform`).
- Последний шаг — обычно **estimator** (имеет `fit`+`predict`), но может быть и transformer, если pipeline сам трансформер.

### `make_pipeline` — короче, но без имён

```python
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(StandardScaler(), Ridge())
# имена сгенерируются автоматически: "standardscaler", "ridge"
```

Удобно для быстрых экспериментов, но в GridSearch адресовать параметры неудобно (зависит от сгенерированных имён). Для серьёзного кода — `Pipeline` с явными именами.

### Доступ к отдельным шагам

```python
pipeline.named_steps["scaler"]       # объект StandardScaler
pipeline["scaler"]                   # эквивалентно
pipeline.steps[0]                    # tuple ("scaler", StandardScaler())
```

## 3. `ColumnTransformer` — разные шаги для разных колонок

Реальная задача: к **числовым** фичам — StandardScaler, к **категориальным** — OneHotEncoder, к **датам** — sin/cos. Pipeline один на всё — не подходит. Нужен `ColumnTransformer`.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import numpy as np

def sin_cos(x, period):
    return np.column_stack([
        np.sin(2 * np.pi * x / period),
        np.cos(2 * np.pi * x / period),
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["temperature_c", "perceived_temperature_c", "humidity", "windspeed_kmh"]),
        ("hour", FunctionTransformer(sin_cos, kw_args={"period": 24}), ["hour_of_day"]),
        ("month", FunctionTransformer(sin_cos, kw_args={"period": 12}), ["month"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["conditions", "location_id"]),
    ],
    remainder="passthrough",  # что делать с колонками не из списка
)
```

**Аргументы:**
- Список **(name, transformer, columns)** triples.
- `columns` может быть список имён, индексов, или slice. **Используй имена** — устойчиво к перестановке колонок.
- `remainder` — что делать с колонками, не попавшими в transformers:
  - `"drop"` (дефолт) — выкинуть.
  - `"passthrough"` — оставить как есть (например, `is_holiday`, `is_weekend` — уже бинарные, не нужно трогать).
  - Можно передать любой transformer.

### ColumnTransformer внутри Pipeline

```python
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model",      XGBRegressor(...)),
])

pipeline.fit(X_train, y_train)
```

Теперь `pipeline` принимает **сырой DataFrame** и сам делает scaling, encoding, sin/cos, прогон через модель.

## 4. Сохранение имён фич после трансформации

После `ColumnTransformer` + `OneHotEncoder` колонок становится больше (one-hot развёртывает), их имена меняются. Чтобы понять, какая фича где:

```python
preprocessor.fit(X_train)
preprocessor.get_feature_names_out()
# array(['num__temperature_c', 'num__humidity', ..., 'cat__conditions_clear', 'cat__conditions_clouds', ...])
```

Доступно для всего pipeline:
```python
pipeline.named_steps["preprocess"].get_feature_names_out()
```

Это критично для **feature importance** в XGBoost — иначе не поймёшь, какая фича на какой позиции.

## 5. Кастомные трансформеры

### Простой через `FunctionTransformer`

Для stateless-функций (не нужен fit, ничего не запоминать):

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

log_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=False)
```

Для bike-rental: log-преобразование таргета через `TransformedTargetRegressor` (см. ниже).

### Полноценный через наследование

Если нужен **state** (что-то сохранить в fit, использовать в transform):

```python
from sklearn.base import BaseEstimator, TransformerMixin

class LagFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, lag_hours=24):
        self.lag_hours = lag_hours

    def fit(self, X, y=None):
        # ничего не учим, просто запомнить
        return self

    def transform(self, X):
        X = X.copy()
        X[f"lag_{self.lag_hours}h"] = X.groupby("location_id")["total_rentals"].shift(self.lag_hours)
        return X
```

Шаблон: `BaseEstimator + TransformerMixin`, методы `fit(X, y=None)` и `transform(X)`. `TransformerMixin` даст бесплатно `fit_transform`.

Замечание: lag-фичи **зависят от таргета** и от группировки — это очень аккуратное место, легко получить лик. На week-3 проще делать lag-фичи в Dagster asset до pipeline, чем кастомным трансформером.

## 6. Трансформация таргета

Иногда нужно **трансформировать `y`** перед обучением (например, `log1p` для длиннохвостого таргета):

```python
from sklearn.compose import TransformedTargetRegressor
import numpy as np

model = TransformedTargetRegressor(
    regressor=Ridge(alpha=1.0),
    func=np.log1p,
    inverse_func=np.expm1,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)   # уже в исходном масштабе (после inverse_func)
```

Что внутри: `y_train_log = log1p(y_train)` → `regressor.fit(X_train, y_train_log)` → `predict` даёт log-предсказания → `expm1` обратно.

**Полезно для bike-rental:** `total_rentals` распределён длиннохвосто (большинство часов мало, пиковые — сильно больше). `log1p` (это `log(1 + x)`, корректно для нулей) выравнивает распределение, и линейные/MSE-модели часто работают лучше.

## 7. Воспроизводимость — `random_state`

Все sklearn-компоненты с рандомизацией принимают `random_state`. Зафиксируй везде:

```python
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model",      XGBRegressor(random_state=42)),
])
```

`train_test_split` тоже принимает `random_state`. Без этого результаты «плавают» между запусками.

## 8. Кэширование шагов — `memory`

Если препроцессинг тяжёлый, а ты крутишь GridSearch по параметрам **модели** — `Pipeline(..., memory="cache_dir")` кэширует промежуточные результаты препроцессинга:

```python
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model",      XGBRegressor(...)),
], memory="./.sklearn_cache")
```

Для week-3 — необязательно, но полезно знать для будущих более тяжёлых пайплайнов.

## 9. Сборка для bike-rental — концептуально

```python
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from xgboost import XGBRegressor
import numpy as np

NUM_FEATURES = ["temperature_c", "perceived_temperature_c", "humidity", "windspeed_kmh"]
CYC_FEATURES = {"hour_of_day": 24, "month": 12, "day_of_week": 7}
CAT_FEATURES = ["conditions", "location_id"]
BIN_FEATURES = ["is_weekend", "is_holiday"]

def sin_cos(x, period):
    return np.column_stack([
        np.sin(2 * np.pi * x / period),
        np.cos(2 * np.pi * x / period),
    ])

# Для XGBoost (tree-based) — без cyclic, без scaling, просто ordinal + passthrough
preprocessor_tree = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ],
    remainder="passthrough",
)

pipeline_tree = Pipeline([
    ("preprocess", preprocessor_tree),
    ("model", XGBRegressor(
        n_estimators=2000, learning_rate=0.05, max_depth=6,
        random_state=42, n_jobs=-1,
    )),
])

# Для Ridge — со scaling и cyclic
preprocessor_linear = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_FEATURES),
        *[
            (f"cyc_{name}", FunctionTransformer(sin_cos, kw_args={"period": period}), [name])
            for name, period in CYC_FEATURES.items()
        ],
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ],
    remainder="passthrough",
)

pipeline_linear = TransformedTargetRegressor(
    regressor=Pipeline([
        ("preprocess", preprocessor_linear),
        ("model", Ridge(alpha=1.0)),
    ]),
    func=np.log1p,
    inverse_func=np.expm1,
)
```

Два разных конвейера для двух типов моделей — но оба одинаково используются `.fit(X, y)` / `.predict(X)`.

## 10. Подводные камни

### (a) Pandas vs numpy на выходе

По умолчанию `ColumnTransformer.transform` возвращает **numpy array** — теряются имена колонок. Можно получить DataFrame:

```python
from sklearn import set_config
set_config(transform_output="pandas")
```

Глобальная настройка sklearn (>= 1.2). Полезно для дебага. В проде — лучше numpy (быстрее).

### (b) OneHotEncoder и неизвестные категории

Если в test-выборке появится новое значение `location_id` (которого не было в train), `OneHotEncoder` без `handle_unknown="ignore"` упадёт. **Всегда указывай `handle_unknown="ignore"`** — оно превратит неизвестную категорию в нули по всем dummies.

Для bike-rental: `location_id` фиксированный набор (0..20), но привычка должна быть.

### (c) Sparse output

`OneHotEncoder(sparse_output=True)` (дефолт) возвращает разрежённую матрицу. Это хорошо для линейной модели, но XGBoost любит плотные данные. `OneHotEncoder(sparse_output=False)` — если будут проблемы со скоростью.

### (d) ColumnTransformer не переставляет колонки в исходном порядке

Выходные колонки идут **в порядке transformers**, не в исходном порядке. Не полагайся на порядок — используй `get_feature_names_out()`.

## 11. Резюме

- **Всегда** оборачивай препроцессинг + модель в `Pipeline`. Это защита от лика и единый артефакт для сериализации.
- Для **разных типов колонок** — `ColumnTransformer`. Для разных моделей — разные ColumnTransformer (cyclic для линейных, ordinal для деревьев).
- Сложный препроцессинг (`log1p` таргета) — через `TransformedTargetRegressor`.
- `random_state=42` везде где есть рандом.
- `handle_unknown="ignore"` для всех encoder-ов.
- Pipeline сохраняется как один объект (`joblib.dump(pipeline, ...)`), а не три отдельных артефакта (см. [[_notes_persistence_joblib]]).
