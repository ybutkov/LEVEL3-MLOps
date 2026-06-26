# Persistence: сохранение моделей через joblib

> Конспект для week-3. Как правильно сохранить обученную модель, как её безопасно грузить, и как встроить это в Dagster.

## 1. Pickle vs joblib — что выбирать

### Pickle — стандартный Python

`pickle` — это встроенный Python-механизм сериализации **любого** Python-объекта в байтовый поток. Работает с моделями sklearn, но:

- Не оптимизирован для **больших numpy-массивов**. Внутри модели sklearn веса лежат как numpy — pickle сериализует их сравнительно медленно.
- Не делает компрессию (нужно отдельно через `gzip`).

### joblib — оптимизация для numpy

`joblib.dump` / `joblib.load` — это надстройка над pickle, **специально оптимизированная** для объектов с большими numpy-массивами:

- Сохраняет numpy через memory-mapping вместо «честной» сериализации → в разы быстрее на больших моделях (Random Forest с 1000 деревьями, XGBoost с большими структурами).
- Встроенная компрессия (`compress=N`, где N — уровень сжатия).
- Параллельная загрузка для разделённых массивов.

**Правило:** для всего, что построено на numpy/scipy/sklearn — **joblib**. Для общего Python-кода — pickle. Для production / cross-language обмена — лучше альтернативы (см. §8).

## 2. Базовый API

```python
import joblib

# Сохранить
joblib.dump(pipeline, "model.joblib")

# Загрузить
loaded = joblib.load("model.joblib")
y_pred = loaded.predict(X_new)
```

Файл расширения `.joblib` или `.pkl` — соглашение, не требование.

## 3. Компрессия — `compress`

```python
joblib.dump(pipeline, "model.joblib.gz", compress=3)
```

- `compress=0` — без сжатия (быстро, большой файл).
- `compress=3` — баланс скорости и размера (обычно дефолт de facto).
- `compress=9` — максимальное сжатие (медленно, малый файл).

Для XGBoost-модели с 2000 деревьями: `compress=0` ≈ 50 MB, `compress=3` ≈ 10 MB. Загрузка занимает чуть дольше, но дисковое место экономится. **Для week-3 ставь `compress=3`.**

Поддерживаемые форматы: gzip (`.gz`), bz2 (`.bz2`), lz4 (`.lz4`, нужен пакет), xz (`.xz`). По расширению joblib сам выбирает алгоритм.

### Tuple-form для управления

```python
joblib.dump(pipeline, "model.joblib", compress=("gzip", 3))
```

## 4. Что именно сохранять

### Сохраняй весь Pipeline, а не модель отдельно

Если ты построил:
```python
pipeline = Pipeline([
    ("preprocess", ColumnTransformer(...)),
    ("model",      XGBRegressor(...)),
])
```

то `joblib.dump(pipeline, ...)` сериализует **всё**: scaler с выученными mean/std, encoder с выученными категориями, модель с выученными деревьями.

**Никогда** не сохраняй только модель:
```python
# ПЛОХО
joblib.dump(pipeline.named_steps["model"], "model.joblib")
```
Тогда в проде придётся отдельно загружать препроцессинг, и любое расхождение в коде препроцессинга → bug на проде.

### Что сохранять помимо самой модели

В production-практике вместе с моделью обычно сохраняется **manifest** — метаданные:

```python
import joblib
import sklearn
import xgboost

artifact = {
    "model":             pipeline,
    "sklearn_version":   sklearn.__version__,
    "xgboost_version":   xgboost.__version__,
    "python_version":    sys.version,
    "feature_names":     list(X_train.columns),
    "target_name":       "total_rentals",
    "trained_at":        datetime.utcnow().isoformat(),
    "training_metrics":  {"mae": mae_train, "rmse": rmse_train},
    "test_metrics":      {"mae": mae_test,  "rmse": rmse_test},
    "training_window":   {"start": "2011-01-01", "end": "2012-07-01"},
}

joblib.dump(artifact, "model.joblib", compress=3)
```

Это даёт:
- Возможность проверить совместимость версий при загрузке.
- Историю, какие именно данные использовались.
- Метрики «на момент тренировки» как ground truth.

Для Dagster часть этого уйдёт в metadata asset-а (см. §7).

## 5. Версионная совместимость — главная боль

**Загрузить модель, обученную в sklearn 1.3, через sklearn 1.5 — может не работать.** Структуры классов меняются между минорными версиями.

### Симптомы

- `AttributeError: ... has no attribute ...` при `joblib.load`.
- Тихо «работает», но `predict` даёт ерунду (изменилось внутреннее представление).
- Warning от sklearn: `InconsistentVersionWarning: Trying to unpickle estimator ... from version X.Y with version A.B`.

### Что делать

1. **Сохранять версии в manifest** (§4). При загрузке сравнивать с текущими — warning или refuse.
2. **Pinning версий** в `pyproject.toml`. Если модель в проде была обучена с sklearn==1.5.2, не апгрейдиться без переобучения.
3. **Переобучать** при любом upgrade зависимостей — это самый надёжный путь.

```python
import sklearn
import warnings

def load_with_check(path):
    artifact = joblib.load(path)
    if artifact["sklearn_version"] != sklearn.__version__:
        warnings.warn(
            f"sklearn version mismatch: trained with {artifact['sklearn_version']}, "
            f"loading with {sklearn.__version__}"
        )
    return artifact
```

## 6. Безопасность — НИКОГДА не загружай untrusted pickle

`pickle.load` (и joblib.load по умолчанию) **выполняет произвольный Python-код** при загрузке. Это by-design: pickle восстанавливает состояние объектов через вызов их методов.

**Атака:** злонамеренно собранный pickle-файл при загрузке запускает `os.system("rm -rf /")` или хуже.

**Правило:** никогда не загружай joblib/pickle файлы из непроверенных источников. Если в проде модели приходят из артефакт-стора — стор должен быть **trusted** (доступ через auth, integrity check через checksum).

Альтернатива для untrusted сред — **ONNX** или **PMML** (см. §8). Они описывают модель декларативно и не запускают код при загрузке.

## 7. Интеграция в Dagster — IO manager для joblib

В bike-rental пайплайне уже есть `csv_io.py` (см. `bike-rental/src/bike_rental/defs/io_managers/csv_io.py`). По аналогии нужен `joblib_io.py`.

### Базовая идея IO manager в Dagster

```python
import joblib
import dagster as dg

class JoblibIOManager(dg.ConfigurableIOManager):
    base_dir: str

    def handle_output(self, context: dg.OutputContext, obj):
        path = self._path(context)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path, compress=3)
        context.add_output_metadata({"path": str(path), "size_mb": path.stat().st_size / 1e6})

    def load_input(self, context: dg.InputContext):
        return joblib.load(self._path(context.upstream_output))

    def _path(self, context):
        # Канонический путь по имени asset-а
        from pathlib import Path
        return Path(self.base_dir) / f"{'.'.join(context.asset_key.path)}.joblib"
```

Тогда asset просто возвращает Pipeline, и сериализация — забота IO manager-а:

```python
@dg.asset(io_manager_key="joblib_io", group_name="model")
def trained_model(final_dataset: pd.DataFrame) -> Pipeline:
    X = final_dataset.drop(columns="total_rentals")
    y = final_dataset["total_rentals"]
    # ... train/test split, fit pipeline ...
    return pipeline
```

Это идиоматичный Dagster-подход: asset знает только «что вернуть», IO manager знает «куда и как сохранить».

### Альтернатива — сохранение вручную из asset-а

Если делать IO manager лень или нужен полный контроль:

```python
@dg.asset(group_name="model", kinds={"sklearn"})
def trained_model(final_dataset: pd.DataFrame) -> dg.MaterializeResult:
    # ... train ...
    output_path = Path("data/models/bike_rental_model.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path, compress=3)
    return dg.MaterializeResult(
        metadata={
            "path":    dg.MetadataValue.path(str(output_path)),
            "size_mb": dg.MetadataValue.float(output_path.stat().st_size / 1e6),
            "mae":     dg.MetadataValue.float(mae),
            "rmse":    dg.MetadataValue.float(rmse),
        }
    )
```

Менее идиоматично, но проще для первого захода. **Для week-3 — этого хватит.** Кастомный IO manager — рефактор на потом, если будет несколько моделей или нужна централизация.

## 8. Альтернативы joblib

### ONNX (Open Neural Network Exchange)

Стандартизированный формат для моделей. Преимущества:
- Кросс-языковой (Python → C++/Java/JS для inference).
- Не запускает код при загрузке — безопасно.
- Оптимизация inference (ONNX Runtime).

Минусы:
- Не все sklearn-модели полностью поддерживаются.
- Сложнее тулинг.

Для XGBoost есть `xgboost.Booster.save_model` (нативный JSON-формат) — тоже безопаснее и совместимее, чем pickle.

### PMML (Predictive Model Markup Language)

XML-формат для классических ML-моделей. Старый стандарт, поддерживается во многих enterprise-средах. Для week-3 — overkill.

### Native model save (XGBoost / LightGBM)

```python
model.save_model("model.json")     # XGBoost-native, безопасно
loaded = XGBRegressor()
loaded.load_model("model.json")
```

**Но:** это сохраняет **только** саму модель, без препроцессинга. Если у тебя Pipeline — придётся комбинировать: native save для модели + joblib для препроцессинга. Или joblib для всего.

**Для week-3:** joblib на весь Pipeline. Это стандарт и проще всего.

## 9. Проверка после сохранения — sanity check

Никогда не доверяй «успешно сохранено». Всегда проверяй, что обратно загруженный объект работает:

```python
joblib.dump(pipeline, "model.joblib", compress=3)

# Загружаем и проверяем
loaded = joblib.load("model.joblib")

y_pred_original = pipeline.predict(X_test[:100])
y_pred_loaded   = loaded.predict(X_test[:100])

import numpy as np
assert np.allclose(y_pred_original, y_pred_loaded), "Round-trip failed!"
print("Sanity check passed")
```

Это ловит случаи, когда что-то в pipeline не сериализуется корректно (например, lambda-функции, локальные классы).

## 10. Кодовый шаблон для bike-rental

```python
from pathlib import Path
import joblib
import sklearn
import xgboost
from datetime import datetime
import sys

def save_model(pipeline, X_train, X_test, y_test, y_pred, output_path):
    """Сохранить обученный pipeline вместе с метаданными."""
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": pipeline,
        "metadata": {
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__,
            "python_version":  sys.version.split()[0],
            "feature_names":   list(X_train.columns),
            "trained_at":      datetime.utcnow().isoformat(),
            "test_metrics": {
                "mae":  float(mean_absolute_error(y_test, y_pred)),
                "rmse": float(root_mean_squared_error(y_test, y_pred)),
                "r2":   float(r2_score(y_test, y_pred)),
            },
        },
    }

    joblib.dump(artifact, output_path, compress=3)

    # Sanity check
    loaded = joblib.load(output_path)
    assert (loaded["model"].predict(X_test[:10]) == y_pred[:10]).all(), "Round-trip failed"
    return output_path


def load_model(path):
    """Загрузить pipeline с проверкой версий."""
    artifact = joblib.load(path)
    md = artifact["metadata"]

    if md["sklearn_version"] != sklearn.__version__:
        import warnings
        warnings.warn(
            f"sklearn version mismatch: trained={md['sklearn_version']}, current={sklearn.__version__}"
        )

    return artifact["model"], md
```

## 11. Резюме

- **joblib** для всего sklearn/numpy-кода. `compress=3`, расширение `.joblib`.
- Сохраняй **весь Pipeline**, не только модель.
- В manifest клади **версии библиотек, feature_names, метрики, дату обучения**.
- При загрузке — **проверяй версии**, предупреждай о mismatch.
- Никогда не грузи **untrusted pickle-файлы** — это исполнение кода.
- Всегда делай **round-trip sanity check** после сохранения.
- Для Dagster в week-3 — сохранение в asset через MaterializeResult, кастомный IO manager — рефактор на потом.
