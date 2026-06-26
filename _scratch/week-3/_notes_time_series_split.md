# Train/test split на временных рядах (time-series split)

> Конспект для week-3. Почему random split — мина под отчётность, и как делать split правильно.

## 1. Почему random split опасен

Стандартный `train_test_split(..., shuffle=True)` тасует строки случайно. Для **i.i.d.** данных (independent and identically distributed) это правильно. Для временных рядов — **нет**.

### Источники лика (data leakage) при random split

**(a) Автокорреляция (autocorrelation):**
В соседних часах таргет похож. Сейчас bike-rental на 17:00 пятницы похож на 16:00 и 18:00 той же пятницы. Если 17:00 в train, а 16:00 и 18:00 в test — модель «знает соседей» через train, и метрика на test завышена.

**(b) Календарные фичи (calendar features):**
Если в train попадает половина 4 июля 2012 (Independence Day), а другая половина — в test, то модель видит holiday-эффект для этой конкретной даты при обучении и потом «отчитывается» о точном прогнозе на test. Это не предсказание, это запоминание.

**(c) Долгосрочные тренды (long-term trends):**
В bike-rental сервис вырос на +65% между 2011 и 2012 (см. `_exploration_notes.md`). Random split даст в train примерно равное число строк из 2011 и 2012 → модель «видела будущее» и легко выучивает уровень 2012. В проде же, когда придёт 2013, модель никогда такого не видела.

**Простое правило:** если у строк есть **временной порядок** и **временные зависимости** (autocorrelation, trends, seasonality), random split лик.

## 2. Что мы вообще измеряем

Прежде чем выбирать split, надо понять, на какой вопрос ты отвечаешь:

| Вопрос | Корректный split |
|---|---|
| «Сможет ли модель предсказать **будущие** часы?» | **Temporal holdout** — train на ранних датах, test на поздних |
| «Сможет ли модель предсказать **новые локации**?» | **Group-based holdout** — train на одних `location_id`, test на других |
| «Стабильно ли работает в разных периодах?» | **Walk-forward / TimeSeriesSplit** — несколько временных фолдов |
| «Какой средний MAE на любой час из этого набора?» | i.i.d. CV — но только если автокорреляции нет (редко) |

Для нашего bike-rental основной вопрос — первый. Возможно — третий.

## 3. Temporal holdout — простой временной split

Самая базовая корректная схема:

```python
TRAIN_END = "2012-07-01"

train = dataset[dataset["datetime_hourly"] <  TRAIN_END]
test  = dataset[dataset["datetime_hourly"] >= TRAIN_END]

X_train, y_train = train.drop(columns=TARGET), train[TARGET]
X_test,  y_test  = test.drop(columns=TARGET),  test[TARGET]
```

**Плюсы:** просто, имитирует реальную задачу («сегодня обучили, завтра предсказываем»).

**Минусы:** одна оценка → высокая дисперсия. На «хорошем» test-окне модель выглядит хорошо, на «плохом» — плохо. Не знаешь, насколько эта одна точка репрезентативна.

### Варианты для bike-rental

| Граница | Train | Test | Что показывает |
|---|---|---|---|
| 2012-01-01 | 2011 | 2012 | Жёсткий тест: модель должна экстраполировать через рост +65% |
| 2012-07-01 | 2011 + H1 2012 | H2 2012 | Реалистичнее: год+ исторических данных, прогноз на полгода |
| 2012-10-01 | 2011 + 9 мес 2012 | Q4 2012 | Близко к проду: много данных, короткий прогноз |

Жёсткий вариант показывает реальный «вызов» модели. Реалистичный — то, что будет в продакшене. Имеет смысл попробовать оба и сравнить.

## 4. Walk-forward validation / Expanding window

Идея: вместо одного train/test разреза — **несколько** последовательных, каждый сдвинут вперёд по времени.

**Expanding window (расширяющееся окно):**
```
fold 1: train=[t1..t100],   test=[t101..t150]
fold 2: train=[t1..t150],   test=[t151..t200]
fold 3: train=[t1..t200],   test=[t201..t250]
```
Train растёт с каждым фолдом, test «скользит» вперёд.

**Sliding window (скользящее окно):**
```
fold 1: train=[t1..t100],   test=[t101..t150]
fold 2: train=[t51..t150],  test=[t151..t200]
fold 3: train=[t101..t200], test=[t201..t250]
```
Train фиксированного размера, окно целиком сдвигается вперёд.

**Когда что:**
- Expanding window — если **больше данных ≈ лучше модель** и нет concept drift.
- Sliding window — если есть **concept drift** (поведение со временем меняется), и старые данные мешают.

Для bike-rental: рост +65% — это форма distribution shift. Можно проверить, помогает ли sliding window игнорировать «слишком старый» 2011 при предсказании на 2012.

## 5. sklearn `TimeSeriesSplit`

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)  # по умолчанию expanding window

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]
    # ... обучить, замерить
```

**Важно:** `TimeSeriesSplit` предполагает, что строки **уже отсортированы по времени**. Иначе результат бессмыслен. Сортируй перед `.split()`.

**Полезные параметры:**
- `n_splits=5` — число фолдов
- `test_size=None` — фиксированный размер test-фолда (если задан) или равномерное деление (если None)
- `max_train_size=None` — для sliding window: ограничивает train сверху
- `gap=0` — отступ между концом train и началом test. Полезно когда:
  - У таргета есть сильная автокорреляция → gap снижает оптимизм метрики.
  - При создании lag-фич: lag=24 часа → нужен gap=24, иначе train «видит» свой test через лаг.

### Как использовать с `cross_val_score` / `GridSearchCV`

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    pipeline, X, y,
    cv=TimeSeriesSplit(n_splits=5, gap=24),
    scoring="neg_root_mean_squared_error",
)
print(f"RMSE по фолдам: {-scores}")
```

## 6. Особые случаи и подводные камни

### (a) Bike-rental — несколько строк на один час (`(hour, location_id)`)

В нашем `final_dataset` каждый час представлен 21 строкой (по локациям). Это значит:
- В одном «временном моменте» 21 строка.
- Если split случайный, эти 21 строка могут разъехаться между train и test, и модель будет «знать», как ведут себя другие kiosk-и в этот же час.

**Решение:** split по **уникальному часу**, а не по строке.

```python
unique_hours = pd.Series(dataset["datetime_hourly"].unique()).sort_values()
split_hour = unique_hours.iloc[int(len(unique_hours) * 0.7)]

train = dataset[dataset["datetime_hourly"] <  split_hour]
test  = dataset[dataset["datetime_hourly"] >= split_hour]
```

Так все строки одного часа гарантированно по одну сторону.

### (b) Lag-фичи (запаздывающие признаки)

Если планируется фича `total_rentals_lag_24h` (аренда за тот же час сутки назад), при вычислении этой фичи нельзя «смотреть в будущее». То есть:
1. **Считать лаги сначала** (на полном датасете).
2. **Потом split.** Внутри split lag-фичи уже корректны.

Альтернативно: считать лаги только относительно train для test-строк. Сложнее, но честнее, если важно избежать любой утечки.

### (c) Group-based + temporal вместе

Если хочешь проверить, насколько модель переносится на **новые локации в будущем** — это иerarchical: сначала group split (часть локаций отложить), потом temporal split (вторая часть по времени). В sklearn явного класса для этого нет — собирается вручную.

### (d) Целевая статистика (target encoding) и временной порядок

Если в препроцессинге будет target encoding для какой-то категории (например `location_id`), его надо считать **только на train**, иначе таргет утекает в фичи. С `TimeSeriesSplit` + `Pipeline` это решается само (fit вызывается только на train-фолде). Без pipeline — легко ошибиться.

## 7. Резюме для bike-rental week-3

**Минимум:**
- Сначала **temporal holdout** (train=`< 2012-07-01`, test=`>= 2012-07-01`).
- Замерить MAE / RMSE / R² на test.

**Если будет время на CV:**
- `TimeSeriesSplit(n_splits=5)` для подбора гиперпараметров и оценки стабильности.
- Если планируем lag-фичи — задать `gap` равным максимальному лагу.

**Чего не делать:**
- `train_test_split(shuffle=True)` — даст обманчиво высокий score.
- Random KFold — то же.
- Считать препроцессинг (scaler, target encoding) на полном датасете до split — даже минимальное утекание искажает результат.
