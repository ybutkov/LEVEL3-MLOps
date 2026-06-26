# Варианты улучшений: config-driven сборка датасетов

Контекст: собрали свой «registry + typed-config + factory» для сборки датасетов
(`_dataset_builder.build_dataset` + `dataset_config.DatasetStep` + `recipe_loader`
+ `dataset_recipes.yaml`). Ниже — продуктовые решения, которыми это закрывают в
проде, и локальные доработки нашего кода. Это бэклог идей, не план на сейчас.

## Продуктовые решения (по слоям)

| Слой | Что у нас руками | Продукт |
|---|---|---|
| Сами трансформации фич | `add_cyclic_features`, `_features.py` | **Feature-engine**, scikit-lego, Featuretools, tsfresh, skrub |
| Сборка пайплайна из конфига | `TRANSFORMS` + `build_dataset` + `DatasetConfig` | **Hydra** (`instantiate`/`_target_`), Gin-config |
| Декларативный каталог + пайплайн | YAML-рецепты + лоадер | **Kedro** (Data Catalog + params) |
| Переиспользование фич + train/serve parity | общий `_features.py` | **Feast**, Tecton, Hopsworks, cloud feature stores |
| FE на SQL/складе | — | **dbt**, SQLMesh (нативная интеграция в Dagster) |

### Самое релевантное стеку (Dagster + sklearn + pydantic)

1. **Feature-engine** — sklearn-совместимые FE-трансформеры (`CyclicalFeatures`,
   `DatetimeFeatures`, энкодеры, выбросы). Наш `add_cyclic_features` ≈ их
   `CyclicalFeatures`. Можно подменить реализации внутри `TRANSFORMS`, оставив
   наш реестр/конфиг. Низкая цена, не пишем FE с нуля. **← первый кандидат.**
2. **Hydra** — продуктовая версия нашего registry+factory (`_target_` в конфиге,
   фреймворк инстанцирует). Снимает ручной `kind`-диспатч. Минус: с config-слоем
   Dagster не дружит — жить должен *внутри* шага, не как config ассета.
3. **Kedro** — целый аналог нашего подхода (каталог + конфиг-пайплайны). Но это
   конкурент-оркестратор, а мы на Dagster → тащить оба незачем. Только как
   референс идей.
4. **Feast** — когда фичи надо шарить между моделями/командами + гарантировать
   train/serve consistency. Тяжёлая артиллерия (online/offline store). Не сейчас.
5. **dbt** — если фичи переедут в склад (DuckDB/BigQuery/Snowflake): декларативный
   SQL + lineage + тесты, Dagster грузит dbt-модели как ассеты. Не для pandas-стадии.

### Порядок усиления (рекомендация)
1. Feature-engine в `TRANSFORMS` (дёшево, сразу польза).
2. Hydra — если конфиг разрастётся и `kind`-диспатч станет тесным.
3. Feast/dbt — только при реальном сервинге или шеринге фич (week-N+).

Вывод: наш паттерn (registry + typed-config + factory + YAML) — легитимный
прод-подход; продукты дают готовые трансформеры/UI/lineage/serving ценой
зависимости и lock-in.

## Локальные доработки нашего кода (бэклог)

- **Убрать дублирование** `_FALLBACK_RECIPES` ↔ `dataset_recipes.yaml` — сейчас
  синхронятся руками. Варианты: генерировать одно из другого, или дропнуть
  fallback и падать при отсутствии файла.
- **Снять двойную привязку фич для linear** — фича сидит и в рецепте датасета, и в
  `LinearTrainConfig.scale`. Сделать в `linear_hourly` `scale ∩ присутствующие
  колонки` → удаление фичи = одно место (как у деревьев).
- **`model_evaluation` asset** — финальная one-shot оценка на test (сейчас заглушка
  `_eval.evaluate_on_test`), когда зафиксируется победитель; ляжет на MLflow (week-4).
- **Подчистить `LinearTrainConfig`** — модель теперь юзает только `scale`+`target`;
  `cyclic/passthrough/one_hot` переехали в рецепт датасета.
- **Партиции/имена для нескольких датасетов** — сейчас `csv_io` перезаписывает один
  CSV; для сравнения рецептов бок-о-бок нужны партиции или разные имена ассетов.
- **Логи репро** — `steps` уже в metadata; при MLflow логировать рецепт+хеш как params.
