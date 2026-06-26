# MLOps-стек bike-rental (LakeFS + MLflow + MinIO + Postgres)

Локальный/VPS стек: **LakeFS** (версии данных) + **MLflow** (трекинг/реестр) на общих
**MinIO** (S3) и **Postgres**. Основа — официальный lakeFS *everything-bagel*, докручено под наш сервис.

Один файл — `docker-compose.yml`. **Ручных init-шагов нет**: всё нужное (бакеты, БД, setup
LakeFS, репозиторий) создаётся сервисами при `docker compose up -d`, идемпотентно.

## Состав и порты

| сервис | роль | порт наружу |
|---|---|---|
| `postgres` | backend store (БД `postgres` → LakeFS, `mlflow` → MLflow) | — (внутр.) |
| `minio` | S3-хранилище | 9000 (API) / 9001 (консоль) |
| `minio-setup` | one-shot: создаёт бакеты, выходит | — |
| `db-init` | one-shot: создаёт БД `mlflow`, выходит | — |
| `lakefs` | версии данных, репо `bike-rental` | 8000 |
| `mlflow` | трекинг + реестр, артефакты proxied в MinIO | 5000 |

## Что инициализируется автоматически (и кем)

| сущность | кем | идемпотентно |
|---|---|---|
| бакеты MinIO `bike-rental`, `mlflow` | `minio-setup` (`mc mb -p`) | да |
| БД Postgres `mlflow` | `db-init` (`CREATE DATABASE` если нет) | да |
| админ LakeFS + ключи | `lakefs` entrypoint (`lakefs setup`) | да (`|| true`) |
| репозиторий LakeFS `bike-rental` (`s3://bike-rental`, ветка `main`) | `lakefs` entrypoint (ждёт API через `lakectl repo list` → `lakectl repo create`) | да (повтор → already exists) |
| сервер MLflow (backend `mlflow`, artifacts `s3://mlflow/`) | `mlflow` | — |

## Доступы (УЧЕБНЫЙ сервер — креды дефолтные/публичные!)

| сервис | endpoint | ключ / логин | секрет / пароль |
|---|---|---|---|
| MinIO | `:9000` API, `:9001` UI | `minioadmin` | `minioadmin` |
| LakeFS | `:8000` | `AKIAIOSFODNN7EXAMPLE` | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |
| MLflow | `:5000` | без авторизации | — |
| Postgres | внутр. `postgres:5432` | `postgres` | `postgres` |

> ⚠️ Ключи LakeFS — публичные примеры из bagel; пароли дефолтные. Годится только для
> учебного стенда. Для чего-то ценного: вынести в `.env`, сменить, закрыть порты файрволом.

## Запуск

```bash
docker compose up -d
docker compose ps                 # все сервисы Up; minio-setup/db-init = Exited (0) — это норма
docker compose logs lakefs        # ждём "repo create -> HTTP 201" (или 409, если уже есть)
docker compose logs -f mlflow     # ждём "Listening at: http://0.0.0.0:5000"
```

Обновление после правок `docker-compose.yml` (с локальной машины):
```bash
scp docker-compose.yml <user>@<host>:~/LEVEL/mlops_stack/
ssh <user>@<host> 'cd ~/LEVEL/mlops_stack && docker compose up -d'
```

Полный пересбор с нуля (⚠️ стирает данные — тома `pgdata`/`miniodata`):
```bash
docker compose down -v && docker compose up -d
```

## Проверка здоровья

```bash
curl -s http://localhost:5000/health && echo " mlflow OK"
curl -s http://localhost:8000/_health && echo " lakefs OK"
curl -s -u AKIAIOSFODNN7EXAMPLE:wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY \
  http://localhost:8000/api/v1/repositories      # должен быть bike-rental
```

## Клиент (ноутбук `notebooks/drafts/mlflow_playground.ipynb`)

```python
MLFLOW_TRACKING_URI = 'http://<host>:5000'        # proxied artifacts → S3-креды клиенту не нужны
LAKEFS_HOST   = 'http://<host>:8000'
LAKEFS_KEY    = 'AKIAIOSFODNN7EXAMPLE'
LAKEFS_SECRET = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
REPO, BRANCH  = 'bike-rental', 'main'
# storage_namespace='s3://bike-rental'
```

## Траблшутинг (грабли, которые уже учтены в compose)

| симптом | причина | как закрыто |
|---|---|---|
| MLflow `Invalid Host header - possible DNS rebinding attack` | MLflow 3.5+ security-middleware: Host-проверка | флаг `--allowed-hosts '*'` |
| MLflow Models → `You do not have permission to access this resource` (403) | та же middleware, но CORS: браузерный `Origin` не в allowlist (curl с сервера работал — он без Origin) | флаг `--cors-allowed-origins '*'` |
| MLflow упал, `database "mlflow" does not exist` | initdb-скрипт срабатывает только на пустом томе | сервис `db-init` (идемпотентно при каждом старте) |
| LakeFS на `/setup` / репо не создаётся (`results:[]`) | `wait-for` ждал только порт, не готовность API | entrypoint ждёт через `lakectl repo list` → потом создаёт |
| LakeFS entrypoint: `/bin/sh: curl: not found` | в образе `treeverse/lakefs:1` нет `curl` | используем встроенный `lakectl` (env `LAKECTL_*`) |

> ⚠️ `--allowed-hosts '*'` + `--cors-allowed-origins '*'` = открыто для всех (DNS-rebinding/CORS
> защита отключена). Для учебного стенда ок; на проде указывать конкретные host/origin.

## Стратегия версионирования данных (LakeFS)

Простая ветко-стратегия под размер задачи. Репозиторий `bike-rental`, сырьё под
префиксом `raw/`. Чтение и запись разведены на два конфиг-ключа: **`read_ref`** (откуда
пайплайн читает сырьё) и **`merge_into`** (writable trunk, куда мёржится ingest и снапшоты);
по дефолту оба = `main`.

- **Trunk (`merge_into`) — только проверенные данные.** Пайплайн (`LakeFSSourceResource`)
  читает сырьё из **`read_ref`** — это либо trunk, либо запиннутый commit/тег для
  детерминированного прогона. Каждое изменение данных — отдельный commit.
- **Защита trunk от плохих данных.** Новое сырьё не льётся в trunk напрямую: оно
  попадает на ветку `ingest`, валидируется теми же Pandera-схемами, что и asset-checks
  пайплайна, и **мёржится в `merge_into` только если все файлы прошли**. Бэйд-данные
  остаются на `ingest`, trunk чист. Делает это `scripts/seed_lakefs.py` (запускается
  разово / при смене сырья):
  ```bash
  LAKEFS_ACCESS_KEY=... LAKEFS_SECRET_KEY=... uv run python scripts/seed_lakefs.py
  ```
- **Зачем два ключа.** `read_ref` можно запиннить на commit-id/тег ради воспроизводимого
  прогона, а `merge_into` при этом остаётся writable-веткой — ingest/снапшоты не ломаются
  (мёрж в commit-id невозможен, только в ветку).
- **Какая версия данных обучила модель.** Обучение читает из `read_ref`; commit-id этой
  версии записывается рядом с моделью в MLflow (тег), так что по модели всегда виден источник.
- **Авто-приём новых данных** (сенсор на изменение репо) — это bonus, намеренно не делаем.

Креды LakeFS пайплайну передаются через env `LAKEFS_ACCESS_KEY` / `LAKEFS_SECRET_KEY`
(не в коде/yaml); host/repo/`read_ref`/`merge_into` — в `config/base.yaml`.
