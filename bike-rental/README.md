# Bike Rental — demand MLOps project (weeks 2–4)

An end-to-end MLOps project on a city bike-sharing dataset that predicts **hourly rental
demand**. It grows across weeks on the same data and the same Dagster pipeline — week 2 builds
the data-preprocessing pipeline, week 3 adds the models, week 4 adds experiment tracking, data
versioning, and an HTTP API.

The model answers a simple operational question: given an hour and the weather, roughly how many
rentals to expect city-wide.

## Pipeline

A full run is a Dagster asset graph that goes through these stages:

1. **Read raw** from LakeFS (versioned source data).
2. **Build one feature table** (`feature_rentals_hourly`) and split it chronologically into
   **train / val / test**.
3. **Snapshot the splits to LakeFS** (`data_commit`) — an immutable handle to the data this run used.
4. **Train 3 models** (linear / rf / hgb) from a small factory; each logs its run to **MLflow** and
   is registered as a new version, tagged with its `data_commit` and `git_commit`.
5. **Promote** the best by validation metric to the `@champion` and `@production` registry aliases.

The **serving API** is a separate process; it meets the pipeline only at the MLflow registry,
loading `@production` by alias. Infra (MLflow + LakeFS + MinIO + Postgres, docker compose) is
described in [`docker/README.md`](docker/README.md).

## A few design choices

- **One feature table, per-model representation in the pipeline.** The stored dataset holds the
  raw feature columns; the recipe per model (cyclic encoding / scaling, or none for trees) lives
  inside the training `Pipeline`, so a single dataset serves every model.
- **The model carries its own data lineage.** Each run logs the training data as an MLflow dataset
  (`mlflow.data` — schema, digest, source), and tags the model with the LakeFS `data_commit` and
  the `git_commit`. Given a registered version, the data and code behind it are queryable.
- **Read and write paths are separate in LakeFS.** `read_ref` is where the pipeline reads raw
  (pinnable to a commit for a reproducible run); `merge_into` is the writable trunk that ingest
  and dataset snapshots merge into.
- **Serving avoids train/serve skew.** A request carries only a timestamp and weather; calendar,
  holiday, and trend features are derived server-side with the same constants and formulas as the
  pipeline. The `@production` model is cached and re-fetched only when the alias points to a new
  version (a cheap metadata check, not a download per request).

## Structure

```
bike-rental/
├── config/                      # per-stand config + recipes (column select / transforms / split)
├── data/raw/                    # source CSVs (seeded into LakeFS); processed/ is a gitignored cache
├── docker/                      # MLflow/LakeFS/MinIO/Postgres stack (docker/README.md)
├── scripts/                     # seed_lakefs · clean_lakefs · check_api (serving smoke test)
└── src/bike_rental/
    ├── config.py                # AppConfig — typed config from config/*.yaml
    ├── definitions.py           # Dagster Definitions (assets, checks, resources, IO managers)
    ├── defs/
    │   ├── assets/data/         # raw → intermediate → primary → feature (hourly_total)
    │   ├── assets/ml/datasets/  # feature table + train/val/test splits + data_commit
    │   ├── assets/ml/models/    # model factory, catalog, champion promotion
    │   ├── assets/ml/recipes/   # recipe schema + dataset/preprocessor builders
    │   └── resources/           # LakeFS source/versioning, MLflow experiment tracker
    └── serving/                 # FastAPI app, champion cache, request/response schemas
```

## Usage

```bash
uv sync                                  # env + deps
# .env: LAKEFS_ACCESS_KEY / LAKEFS_SECRET_KEY (gitignored, auto-loaded)

uv run python scripts/seed_lakefs.py                              # seed raw into LakeFS (once)
uv run dagster asset materialize --select '*' -m bike_rental.definitions   # full run (or: dagster dev)

uv run uvicorn bike_rental.serving.app:app --port 8000           # serve @production model
```

Query the API (Swagger UI at `http://localhost:8000/docs`):

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "timestamp": "2013-07-04T08:00:00", "conditions": "clouds",
  "temperature_c": 25, "perceived_temperature_c": 27, "humidity": 60, "windspeed_kmh": 12
}'
# → {"predicted_rentals": 504, "model_version": "44", "data_commit": "bc267e03...", ...}

uv run python scripts/check_api.py       # smoke test: /health + sample /predict calls
```

## Configuration

Endpoints and paths come from `config/`, chosen by `DAGSTER_DEPLOYMENT` (default `local`).
`base.yaml` holds the shared layout plus the MLflow and LakeFS endpoints; `{stand}.yaml` holds
only the roots.

## Quality

```bash
uv run dagster definitions validate -m bike_rental.definitions   # graph loads (no materialize)
uv run ruff check .                                              # lint
```
