# Bike Rental — demand MLOps project (weeks 2–4)

An end-to-end project on a city bike-sharing dataset. It grows across weeks on the
**same data** and the **same Dagster pipeline**:

- **Week 2 (done):** data preprocessing pipeline → one analysis-ready dataset.
- **Week 3:** EDA + regression model to predict hourly demand, training added to the pipeline.
- **Week 4:** serving (tbd).

## Layout

```
bike-rental/
├── config/                  # per-stand config (see "Configuration")
│   ├── base.yaml            # env-invariant data layout
│   ├── local.yaml           # local roots
│   └── prod.yaml            # prod roots
├── data/
│   ├── raw/                 # source CSVs (in git)
│   ├── processed/           # pipeline output (final_dataset.csv) — gitignored
│   └── quarantine/          # rows that failed row-level validation — gitignored
├── notebooks/
│   └── eda_and_preprocessing.ipynb   # week-2 exploration + dataset build
├── handout/                 # assignment briefs (week-2, week-3)
└── src/bike_rental/
    ├── config.py            # AppConfig — typed config loaded from config/*.yaml
    ├── definitions.py       # Dagster Definitions (assets, checks, resources, IO managers)
    └── defs/
        ├── assets/data/     # raw → intermediate (split) → primary layers
        ├── assets/ml/       # feature layer (final_dataset)
        ├── asset_checks/    # pandera data contracts on raw sources
        ├── resources/       # source-data resource
        ├── io_managers/     # CSV IO manager
        ├── schemas.py       # pandera schemas + validate_and_split
        └── io_utils.py      # shared CSV-read helper
```

## Pipeline (week 2)

```
raw CSVs ──(pandera contracts, blocking)──► intermediate split ──► primary ──► feature
   per source                               typed + quarantine     hourly      final_dataset.csv
```

- **raw** — thin loaders, one per source CSV.
- **contracts** — pandera schemas validate each raw source; a failure blocks downstream.
- **intermediate** — parse timestamps, split clean rows from unparseable ones (→ `quarantine/`).
- **primary** — aggregate rentals to hourly counts per location; clean/encode weather.
- **feature** — join weather + holiday flag + time features → `data/processed/final_dataset.csv`
  (one row per hour × location).

## Getting started

```bash
uv sync                  # create the env and install deps
uv run dagster dev       # open http://localhost:3000, then "Materialize all"
```

## Configuration (stands)

Paths come from `config/`, chosen by the `DAGSTER_DEPLOYMENT` env var (default `local`):

- `base.yaml` — the data layout, shared by every stand.
- `{stand}.yaml` — only the roots (`data_root`, `dagster_dir`). Relative paths are
  anchored to the project root, so they don't depend on the working directory.

```bash
DAGSTER_DEPLOYMENT=prod uv run dagster dev   # use config/prod.yaml
```

## Quality

```bash
uv run dg check defs     # validate definitions load (no materialization)
uv run ruff check .      # lint
uv run ruff format .     # format
```
