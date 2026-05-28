# ML Engineering Track

A hands-on course building end-to-end machine learning systems and MLOps workflows — from from-scratch models to production pipelines with Dagster.

Each project is **self-contained** with its own environment. To work on one:

```bash
cd <project> && uv sync
```

## Projects

| Project | Weeks | Topic |
|---------|-------|-------|
| [titanic/](titanic/) | 1 | Classification — Titanic EDA, scikit-learn and from-scratch logistic regression (intro) |
| [bike-rental/](bike-rental/) | 2–4 | Bike demand MLOps — one evolving project: data pipeline (w2) → regression training (w3) → serving (w4), orchestrated with Dagster |

## Stack

- **Python ≥ 3.10**, **uv** for environments
- **NumPy / pandas / scikit-learn / matplotlib / seaborn** — data and ML
- **Dagster** — pipeline orchestration
- **Ruff** — lint and format (defaults shared via the root `ruff.toml`)
