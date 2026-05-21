# ML Engineering Track

A hands-on course building end-to-end machine learning systems and MLOps workflows — from scratch classification models to production pipelines with Dagster, MLflow, and LakeFS.

## Weeks

| Week | Topic | Description |
|------|-------|-------------|
| [Week 1](week-1/) | Classification | EDA, logistic regression with scikit-learn, and logistic regression from scratch using NumPy and gradient descent |
| [Week 2](week-2/) | Data Pipelines | Multi-source data preprocessing pipeline orchestrated with Dagster |

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
make install   # create virtual environment and install dependencies
make test      # run tests
make lint      # run flake8 and mypy
```

## Stack

- **Python 3.11**
- **NumPy / pandas / scikit-learn / matplotlib / seaborn** — data and ML
- **Dagster** — pipeline orchestration
- **MLflow** — experiment tracking
- **LakeFS** — data versioning
- **pytest** — testing
- **uv** — dependency management
