from dataclasses import dataclass

import dagster as dg
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from bike_rental.defs.assets.ml.recipes.schema import DatasetConfig
from bike_rental.defs.assets.ml.recipes.builders import (
    assert_recipe_columns,
    build_preprocessor,
    restrict_to_features,
)
from bike_rental.defs.assets.ml.models.wrappers import NonNegativeRegressor
from bike_rental.defs.assets.ml.guards import assert_no_target_leak

TIME_KEY = "datetime_hourly"


@dataclass
class TrainingResult:
    """Outcome of one training run, ready for Dagster metadata or MLflow."""

    pipeline: Pipeline
    metrics: dict[str, float]
    metadata: dict
    features: list[str]

def regression_metrics(y_true, y_pred) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    rmse_mae_ratio = float(rmse / mae) if mae else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "rmse/mae": rmse_mae_ratio
    }

def train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    dataset_config: DatasetConfig,
    estimator: BaseEstimator,
    features: list[str],
) -> TrainingResult:

    target = dataset_config.target

    # dataset is full feature table; keep only requested features
    # that are actually present (drop rest silently)
    features = [c for c in features if c in train_df.columns and c not in (target, TIME_KEY)]

    dataset_config = restrict_to_features(dataset_config, features)

    assert_no_target_leak(features, target)
    assert_recipe_columns(dataset_config, train_df.columns)

    pipe = Pipeline([("pre", build_preprocessor(dataset_config)), ("model", NonNegativeRegressor(estimator))])
    pipe.fit(train_df[features], train_df[target])
    val_pred = pipe.predict(val_df[features])

    metrics = regression_metrics(val_df[target], val_pred)
    metadata = {k: dg.MetadataValue.float(v) for k, v in metrics.items()}

    return TrainingResult(pipeline=pipe, metrics=metrics, metadata=metadata, features=features)
