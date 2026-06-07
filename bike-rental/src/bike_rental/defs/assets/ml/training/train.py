from dataclasses import dataclass

import dagster as dg
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from bike_rental.defs.assets.ml.recipe.recipe_config import RecipeConfig
from bike_rental.defs.assets.ml.recipe.schema import DatasetConfig
from bike_rental.defs.assets.ml.recipe.apply import assert_recipe_columns, build_preprocessor
from bike_rental.defs.assets.ml.training.guards import assert_no_target_leak

TIME_KEY = "datetime_hourly"


@dataclass
class TrainingResult:
    """Outcome of one training run, ready for Dagster metadata or MLflow."""

    pipeline: Pipeline
    metrics: dict[str, float]
    metadata: dict

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
    recipe_config: RecipeConfig,
    recipe_name: str,
    estimator: BaseEstimator,
) -> TrainingResult:
    
    dataset_config = DatasetConfig.from_recipe(recipe_config, recipe_name)
    target = dataset_config.target
    features = [c for c in train_df.columns if c not in (target, TIME_KEY)]

    assert_no_target_leak(features, target)
    assert_recipe_columns(dataset_config, train_df.columns)

    pipe = Pipeline([("pre", build_preprocessor(dataset_config)), ("model", estimator)])
    pipe.fit(train_df[features], train_df[target])
    val_pred = np.clip(pipe.predict(val_df[features]), 0, None)

    metrics = regression_metrics(val_df[target], val_pred)
    metadata = {k: dg.MetadataValue.float(v) for k, v in metrics.items()}

    return TrainingResult(pipeline=pipe, metrics=metrics, metadata=metadata)
