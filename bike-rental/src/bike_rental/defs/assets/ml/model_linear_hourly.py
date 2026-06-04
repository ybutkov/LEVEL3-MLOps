"""Linear regression model on city-wide hourly demand.

Stateless feature engineering is done upstream in `linear_dataset_hourly`. This asset
reads the SAME recipe and builds only its stateful steps (scaling) into a
ColumnTransformer fit on train — so those steps travel with the saved model and
no test statistics leak. Single source of truth for the feature plan: the recipe.
"""

import dagster as dg
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from bike_rental.defs.assets.ml.dataset_linear_hourly import LinearDatasetConfig
from bike_rental.defs.assets.ml.recipe.apply import assert_recipe_columns, build_preprocessor
from bike_rental.defs.assets.ml.training.guards import assert_no_target_leak
from bike_rental.defs.assets.ml.training.split import (
    describe_time_split,
    train_validate_test_time_split,
)


@dg.asset(group_name="models", io_manager_key="model_io", kinds={"sklearn"})
def linear_hourly(
    linear_dataset_hourly: pd.DataFrame, config: LinearDatasetConfig
) -> dg.MaterializeResult:
    """Train linear regression on `linear_dataset_hourly`, scaling per the recipe.

    Features are every column except the target and the time key. The recipe's
    stateful steps (scale/one_hot) become a ColumnTransformer fit on train; all
    other columns (cyclic sin/cos, binary flags) pass through.
    """
    target = config.target
    features = [c for c in linear_dataset_hourly.columns if c not in (target, "datetime_hourly")]
    assert_no_target_leak(features, target)
    assert_recipe_columns(config, linear_dataset_hourly.columns)

    # _X_test/_y_test held out for a future one-shot test-set evaluation;
    # training only reports validation metrics, used for model selection.
    X_train, y_train, X_val, y_val, _X_test, _y_test = train_validate_test_time_split(
        linear_dataset_hourly, "datetime_hourly", features, target
    )

    pipe = Pipeline([("pre", build_preprocessor(config)), ("model", LinearRegression())])
    pipe.fit(X_train, y_train)
    val_pred = np.clip(pipe.predict(X_val), 0, None)

    rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    mae = float(mean_absolute_error(y_val, val_pred))
    r2 = float(r2_score(y_val, val_pred))
    # RMSE/MAE ratio: error-spread indicator (>=1; ~1.25 ~ normal errors,
    # higher => heavy tails / a few large misses dominating).
    rmse_mae_ratio = float(rmse / mae) if mae else float("nan")

    return dg.MaterializeResult(
        value=pipe,
        metadata={
            "val_mae": dg.MetadataValue.float(mae),
            "val_rmse": dg.MetadataValue.float(rmse),
            "val_r2": dg.MetadataValue.float(r2),
            "val_rmse/mae": dg.MetadataValue.float(rmse_mae_ratio),
            **describe_time_split(linear_dataset_hourly, "datetime_hourly"),
        },
    )
