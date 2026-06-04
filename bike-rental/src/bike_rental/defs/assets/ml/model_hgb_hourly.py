"""Gradient-boosted trees on city-wide hourly demand.

The intended model here is gradient boosting. XGBoost is the usual pick, but its
macOS wheel needs the OpenMP runtime (`libomp.dylib`), which is only available via
`brew install libomp` — outside the uv-managed environment. sklearn's
`HistGradientBoostingRegressor` is the uv-native equivalent: a fast, histogram-based
GBM with no system-library dependency, so it stays within `uv` like every other dep.

Like `rf_hourly`, it reads the shared tree recipe (`tree_dataset_hourly`): trees use
raw integer features, so the preprocessor is a passthrough — but it's wired exactly
like the other models, so adding a recipe step later would just work.
"""

import dagster as dg
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from bike_rental.defs.assets.ml.dataset_tree_hourly import TreeDatasetConfig
from bike_rental.defs.assets.ml.recipe.apply import assert_recipe_columns, build_preprocessor
from bike_rental.defs.assets.ml.training.guards import assert_no_target_leak
from bike_rental.defs.assets.ml.training.split import (
    describe_time_split,
    train_validate_test_time_split,
)


class HGBModelConfig(TreeDatasetConfig):
    """Tree recipe (target + steps) plus HistGradientBoosting hyperparameters."""

    learning_rate: float = 0.1
    max_iter: int = 300
    max_depth: int | None = None
    max_leaf_nodes: int = 31
    l2_regularization: float = 0.0
    random_state: int = 42


@dg.asset(group_name="models", io_manager_key="model_io", kinds={"sklearn"})
def hgb_hourly(
    tree_dataset_hourly: pd.DataFrame, config: HGBModelConfig
) -> dg.MaterializeResult:
    """Train a HistGradientBoosting GBM on ``tree_dataset_hourly`` (raw integer features).

    Trees need no stateless encoding, so the recipe preprocessor is a
    passthrough. Reports validation metrics only; the test split is held out.

    Parameters
    ----------
    tree_dataset_hourly : pandas.DataFrame
        Assembled tree input table (raw features + target + time key).
    config : HGBModelConfig
        Tree recipe plus HistGradientBoosting hyperparameters.

    Returns
    -------
    dagster.MaterializeResult
        The fitted sklearn ``Pipeline``, with validation MAE / RMSE / R² /
        RMSE-over-MAE and chronological-split metadata.
    """
    target = config.target
    features = [c for c in tree_dataset_hourly.columns if c not in (target, "datetime_hourly")]
    assert_no_target_leak(features, target)
    assert_recipe_columns(config, tree_dataset_hourly.columns)

    # _X_test/_y_test held out for a future one-shot test-set evaluation;
    # training only reports validation metrics, used for model selection.
    X_train, y_train, X_val, y_val, _X_test, _y_test = train_validate_test_time_split(
        tree_dataset_hourly, "datetime_hourly", features, target
    )

    model = HistGradientBoostingRegressor(
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_depth=config.max_depth,
        max_leaf_nodes=config.max_leaf_nodes,
        l2_regularization=config.l2_regularization,
        random_state=config.random_state,
    )
    pipe = Pipeline([("pre", build_preprocessor(config)), ("model", model)])
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
            **describe_time_split(tree_dataset_hourly, "datetime_hourly"),
        },
    )
