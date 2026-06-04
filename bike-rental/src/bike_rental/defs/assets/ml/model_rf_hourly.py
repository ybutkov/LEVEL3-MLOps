"""RandomForest model on city-wide hourly demand.

The feature plan lives in the shared recipe (`tree_dataset_hourly`). Trees use raw
integer features, so the tree recipe has no stateful steps and the preprocessor
is a passthrough — but it's wired exactly like the linear model, so adding e.g. a
`one_hot` step (per-location) would just work without touching this asset.
"""

import dagster as dg
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from bike_rental.defs.assets.ml.dataset_tree_hourly import TreeDatasetConfig
from bike_rental.defs.assets.ml.recipe.apply import assert_recipe_columns, build_preprocessor
from bike_rental.defs.assets.ml.training.guards import assert_no_target_leak
from bike_rental.defs.assets.ml.training.split import (
    describe_time_split,
    train_validate_test_time_split,
)


class TreeModelConfig(TreeDatasetConfig):
    """Tree recipe (target + steps) plus RandomForest hyperparameters."""

    n_estimators: int = 200
    max_depth: int | None = None
    random_state: int = 42


@dg.asset(group_name="models", io_manager_key="model_io", kinds={"sklearn"})
def rf_hourly(
    tree_dataset_hourly: pd.DataFrame, config: TreeModelConfig
) -> dg.MaterializeResult:
    """Train a RandomForest on ``tree_dataset_hourly`` (raw integer features).

    Trees need no stateless encoding, so the recipe preprocessor is a
    passthrough — but it is wired like the linear model, so adding a step later
    would just work. Reports validation metrics only; the test split is held out.

    Parameters
    ----------
    tree_dataset_hourly : pandas.DataFrame
        Assembled tree input table (raw features + target + time key).
    config : TreeModelConfig
        Tree recipe plus RandomForest hyperparameters.

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

    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state,
        n_jobs=-1,
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
