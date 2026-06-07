import dagster as dg
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from bike_rental.defs.assets.ml.recipe.recipe_config import RecipeConfig
from bike_rental.defs.assets.ml.training.train import train_and_evaluate


class RandomForestConfig(dg.Config):
    """RandomForest hyperparameters (editable per-run in the Launchpad)."""

    n_estimators: int = 200
    max_depth: int | None = None
    random_state: int = 42


@dg.asset(group_name="models", io_manager_key="model_io", kinds={"sklearn"})
def rf_hourly(
    tree_dataset_hourly_train: pd.DataFrame,
    tree_dataset_hourly_val: pd.DataFrame,
    recipe_config: RecipeConfig,
    config: RandomForestConfig,
) -> dg.MaterializeResult:
    
    estimator = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state,
        n_jobs=-1,   # infra, not an experiment knob -> stays out of config
    )
    result = train_and_evaluate(tree_dataset_hourly_train, tree_dataset_hourly_val,
                                recipe_config, "tree", estimator)
    return dg.MaterializeResult(
        value=result.pipeline,
        metadata={
            **result.metadata,
            "model_type": dg.MetadataValue.text("rf"),
            "params": dg.MetadataValue.json(config.model_dump()),
        },
    )
