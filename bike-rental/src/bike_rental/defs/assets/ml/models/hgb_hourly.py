import dagster as dg
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from bike_rental.defs.assets.ml.recipes.recipe_config import RecipeConfig
from bike_rental.defs.assets.ml.models.training import train_and_evaluate


class HGBModelConfig(dg.Config):
    """HistGradientBoosting hyperparameters (editable per-run in the Launchpad)."""

    learning_rate: float = 0.1
    max_iter: int = 300
    max_depth: int | None = None
    max_leaf_nodes: int = 31
    l2_regularization: float = 0.0
    random_state: int = 42


@dg.asset(group_name="models", io_manager_key="model_io", kinds={"sklearn"})
def hgb_hourly(
    tree_dataset_hourly_train: pd.DataFrame,
    tree_dataset_hourly_val: pd.DataFrame,
    recipe_config: RecipeConfig,
    config: HGBModelConfig,
) -> dg.MaterializeResult:

    estimator = HistGradientBoostingRegressor(**config.model_dump())
    trainingResult = train_and_evaluate(tree_dataset_hourly_train, tree_dataset_hourly_val, 
                                recipe_config, "tree", estimator)
    return dg.MaterializeResult(
        value=trainingResult.pipeline,
        metadata={
            **trainingResult.metadata,
            "model_type": dg.MetadataValue.text("hgb"),
            "params": dg.MetadataValue.json(config.model_dump()),
        },
    )
