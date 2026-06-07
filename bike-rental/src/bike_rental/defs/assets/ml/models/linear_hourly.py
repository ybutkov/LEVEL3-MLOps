import dagster as dg
import pandas as pd
from sklearn.linear_model import LinearRegression

from bike_rental.defs.assets.ml.recipes.recipe_config import RecipeConfig
from bike_rental.defs.assets.ml.models.training import train_and_evaluate


class LinearModelConfig(dg.Config):
    """LinearRegression hyperparameters (editable per-run in the Launchpad)."""

    fit_intercept: bool = True
    positive: bool = False

@dg.asset(group_name="models", io_manager_key="model_io", kinds={"sklearn"})
def linear_hourly(
    linear_dataset_hourly_train: pd.DataFrame,
    linear_dataset_hourly_val: pd.DataFrame,
    recipe_config: RecipeConfig,
    config: LinearModelConfig,
) -> dg.MaterializeResult:
    estimator = LinearRegression(**config.model_dump())
    trainingResult = train_and_evaluate(linear_dataset_hourly_train, linear_dataset_hourly_val, 
                                recipe_config, "linear", estimator)
    return dg.MaterializeResult(
        value=trainingResult.pipeline,
        metadata={
            **trainingResult.metadata,
            "model_type": dg.MetadataValue.text("linear"),
            "params": dg.MetadataValue.json(config.model_dump()),
        },
    )
