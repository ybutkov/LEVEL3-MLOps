import dagster as dg
import pandas as pd

from bike_rental.defs.assets.ml.recipes.recipe_config import RecipeConfig
from bike_rental.defs.assets.ml.models.training import train_and_evaluate
from bike_rental.defs.assets.ml.models.catalog import build_estimator, recipe_name_for
from bike_rental.defs.resources.experiment import ExperimentConfig


class LinearModelConfig(dg.Config):
    """LinearRegression hyperparameters (editable per-run in the Launchpad)."""

    fit_intercept: bool = True
    positive: bool = False

@dg.asset(group_name="models", io_manager_key="model_io", kinds={"sklearn"})
def linear_hourly(
    linear_dataset_hourly_train: pd.DataFrame,
    linear_dataset_hourly_val: pd.DataFrame,
    recipe_config: RecipeConfig,
    experiment_config: ExperimentConfig,
    config: LinearModelConfig,
) -> dg.MaterializeResult:
    model_type = "linear"
    estimator = build_estimator(model_type, config.model_dump())
    trainingResult = train_and_evaluate(
        linear_dataset_hourly_train, linear_dataset_hourly_val,
        recipe_config, recipe_name_for(model_type), estimator,
        features=experiment_config.features,
    )
    return dg.MaterializeResult(
        value=trainingResult.pipeline,
        metadata={
            **trainingResult.metadata,
            "model_type": dg.MetadataValue.text(model_type),
            "params": dg.MetadataValue.json(config.model_dump()),
            "features": dg.MetadataValue.json(experiment_config.features),
        },
    )
