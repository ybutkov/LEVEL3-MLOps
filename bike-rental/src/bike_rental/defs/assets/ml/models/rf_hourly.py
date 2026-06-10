import dagster as dg
import pandas as pd

from bike_rental.defs.assets.ml.recipes.recipe_config import RecipeConfig
from bike_rental.defs.assets.ml.models.training import train_and_evaluate
from bike_rental.defs.assets.ml.models.catalog import build_estimator, recipe_name_for
from bike_rental.defs.resources.experiment import ExperimentConfig


class RandomForestConfig(dg.Config):
    """RandomForest hyperparameters (editable per-run in the Launchpad)."""

    n_estimators: int = 242
    max_depth: int | None = None
    random_state: int = 42
    n_jobs: int = -1


@dg.asset(group_name="models", io_manager_key="model_io", kinds={"sklearn"})
def rf_hourly(
    tree_dataset_hourly_train: pd.DataFrame,
    tree_dataset_hourly_val: pd.DataFrame,
    recipe_config: RecipeConfig,
    experiment_config: ExperimentConfig,
    config: RandomForestConfig,
) -> dg.MaterializeResult:

    model_type = "rf"
    estimator = build_estimator(model_type, config.model_dump())
    trainingResult = train_and_evaluate(
        tree_dataset_hourly_train, tree_dataset_hourly_val,
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
