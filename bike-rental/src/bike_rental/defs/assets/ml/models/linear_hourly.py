import dagster as dg
import pandas as pd

from bike_rental.defs.assets.ml.recipes.recipe_config import RecipeConfig
from bike_rental.defs.assets.ml.recipes.schema import DatasetConfig
from bike_rental.defs.assets.ml.models.training import train_and_evaluate
from bike_rental.defs.assets.ml.models.catalog import build_estimator, recipe_name_for
from bike_rental.defs.resources.experiment import ExperimentConfig
from bike_rental.defs.resources.experiment_tracker import ExperimentTracker
from bike_rental.defs.assets.ml.registry import Candidate


class LinearModelConfig(dg.Config):
    """LinearRegression hyperparameters (editable per-run in the Launchpad)."""

    fit_intercept: bool = True
    positive: bool = False

@dg.asset(group_name="models", kinds={"sklearn"})
def linear_hourly(
    context: dg.AssetExecutionContext,
    linear_dataset_hourly_train: pd.DataFrame,
    linear_dataset_hourly_val: pd.DataFrame,
    recipe_config: RecipeConfig,
    experiment_config: ExperimentConfig,
    experiment_tracker: ExperimentTracker,
    config: LinearModelConfig,
) -> dg.MaterializeResult:
    model_type = "linear"
    estimator = build_estimator(model_type, config.model_dump())
    dataset_config = DatasetConfig.from_recipe(recipe_config, recipe_name_for(model_type))
    trainingResult = train_and_evaluate(
        linear_dataset_hourly_train, linear_dataset_hourly_val,
        dataset_config, estimator,
        features=experiment_config.features,
    )
    logged = experiment_tracker.log_run(
        run_name=model_type,
        params=config.model_dump(),
        metrics=trainingResult.metrics,
        pipeline=trainingResult.pipeline,
        X_example=linear_dataset_hourly_train[trainingResult.features].head(),
        tags={"model_type": model_type,
              "recipe": recipe_name_for(model_type),
              "dagster_run_id": context.run_id
        },
    )
    candidate = Candidate(
        version=logged.model_version,
        model_type=model_type,
        metrics=trainingResult.metrics,
        run_id=logged.run_id,
    )
    return dg.MaterializeResult(
        value=candidate,
        metadata={
            **trainingResult.metadata,
            "model_type": dg.MetadataValue.text(model_type),
            "params": dg.MetadataValue.json(config.model_dump()),
            "features": dg.MetadataValue.json(trainingResult.features),
            "mlflow_run_id": dg.MetadataValue.text(logged.run_id),
            "model_version": dg.MetadataValue.text(str(logged.model_version)),
        },
    )
