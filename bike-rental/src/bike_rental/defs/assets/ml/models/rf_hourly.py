import dagster as dg
import pandas as pd

from bike_rental.defs.assets.ml.recipes.recipe_config import RecipeConfig
from bike_rental.defs.assets.ml.recipes.schema import DatasetConfig
from bike_rental.defs.assets.ml.models.training import train_and_evaluate
from bike_rental.defs.assets.ml.models.catalog import build_estimator, recipe_name_for
from bike_rental.defs.resources.experiment import ExperimentConfig
from bike_rental.defs.resources.experiment_tracker import ExperimentTracker
from bike_rental.defs.assets.ml.registry import Candidate
from bike_rental.defs.utils.git_operations import get_git_commit


class RandomForestConfig(dg.Config):
    """RandomForest hyperparameters (editable per-run in the Launchpad)."""

    n_estimators: int = 242
    max_depth: int | None = None
    random_state: int = 42
    n_jobs: int = -1


@dg.asset(group_name="models", kinds={"sklearn"})
def rf_hourly(
    context: dg.AssetExecutionContext,
    tree_dataset_hourly_train: pd.DataFrame,
    tree_dataset_hourly_val: pd.DataFrame,
    recipe_config: RecipeConfig,
    experiment_config: ExperimentConfig,
    experiment_tracker: ExperimentTracker,
    data_commit: str,
    config: RandomForestConfig,
) -> dg.MaterializeResult:

    model_type = "rf"
    estimator = build_estimator(model_type, config.model_dump())
    dataset_config = DatasetConfig.from_recipe(recipe_config, recipe_name_for(model_type))
    trainingResult = train_and_evaluate(
        tree_dataset_hourly_train, tree_dataset_hourly_val,
        dataset_config, estimator,
        features=experiment_config.features,
    )
    logged = experiment_tracker.log_run(
        run_name=model_type,
        params=config.model_dump(),
        metrics=trainingResult.metrics,
        pipeline=trainingResult.pipeline,
        X_example=tree_dataset_hourly_train[trainingResult.features].head(),
        tags={"model_type": model_type,
              "recipe": recipe_name_for(model_type),
              "dagster_run_id": context.run_id,
              "data_commit": data_commit,
              "git_commit": get_git_commit(),
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
