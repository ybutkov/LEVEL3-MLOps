"""Hourly-demand model assets, generated from one factory.

The three model assets (linear / rf / hgb) share an identical body — train on a
recipe, log the run to MLflow, return a scored :class:`Candidate`. They differ
only in four things, passed to :func:`_model_asset`: the ``model_type``, its
per-model hyperparameter ``dg.Config``, and the train/val dataset assets it reads
(tree vs linear recipe). Same factory pattern as the raw loaders in ``raw.py``.

The per-model ``Config`` classes stay explicit: their fields *are* the meaningful
difference between models (the hyperparameter schema shown in the Launchpad).
"""

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


class LinearModelConfig(dg.Config):
    """LinearRegression hyperparameters (editable per-run in the Launchpad)."""

    fit_intercept: bool = True
    positive: bool = False


class RandomForestConfig(dg.Config):
    """RandomForest hyperparameters (editable per-run in the Launchpad)."""

    n_estimators: int = 242
    max_depth: int | None = None
    random_state: int = 42
    n_jobs: int = -1


class HGBModelConfig(dg.Config):
    """HistGradientBoosting hyperparameters (editable per-run in the Launchpad)."""

    learning_rate: float = 0.1
    max_iter: int = 300
    max_depth: int | None = None
    max_leaf_nodes: int = 31
    l2_regularization: float = 0.0
    random_state: int = 42


def _model_asset(*, model_type: str, config_cls: type[dg.Config], train_asset: str, val_asset: str):
    """Build a model asset that trains ``model_type`` on the given dataset assets.

    Parameters
    ----------
    model_type : str
        Catalog key selecting the estimator and recipe (e.g. ``"hgb"``).
    config_cls : type[dg.Config]
        Per-model hyperparameter config; its annotation drives the Launchpad schema.
    train_asset, val_asset : str
        Upstream split-dataset asset keys to train and validate on.
    """

    @dg.asset(
        name=f"{model_type}_hourly",
        group_name="models",
        kinds={"sklearn"},
        ins={"train_df": dg.AssetIn(train_asset), "val_df": dg.AssetIn(val_asset)},
    )
    def _asset(
        context: dg.AssetExecutionContext,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        recipe_config: RecipeConfig,
        experiment_config: ExperimentConfig,
        experiment_tracker: ExperimentTracker,
        data_commit: str,
        config: config_cls,
    ) -> dg.MaterializeResult:

        estimator = build_estimator(model_type, config.model_dump())
        dataset_config = DatasetConfig.from_recipe(recipe_config, recipe_name_for(model_type))
        trainingResult = train_and_evaluate(
            train_df, val_df,
            dataset_config, estimator,
            features=experiment_config.features,
        )
        logged = experiment_tracker.log_run(
            run_name=model_type,
            params=config.model_dump(),
            metrics=trainingResult.metrics,
            pipeline=trainingResult.pipeline,
            train_df=train_df[trainingResult.features + [dataset_config.target]],
            target=dataset_config.target,
            data_source=data_commit,
            dataset_name=f"{recipe_name_for(model_type)}_hourly_train",
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

    return _asset


linear_hourly = _model_asset(
    model_type="linear", config_cls=LinearModelConfig,
    train_asset="linear_dataset_hourly_train", val_asset="linear_dataset_hourly_val",
)
rf_hourly = _model_asset(
    model_type="rf", config_cls=RandomForestConfig,
    train_asset="tree_dataset_hourly_train", val_asset="tree_dataset_hourly_val",
)
hgb_hourly = _model_asset(
    model_type="hgb", config_cls=HGBModelConfig,
    train_asset="tree_dataset_hourly_train", val_asset="tree_dataset_hourly_val",
)
