"""Experiment-tracking resource: the DI seam for recording a training run.

``ExperimentTracker`` is the contract (``log_run``) the model assets depend on;
``MlflowExperimentTracker`` is the MLflow implementation. A future impl (e.g. a
no-op for offline runs/tests) subclasses the contract and is swapped in by
binding it to the ``experiment_tracker`` resource key — no asset changes needed
(same pattern as ``SourceResource``).
"""

from dataclasses import dataclass

import dagster as dg
import pandas as pd
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.data
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from bike_rental.defs.assets.ml.registry import Candidate


@dataclass(frozen=True)
class LoggedRun:
    """Handle to a recorded run, returned to the asset for provenance.

    Parameters
    ----------
    run_id : str
        Tracking run id (``"noop"`` when nothing was recorded).
    model_version : str | None
        Registry version created for this run, if the model was registered.
    """

    run_id: str
    model_version: str | None = None


class ExperimentTracker(dg.ConfigurableResource):
    """Contract for run tracking: record params, metrics, and the fitted model."""

    def log_run(
        self,
        *,
        run_name: str,
        params: dict,
        metrics: dict[str, float],
        pipeline: Pipeline,
        train_df: pd.DataFrame,
        target: str,
        data_source: str,
        dataset_name: str,
        tags: dict[str, str] | None = None,
    ) -> LoggedRun:
        """Record one training run. Implemented by concrete subclasses.

        ``train_df`` is the active feature columns plus the ``target`` column;
        it is logged as the run's input dataset (its schema captures *which*
        columns were active, its digest captures the data) and its feature
        columns drive the model signature. ``data_source`` is the LakeFS commit
        id the data came from; ``dataset_name`` names the dataset in MLflow.
        """
        raise NotImplementedError

    def load_champion(self) -> Candidate | None:
        """The version currently under the champion alias, or ``None`` if unset."""
        raise NotImplementedError

    def set_champion(self, version: str) -> None:
        """Point the champion alias at ``version`` (the promotion write)."""
        raise NotImplementedError


class MlflowExperimentTracker(ExperimentTracker):
    """Log params/metrics and register the self-contained pipeline to MLflow.

    The whole sklearn ``Pipeline`` (preprocessing + ``NonNegativeRegressor``) is
    logged with an inferred signature and registered as a new version under
    ``registered_model`` — promotion to champion stays a separate step.

    Parameters
    ----------
    tracking_uri : str
        MLflow tracking/registry server (stand-specific).
    experiment_name : str
        Experiment that collects the runs.
    registered_model : str
        Registry model name new versions are logged under.
    champion_alias : str, default="champion"
        Registry alias marking the served model.
    """

    tracking_uri: str
    experiment_name: str
    registered_model: str
    champion_alias: str = "champion"

    def _client(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        return MlflowClient()

    def log_run(
        self,
        *,
        run_name: str,
        params: dict,
        metrics: dict[str, float],
        pipeline: Pipeline,
        train_df: pd.DataFrame,
        target: str,
        data_source: str,
        dataset_name: str,
        tags: dict[str, str] | None = None,
    ) -> LoggedRun:

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            if tags:
                mlflow.set_tags(tags)
            mlflow.log_params(params)
            # MLflow metric keys disallow '/': rmse/mae -> rmse_mae
            # TODO Should write keys rmse_mae instead of rmse/mae
            mlflow.log_metrics({k.replace("/", "_"): v for k, v in metrics.items()})
            # Record the training data as the run's input dataset: schema = active
            # feature set (+ target), digest = content, source = LakeFS commit id.
            dataset = mlflow.data.from_pandas(
                train_df, source=data_source, targets=target, name=dataset_name
            )
            mlflow.log_input(dataset, context="training")
            example = train_df.drop(columns=[target]).head()
            signature = infer_signature(example, pipeline.predict(example))
            info = mlflow.sklearn.log_model(
                pipeline,
                name=run_name,
                signature=signature,
                input_example=example,
                registered_model_name=self.registered_model,
            )
            return LoggedRun(run_id=run.info.run_id, model_version=info.registered_model_version)

    def _candidate_from_version(self, client, model_version) -> Candidate:
        run = client.get_run(model_version.run_id)
        return Candidate(
            version=model_version.version,
            model_type=run.data.tags.get("model_type", "unknown"),
            metrics=run.data.metrics,
            run_id=model_version.run_id,
        )

    def load_champion(self) -> Candidate | None:

        client = self._client()
        try:
            champion = client.get_model_version_by_alias(self.registered_model, self.champion_alias)
        except MlflowException:
            return None
        return self._candidate_from_version(client, champion)

    def set_champion(self, version: str) -> None:
        self._client().set_registered_model_alias(self.registered_model, self.champion_alias, version)
