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
import mlflow.sklearn
from mlflow.models import infer_signature

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
        X_example: pd.DataFrame,
        tags: dict[str, str] | None = None,
    ) -> LoggedRun:
        """Record one training run. Implemented by concrete subclasses."""
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
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(self.tracking_uri)
        return MlflowClient()

    def log_run(
        self,
        *,
        run_name: str,
        params: dict,
        metrics: dict[str, float],
        pipeline: Pipeline,
        X_example: pd.DataFrame,
        tags: dict[str, str] | None = None,
    ) -> LoggedRun:

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            if tags:
                mlflow.set_tags(tags)
            mlflow.log_params(params)
            # MLflow metric keys disallow '/': rmse/mae -> rmse_mae
            mlflow.log_metrics({k.replace("/", "_"): v for k, v in metrics.items()})
            signature = infer_signature(X_example, pipeline.predict(X_example))
            info = mlflow.sklearn.log_model(
                pipeline,
                name=run_name,
                signature=signature,
                input_example=X_example,
                registered_model_name=self.registered_model,
            )
            return LoggedRun(run_id=run.info.run_id, model_version=info.registered_model_version)

    def _candidate_from_version(self, client, mv) -> Candidate:
        run = client.get_run(mv.run_id)
        return Candidate(
            version=mv.version,
            model_type=run.data.tags.get("model_type", "unknown"),
            metrics=run.data.metrics,
            run_id=mv.run_id,
        )

    def load_champion(self) -> Candidate | None:
        from mlflow.exceptions import MlflowException

        client = self._client()
        try:
            mv = client.get_model_version_by_alias(self.registered_model, self.champion_alias)
        except MlflowException:
            return None
        return self._candidate_from_version(client, mv)

    def set_champion(self, version: str) -> None:
        self._client().set_registered_model_alias(self.registered_model, self.champion_alias, version)
