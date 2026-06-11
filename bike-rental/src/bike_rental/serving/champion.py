"""Load the served model from the MLflow registry by alias (never a local file)."""

from dataclasses import dataclass

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class ServedModel:
    """The model behind the serving alias, plus its provenance.

    Attributes
    ----------
    pipeline : Pipeline
        Self-contained sklearn pipeline (preprocessing + non-negative regressor),
        so serving feeds raw features and never re-applies transforms or clipping.
    feature_columns : list[str]
        Input columns in signature order — what a request must resolve to.
    version : str
        Registry version behind the alias.
    data_commit : str | None
        LakeFS commit the model trained on (from the source run's tag).
    """

    pipeline: Pipeline
    feature_columns: list[str]
    version: str
    data_commit: str | None


def load_served_model(tracking_uri: str, model_name: str, alias: str) -> ServedModel:
    """Load ``models:/{model_name}@{alias}`` with its signature columns and lineage.

    Parameters
    ----------
    tracking_uri : str
        MLflow tracking/registry server.
    model_name : str
        Registered model name.
    alias : str
        Registry alias to serve (e.g. ``"production"``).
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    uri = f"models:/{model_name}@{alias}"

    version = client.get_model_version_by_alias(model_name, alias)
    pipeline = mlflow.sklearn.load_model(uri)
    data_commit = client.get_run(version.run_id).data.tags.get("data_commit")
    feature_columns = mlflow.models.get_model_info(uri).signature.inputs.input_names()

    return ServedModel(
        pipeline=pipeline,
        feature_columns=feature_columns,
        version=version.version,
        data_commit=data_commit,
    )


class ChampionCache:
    """Hold the served model, re-downloading only when the registry alias moves.

    ``get`` does a cheap version check on the alias (a metadata call, no model
    download) and reloads the full model only when the version changed — so a
    newly promoted model is picked up automatically, without re-loading on every
    request.
    """

    def __init__(self, tracking_uri: str, model_name: str, alias: str):
        self._tracking_uri = tracking_uri
        self._model_name = model_name
        self._alias = alias
        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._model: ServedModel | None = None

    def _current_version(self) -> str:
        """Version currently behind the alias (cheap metadata call, no download)."""
        return self._client.get_model_version_by_alias(self._model_name, self._alias).version

    def get(self) -> ServedModel:
        """Return the served model, re-downloading only if the alias version changed."""
        if self._model is None or self._model.version != self._current_version():
            self._model = load_served_model(self._tracking_uri, self._model_name, self._alias)
        return self._model
