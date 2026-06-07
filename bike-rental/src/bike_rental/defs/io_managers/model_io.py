"""IO manager that persists model assets as joblib files on disk."""

from pathlib import Path

import dagster as dg
import joblib

from bike_rental.defs.io_utils import read_model_or_fail


class ModelIOManager(dg.IOManager):
    """Store and load fitted models / sklearn Pipelines as ``.joblib`` files."""

    def __init__(self, base_dir: str):
        """Create the manager pointed at a base directory."""
        self.base_dir = Path(base_dir)

    def _path(self, context) -> Path:
        """Return the on-disk model path for an asset (``<base>/<asset>.joblib``)."""
        return self.base_dir / f"{context.asset_key.path[-1]}.joblib"

    def handle_output(self, context: dg.OutputContext, obj):
        """Serialize the asset's fitted model to a ``.joblib`` file.

        Parameters
        ----------
        context : dagster.OutputContext
            Output context; the asset key determines the file name.
        obj : object or dagster.MaterializeResult
            Fitted model / Pipeline to persist (unwrapped from a MaterializeResult
            if needed).

        Raises
        ------
        dagster.Failure
            If the model cannot be written.
        """
        log = dg.get_dagster_logger()
        model = obj.value if isinstance(obj, dg.MaterializeResult) else obj
        path = self._path(context)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            joblib.dump(model, path)
        except OSError as e:
            raise dg.Failure(
                description=f"Failed to write model {path}: {e}",
                metadata={"path": str(path)},
            ) from e

        log.info("Wrote model")

    def load_input(self, context: dg.InputContext):
        """Load an upstream model asset from its ``.joblib`` file.

        Parameters
        ----------
        context : dagster.InputContext
            Input context; the upstream asset key determines the file name.

        Returns
        -------
        object
            The deserialized model / Pipeline.
        """
        path = self._path(context)
        return read_model_or_fail(
            path,
            not_found_msg=f"Intermediate model not found: {path} (upstream not materialized?)",
        )
