

from pathlib import Path

import dagster as dg
import joblib

from bike_rental.defs.io_utils import read_model_or_fail

class ModelIOManager(dg.IOManager):

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def _path(self, context) -> Path:
        return self.base_dir / f"{context.asset_key.path[-1]}.joblib"

    def handle_output(self, context: dg.OutputContext, obj):
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
        path = self._path(context)
        return read_model_or_fail(
            path,
            not_found_msg=f"Intermediate model not found: {path} (upstream not materialized?)",
        )
