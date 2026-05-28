"""IO manager that persists assets as CSV files on disk."""

from pathlib import Path

import dagster as dg
import pandas as pd

from bike_rental.defs.io_utils import read_csv_or_fail


class CSVIOManager(dg.IOManager):
    """Store and load DataFrame assets as CSV files in a base directory."""

    def __init__(self, base_dir: str):
        """Create the manager and ensure the base directory exists."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, context) -> Path:
        return self.base_dir / f"{context.asset_key.path[-1]}.csv"

    def handle_output(self, context: dg.OutputContext, obj):
        """Write the asset's DataFrame to a CSV file."""
        log = dg.get_dagster_logger()
        df = obj.value if isinstance(obj, dg.MaterializeResult) else obj
        path = self._path(context)

        try:
            df.to_csv(path, index=False)
        except OSError as e:
            raise dg.Failure(
                description=f"Failed to write CSV {path}: {e}",
                metadata={"path": str(path)},
            ) from e

        log.info("Wrote %d rows x %d cols to %s", len(df), df.shape[1], path)

    def load_input(self, context: dg.InputContext) -> pd.DataFrame:
        """Read an upstream asset's CSV file into a DataFrame."""
        path = self._path(context)
        return read_csv_or_fail(
            path,
            not_found_msg=f"Intermediate CSV not found: {path} (upstream not materialized?)",
        )
