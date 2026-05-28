"""Resource that reads source CSV files from a base directory."""

from pathlib import Path

import dagster as dg
import pandas as pd

from bike_rental.defs.io_utils import read_csv_or_fail


class BikeDataDirResource(dg.ConfigurableResource):
    """Dagster resource pointing at the directory of source CSV files."""

    base_path: str

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load a CSV from the resource's base directory by file name."""
        path = Path(self.base_path) / filename
        log = dg.get_dagster_logger()

        log.info("Reading source CSV: %s", path)
        df = read_csv_or_fail(path, not_found_msg=f"Source file not found: {path}")
        if df.empty:
            log.warning("Source CSV is empty (0 rows): %s", path)
        return df
