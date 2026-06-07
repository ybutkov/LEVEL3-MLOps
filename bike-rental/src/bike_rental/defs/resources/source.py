"""Resources for reading source CSV files.

`SourceResource` is the contract (``load_csv``) the raw assets depend on;
`SourceDirResource` is the local-directory implementation. A future
LakeFS-backed resource can subclass `SourceResource` and be swapped in by
binding it to the same ``source`` resource key — no asset changes needed.
"""

from pathlib import Path

import dagster as dg
import pandas as pd

from bike_rental.defs.io_utils import read_csv_or_fail


class SourceResource(dg.ConfigurableResource):
    """Contract for a source-data resource: load a named CSV into a DataFrame."""

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load the named source CSV. Implemented by concrete subclasses."""
        raise NotImplementedError


class SourceDirResource(SourceResource):
    """Read source CSV files from a local base directory."""

    base_path: str

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load a CSV from the resource's base directory by file name.

        Parameters
        ----------
        filename : str
            File name relative to ``base_path``.

        Returns
        -------
        pandas.DataFrame
            The loaded CSV (logs a warning if it has zero rows).
        """
        path = Path(self.base_path) / filename
        log = dg.get_dagster_logger()

        log.info("Reading source CSV: %s", path)
        df = read_csv_or_fail(path, not_found_msg=f"Source file not found: {path}")
        if df.empty:
            log.warning("Source CSV is empty (0 rows): %s", path)
        return df
