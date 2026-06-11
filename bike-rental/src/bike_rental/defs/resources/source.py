"""Resources for reading source CSV files.

`SourceResource` is the contract (``load_csv``) the raw assets depend on;
`SourceDirResource` is the local-directory implementation. A future
LakeFS-backed resource can subclass `SourceResource` and be swapped in by
binding it to the same ``source`` resource key — no asset changes needed.
"""

from pathlib import Path

import dagster as dg
import pandas as pd
from lakefs_spec import LakeFSFileSystem

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


class LakeFSSourceResource(SourceResource):
    """Read raw source CSVs from a LakeFS repo (pinned to ``ref``) via lakefs-spec.

    Reads ``{repo}/{ref}/{raw_prefix}/{filename}``. Versioned-storage counterpart
    of :class:`SourceDirResource`; swap by binding it to the ``source`` key.
    """

    host: str
    repo: str
    ref: str
    raw_prefix: str
    access_key: str
    secret_key: str

    def load_csv(self, filename: str) -> pd.DataFrame:
        
        lake_fs = LakeFSFileSystem(host=self.host, username=self.access_key, password=self.secret_key)
        path = f"{self.repo}/{self.ref}/{self.raw_prefix}/{filename}"
        log = dg.get_dagster_logger()

        log.info("Reading source CSV from LakeFS: %s", path)
        # pre_sign=False: stream via the LakeFS server
        with lake_fs.open(path, "rb", pre_sign=False) as f:
            return pd.read_csv(f)
