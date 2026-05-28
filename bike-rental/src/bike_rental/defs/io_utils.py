"""Shared helpers for reading CSV files with clear failures."""

from pathlib import Path

import dagster as dg
import pandas as pd
from pandas.errors import EmptyDataError, ParserError


def read_csv_or_fail(path: Path, *, not_found_msg: str) -> pd.DataFrame:
    """Read a CSV, translating I/O errors into dg.Failure with context.

    `not_found_msg` is set by the caller because a missing file means different
    things in different layers (missing source data vs. unmaterialized upstream),
    with different causes and fixes.
    """
    log = dg.get_dagster_logger()
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as e:
        raise dg.Failure(
            description=not_found_msg,
            metadata={"path": str(path)},
        ) from e
    except (EmptyDataError, ParserError) as e:
        raise dg.Failure(
            description=f"Failed to parse CSV {path}: {e}",
            metadata={"path": str(path)},
        ) from e

    log.info("Loaded %d rows x %d cols from %s", len(df), df.shape[1], path)
    return df
