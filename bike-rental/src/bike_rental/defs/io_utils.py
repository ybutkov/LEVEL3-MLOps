"""Shared helpers for reading CSV files with clear failures."""

from pathlib import Path

import dagster as dg
import joblib
import pandas as pd
from pandas.errors import EmptyDataError, ParserError


def read_csv_or_fail(path: Path, *, not_found_msg: str) -> pd.DataFrame:
    """Read a CSV, translating I/O errors into ``dagster.Failure`` with context.

    Parameters
    ----------
    path : pathlib.Path
        Path to the CSV file.
    not_found_msg : str
        Message used when the file is missing — set by the caller because a
        missing file means different things in different layers (missing source
        data vs. unmaterialized upstream).

    Returns
    -------
    pandas.DataFrame
        The loaded CSV.

    Raises
    ------
    dagster.Failure
        If the file is missing or cannot be parsed.
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

def read_model_or_fail(path: Path, *, not_found_msg: str):
    """Load a joblib-serialized model, translating I/O errors into ``dagster.Failure``.

    Parameters
    ----------
    path : pathlib.Path
        Path to the ``.joblib`` model file.
    not_found_msg : str
        Message used when the file is missing.

    Returns
    -------
    object
        The deserialized model (typically an sklearn ``Pipeline``).

    Raises
    ------
    dagster.Failure
        If the file is missing or cannot be loaded.
    """
    log = dg.get_dagster_logger()
    try:
        model = joblib.load(path)
    except FileNotFoundError as e:
        raise dg.Failure(
            description=not_found_msg,
            metadata={"path": str(path)},
        ) from e
    except Exception as e:
        raise dg.Failure(
            description=f"Failed to load model {path}: {e}",
            metadata={"path": str(path)},
        ) from e

    log.info("Loaded model from %s", path)
    return model
