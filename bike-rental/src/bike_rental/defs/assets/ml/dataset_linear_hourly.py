"""Per-model input dataset for the linear model, assembled by config-driven steps.

The column selection + stateless feature engineering are described declaratively
as an ordered list of steps (`DatasetConfig`) and applied by `build_dataset`, so
the exact recipe is visible in config and logged in metadata. Stateful steps
(scaling) are deliberately absent — they live in the model's Pipeline, after the
split, to avoid leakage. Defaults below reproduce the previous hard-coded output.
"""

import dagster as dg
import pandas as pd
from pydantic import Field

from bike_rental.defs.assets.ml.recipe.apply import build_dataset
from bike_rental.defs.assets.ml.recipe.loader import load_recipe
from bike_rental.defs.assets.ml.recipe.schema import DatasetConfig, DatasetStep

# Default recipe (target + steps) from config/recipes.yaml (fallback if absent).
_LINEAR_RECIPE = load_recipe("linear")


class LinearDatasetConfig(DatasetConfig):
    """`DatasetConfig` preloaded with the linear recipe (target + steps)."""

    target: str = _LINEAR_RECIPE["target"]
    steps: list[DatasetStep] = Field(default=_LINEAR_RECIPE["steps"], validate_default=True)


@dg.asset(group_name="model_datasets", io_manager_key="csv_io", kinds={"pandas"})
def linear_dataset_hourly(
    hourly_total: pd.DataFrame, config: LinearDatasetConfig
) -> dg.MaterializeResult:
    """Assemble the linear model's input table by applying the configured steps.

    Parameters
    ----------
    hourly_total : pandas.DataFrame
        City-wide hourly base dataset.
    config : LinearDatasetConfig
        Recipe (target + ordered stateless steps). Stateful steps are left for
        the model's Pipeline and not applied here.

    Returns
    -------
    dagster.MaterializeResult
        The assembled feature table with row-count, column list, recipe steps
        and a preview in metadata.
    """
    df = build_dataset(hourly_total, config)

    columns = [c for c in df.columns if c != "datetime_hourly"]
    return dg.MaterializeResult(
        value=df,
        metadata={
            "row_count": dg.MetadataValue.int(len(df)),
            "columns": dg.MetadataValue.text(", ".join(columns)),
            "steps": dg.MetadataValue.json(
                [s.model_dump(exclude_none=True) for s in config.steps]
            ),
            "preview": dg.MetadataValue.md(df.head().to_markdown()),
        },
    )
