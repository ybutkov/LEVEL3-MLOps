"""Per-model input dataset for tree models, assembled by config-driven steps.

Trees need no stateless encoding, so the default recipe is a single `select`
step picking the feature columns + target (+ the time key for the split). The
recipe is declarative (visible in config, logged in metadata); new engineered
steps can be added later without touching this asset.
"""

import dagster as dg
import pandas as pd
from pydantic import Field

from bike_rental.defs.assets.ml.recipe.apply import build_dataset
from bike_rental.defs.assets.ml.recipe.loader import load_recipe
from bike_rental.defs.assets.ml.recipe.schema import DatasetConfig, DatasetStep

# Default recipe (target + steps) from config/recipes.yaml (fallback if absent).
_TREE_RECIPE = load_recipe("tree")


class TreeDatasetConfig(DatasetConfig):
    """`DatasetConfig` preloaded with the tree recipe (target + steps)."""

    target: str = _TREE_RECIPE["target"]
    steps: list[DatasetStep] = Field(default=_TREE_RECIPE["steps"], validate_default=True)


@dg.asset(group_name="model_datasets", io_manager_key="csv_io", kinds={"pandas"})
def tree_dataset_hourly(
    hourly_total: pd.DataFrame, config: TreeDatasetConfig
) -> dg.MaterializeResult:
    """Assemble the tree model's input table by applying the configured steps.

    Parameters
    ----------
    hourly_total : pandas.DataFrame
        City-wide hourly base dataset.
    config : TreeDatasetConfig
        Recipe (target + ordered steps); for trees this is typically a single
        ``select`` step, since no encoding is needed.

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
