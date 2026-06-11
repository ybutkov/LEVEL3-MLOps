"""The single shared feature table asset all models train on."""

import dagster as dg
import pandas as pd

from bike_rental.defs.assets.ml.recipes.builders import build_dataset
from bike_rental.defs.assets.ml.recipes.recipe_config import RecipeConfig
from bike_rental.defs.assets.ml.recipes.schema import DatasetConfig


@dg.asset(group_name="model_datasets", io_manager_key="csv_io", kinds={"pandas"})
def feature_rentals_hourly(
    hourly_total: pd.DataFrame, recipe_config: RecipeConfig
) -> dg.MaterializeResult:
    """Build the single feature table all models train on.

    Applies the shared ``dataset`` recipe (the ``select`` step) to ``hourly_total``
    — picking the safe feature columns and dropping the leakage ones. Per-model
    representation (cyclic/scale for linear, raw for trees) is NOT baked here; it
    lives in each model's training Pipeline, so one stored dataset serves all.
    """
    dataset_config = DatasetConfig.from_recipe(recipe_config, "dataset")
    df = build_dataset(hourly_total, dataset_config)

    columns = [c for c in df.columns if c != "datetime_hourly"]
    return dg.MaterializeResult(
        value=df,
        metadata={
            "row_count": dg.MetadataValue.int(len(df)),
            "columns": dg.MetadataValue.text(", ".join(columns)),
            "steps": dg.MetadataValue.json(
                [s.model_dump(exclude_none=True) for s in dataset_config.steps]
            ),
            "preview": dg.MetadataValue.md(df.head().to_markdown()),
        },
    )
