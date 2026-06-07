import dagster as dg
import pandas as pd

from bike_rental.defs.assets.ml.recipes.builders import build_dataset
from bike_rental.defs.assets.ml.recipes.recipe_config import RecipeConfig
from bike_rental.defs.assets.ml.recipes.schema import DatasetConfig


@dg.asset(group_name="model_datasets", io_manager_key="csv_io", kinds={"pandas"})
def linear_dataset_hourly(
    hourly_total: pd.DataFrame, recipe_config: RecipeConfig
) -> dg.MaterializeResult:

    dataset_config = DatasetConfig.from_recipe(recipe_config, "linear")
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
