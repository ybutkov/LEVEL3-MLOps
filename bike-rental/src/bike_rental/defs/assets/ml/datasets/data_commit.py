"""Publish the run's train/val/test splits to LakeFS and return their commit id.

Runs after the split assets. Uploads every split to LakeFS on a per-run branch,
commits, and merges to ``main``; the returned commit id is the data version. The
model assets take it as input and log it (``data_commit``) so each model points
back at the exact data it trained on. The local CSVs (read for training) and this
LakeFS copy are byte-identical for the run.
"""

import dagster as dg
import pandas as pd

from bike_rental.defs.resources.lakefs import LakeFSVersioningResource
from bike_rental.defs.utils.git_operations import get_git_commit


@dg.asset(group_name="data_versioning", kinds={"lakefs"})
def data_commit(
    context: dg.AssetExecutionContext,
    lakefs: LakeFSVersioningResource,
    linear_dataset_hourly_train: pd.DataFrame,
    linear_dataset_hourly_val: pd.DataFrame,
    linear_dataset_hourly_test: pd.DataFrame,
    tree_dataset_hourly_train: pd.DataFrame,
    tree_dataset_hourly_val: pd.DataFrame,
    tree_dataset_hourly_test: pd.DataFrame,
) -> str:
    """Publish the run's six splits to LakeFS; return the snapshot commit id."""
    frames = {
        "datasets/hourly/linear/train.csv": linear_dataset_hourly_train,
        "datasets/hourly/linear/val.csv": linear_dataset_hourly_val,
        "datasets/hourly/linear/test.csv": linear_dataset_hourly_test,
        "datasets/hourly/tree/train.csv": tree_dataset_hourly_train,
        "datasets/hourly/tree/val.csv": tree_dataset_hourly_val,
        "datasets/hourly/tree/test.csv": tree_dataset_hourly_test,
    }
    files = {path: df.to_csv(index=False).encode() for path, df in frames.items()}

    commit_id = lakefs.publish(
        files=files,
        branch=f"run-{context.run_id}",
        message=f"dataset snapshot · run {context.run_id}",
        metadata={"dagster_run_id": context.run_id, "git_commit": get_git_commit()},
    )

    context.add_output_metadata(
        {
            "data_commit": dg.MetadataValue.text(commit_id),
            "files": dg.MetadataValue.json(list(frames)),
        }
    )
    return commit_id
